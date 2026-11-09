"""为实验报告补绘启发式紧度、逐层剪枝和宝藏决策质量图。"""

from __future__ import annotations

import csv
from heapq import heappop, heappush
from math import inf
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from astar import _weight, search
from extension_experiments import alternating_terrain, ordered_tree
from innovation_experiments import striped_grid
from minimax import TreeNode, _first_leaf
from plot_results import BLUE, ORANGE, read_csv, style
from stage2 import generated_tree


ROOT = Path(__file__).parent
OUTPUT = ROOT / "results" / "final_figures"
DATA = OUTPUT / "analysis_data"


def save_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique(rows: list[dict[str, str]], **filters: str | int) -> dict[str, str]:
    selected = [row for row in rows if all(row.get(key) == str(value) for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"无法唯一定位既有实验记录：{filters}，得到 {len(selected)} 行")
    return selected[0]


def neighbors(r: int, c: int, rows: int, cols: int):
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def true_remaining_cost(grid: list[list], goal: tuple[int, int]) -> list[list[float]]:
    """在反向图上运行 Dijkstra；正向进入 E 的边费用为零。"""
    rows, cols = len(grid), len(grid[0])
    distances = [[inf] * cols for _ in range(rows)]
    distances[goal[0]][goal[1]] = 0
    pending: list[tuple[float, int, int]] = [(0, *goal)]
    while pending:
        cost, r, c = heappop(pending)
        if cost != distances[r][c]:
            continue
        # 反向放松 predecessor -> (r,c)：正向代价只取决于被进入的格子。
        incoming_cost = 0 if (r, c) == goal else _weight(grid[r][c])
        for pr, pc in neighbors(r, c, rows, cols):
            if grid[pr][pc] == "X":
                continue
            proposed = cost + incoming_cost
            if proposed < distances[pr][pc]:
                distances[pr][pc] = proposed
                heappush(pending, (proposed, pr, pc))
    return distances


def axis_heuristic(grid: list[list], goal: tuple[int, int]):
    """复现 innovations.weighted_axis_lower_bound 中的逐格下界公式。"""
    rows, cols = len(grid), len(grid[0])
    row_min = [min((_weight(cell) for cell in row if cell != "X"), default=0) for row in grid]
    col_min = [min((_weight(grid[r][c]) for r in range(rows) if grid[r][c] != "X"), default=0)
               for c in range(cols)]
    row_prefix, col_prefix = [0], [0]
    for cost in row_min:
        row_prefix.append(row_prefix[-1] + cost)
    for cost in col_min:
        col_prefix.append(col_prefix[-1] + cost)

    def estimate(point: tuple[int, int]) -> int:
        r, c = point
        gr, gc = goal
        row_cost = (row_prefix[gr + 1] - row_prefix[r + 1] if r < gr
                    else row_prefix[r] - row_prefix[gr])
        col_cost = (col_prefix[gc + 1] - col_prefix[c + 1] if c < gc
                    else col_prefix[c] - col_prefix[gc])
        return max(row_cost + col_cost - _weight(grid[gr][gc]), 0)

    return estimate


def figure_b(innovation: list[dict[str, str]], extension: list[dict[str, str]]) -> Path:
    settings = [
        ("条纹 25×25", "striped_25x25", striped_grid(), innovation),
        ("交错 25×25", "Q2-02_alternating_25", alternating_terrain(), extension),
    ]
    records: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    for ax, (label, case, grid, source) in zip(axes, settings):
        goal = (len(grid) - 1, len(grid[0]) - 1)
        exact = true_remaining_cost(grid, goal)
        axis_h = axis_heuristic(grid, goal)
        minimum = min(_weight(cell) for row in grid for cell in row if cell != "X")
        global_h_values, axis_h_values, exact_values = [], [], []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == "X" or exact[r][c] == inf:
                    continue
                distance = abs(goal[0] - r) + abs(goal[1] - c)
                global_h = minimum * max(distance - 1, 0)
                axis_value = axis_h((r, c))
                actual = int(exact[r][c])
                if global_h > actual or axis_value > actual:
                    raise AssertionError(f"下界高估真实剩余代价：{case}, {(r, c)}")
                records.append(dict(case=case, row=r, col=c, true_remaining=actual,
                                    h_zero=0, h_global=global_h, h_axis=axis_value,
                                    goal_free=True))
                if actual > 0:
                    exact_values.append(actual)
                    global_h_values.append(global_h)
                    axis_h_values.append(axis_value)

        # 使用已有扩展节点数据核对地图构造和复制的下界公式。
        base = unique(source, case=case, algorithm="A*_global_min")
        improved = unique(source, case=case, algorithm="A*_axis_min")
        global_search = search(grid, (0, 0), goal, weighted=True, free_goal=True)
        axis_search = search(grid, (0, 0), goal, weighted=True, free_goal=True,
                             heuristic=lambda point, destination: axis_h(point), tie_break="deeper")
        if exact[0][0] != global_search.cost or axis_search.cost != global_search.cost:
            raise AssertionError(f"反向最短路与正向 A* 的起点代价不一致：{case}")
        if (global_search.expanded, axis_search.expanded) != (int(base["expanded"]), int(improved["expanded"])):
            raise AssertionError(f"场景或启发值与已有 CSV 不一致：{case}")

        global_tightness = median(h / cost for h, cost in zip(global_h_values, exact_values))
        axis_tightness = median(h / cost for h, cost in zip(axis_h_values, exact_values))
        ax.scatter(exact_values, global_h_values, s=17, alpha=0.35, color=BLUE,
                   label=f"全图最低费下界（中位紧度 {global_tightness:.2f}）")
        ax.scatter(exact_values, axis_h_values, s=24, alpha=0.42, color=ORANGE, marker="o",
                   facecolors="none", linewidths=0.9,
                   label=f"行列下界（中位紧度 {axis_tightness:.2f}）")
        limit = max(exact_values) * 1.04
        ax.plot([0, limit], [0, limit], color="#5C6470", linestyle="--", linewidth=1,
                label="真实代价＝启发值")
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="真实最低剩余代价", ylabel="启发函数估计值",
               title=f"{label}｜进入终点 E 免费")
        ax.grid(color="#E5EAF0", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.93)
        if all(a == b for a, b in zip(global_h_values, axis_h_values)):
            ax.text(0.97, 0.04, "两种下界逐格相同", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color="#555F6B")
    fig.suptitle("题二：启发值与真实最低剩余代价（两张固定 25×25 地图）")
    path = OUTPUT / "图11_加权启发函数紧度检验.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    save_csv(DATA / "task2_heuristic_tightness.csv",
             ("case", "row", "col", "true_remaining", "h_zero", "h_global", "h_axis", "goal_free"), records)
    return path


def trace_tree(root: TreeNode, *, pruning: bool, first_leaf: bool = False) -> tuple[int, list[int], int]:
    """按 minimax._solve 的 MAX/MIN 和 alpha-beta 更新规则记录逐层访问。"""
    counts = [0] * 7
    pruned = 0

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float, level: int) -> int:
        nonlocal pruned
        counts[level] += 1
        if not node.children:
            if node.value is None:
                raise ValueError("叶子缺少数值")
            return node.value
        children = sorted(node.children, key=_first_leaf, reverse=maximizing) if first_leaf else node.children
        best = -inf if maximizing else inf
        for index, child in enumerate(children):
            value = visit(child, not maximizing, alpha, beta, level + 1)
            best = max(best, value) if maximizing else min(best, value)
            if pruning:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    pruned += len(children) - index - 1
                    break
        return int(best)

    return visit(root, True, -inf, inf, 0), counts, pruned


def figure_c(stage: list[dict[str, str]], extension: list[dict[str, str]]) -> Path:
    original = generated_tree(depth=6, branching=4, seed=2026)
    favorable, favorable_value = ordered_tree(original, True, True)
    unfavorable, unfavorable_value = ordered_tree(original, True, False)
    if favorable_value != -36 or unfavorable_value != -36:
        raise AssertionError("事后排序所得根值与现有实验不一致")
    settings = [
        ("Minimax", original, False, False, unique(stage, case="full_tree_d6_b4", algorithm="Minimax")),
        ("Alpha-Beta 原顺序", original, True, False,
         unique(stage, case="full_tree_d6_b4", algorithm="Alpha-Beta", order="left_to_right")),
        ("Alpha-Beta 首叶排序", original, True, True,
         unique(stage, case="full_tree_d6_b4", algorithm="Alpha-Beta", order="first_leaf")),
        ("Alpha-Beta 事后有利", favorable, True, False,
         unique(extension, case="Q3-02_favorable", algorithm="Alpha-Beta", depth=6)),
        ("Alpha-Beta 事后不利", unfavorable, True, False,
         unique(extension, case="Q3-03_unfavorable", algorithm="Alpha-Beta", depth=6)),
    ]
    records: list[dict] = []
    matrix = []
    labels = []
    for label, tree, pruning, first_leaf, expected in settings:
        result, counts, pruned = trace_tree(tree, pruning=pruning, first_leaf=first_leaf)
        if (result, sum(counts), counts[-1], pruned) != (int(expected["result"]), int(expected["visited"]),
                                                       int(expected["leaves"]), int(expected["pruned"])):
            raise AssertionError(f"逐层追踪与已有 CSV 不一致：{label}")
        labels.append(label)
        matrix.append(counts)
        for depth, count in enumerate(counts):
            records.append(dict(strategy=label, tree_seed=2026, tree_depth=6, branching=4,
                                level=depth, visited=count, possible=4 ** depth,
                                root_value=result, total_visited=sum(counts), pruned_direct=pruned))
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10.8, 5.1), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues", norm=LogNorm(vmin=1, vmax=4096), aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = int(values[i, j])
            ax.text(j, i, f"{val}", ha="center", va="center", fontsize=9,
                    color="white" if val >= 200 else "#1F2937")
    ax.set_xticks(range(7), [str(i) for i in range(7)])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("从根节点起的层数（第 6 层为叶子）")
    ax.set_ylabel("搜索策略")
    ax.set_title("题三：同一棵树的逐层实际访问数｜深度 6、分支 4、seed=2026")
    fig.colorbar(image, ax=ax, label="实际访问数（对数色阶）", shrink=0.91)
    fig.text(0.5, -0.065, "事后有利／不利顺序使用真实子树值重排；构造成本不计入，不能当作实用排序器",
             ha="center", fontsize=9, color="#555F6B")
    path = OUTPUT / "图12_博弈树逐层剪枝效果.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    save_csv(DATA / "task3_depth_trace.csv",
             ("strategy", "tree_seed", "tree_depth", "branching", "level", "visited", "possible",
              "root_value", "total_visited", "pruned_direct"), records)
    return path


def figure_d(heuristic: list[dict[str, str]]) -> Path:
    cases = [("pdf_treasure", "实验要求的四宝藏地图", [1, 2, 3, 4]),
             ("six_treasure_grid", "固定六宝藏抽象地图", [1, 2, 3, 4, 6])]
    algorithms = [("current_score", "当前收益差"), ("future_estimate", "未来收益估值")]
    all_regrets = [int(row["regret"]) for row in heuristic if row["case"] in {case for case, _, _ in cases}]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.1), constrained_layout=True)
    for ax, (case, label, depths) in zip(axes, cases):
        matrix = np.zeros((len(algorithms), len(depths)), dtype=int)
        targets = [[""] * len(depths) for _ in algorithms]
        for i, (algorithm, _) in enumerate(algorithms):
            for j, depth in enumerate(depths):
                row = unique(heuristic, case=case, algorithm=algorithm, depth=depth)
                matrix[i, j] = int(row["regret"])
                targets[i][j] = row["best_target"]
                if int(row["full_optimal_value"]) - int(row["realized_value"]) != matrix[i, j]:
                    raise AssertionError(f"完整收益损失不一致：{case} {algorithm} H={depth}")
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(2, max(all_regrets)), aspect="auto")
        for i in range(len(algorithms)):
            for j in range(len(depths)):
                ax.text(j, i, f"{targets[i][j]}\n损失 {matrix[i, j]}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white" if matrix[i, j] >= 2 else "#1F2937")
        ax.set_xticks(range(len(depths)), [f"H={depth}" for depth in depths])
        ax.set_yticks(range(len(algorithms)), [label for _, label in algorithms])
        ax.set_xlabel("搜索深度 H")
        ax.set_title(label)
        ax.set_xticks(np.arange(-0.5, len(depths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(algorithms), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=axes, label="相对完整最优收益的损失（越低越好）", shrink=0.9)
    fig.suptitle("题四：搜索深度、首步选择与完整收益损失｜固定首步后再完整搜索")
    fig.text(0.5, -0.07, "数据：heuristic_evaluation_summary.csv；四宝藏 H=4、六宝藏 H=6 已搜完全部宝藏，无未来估值调用",
             ha="center", fontsize=9, color="#555F6B")
    path = OUTPUT / "图13_宝藏深度决策与收益损失.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    stage = read_csv("results/stage2/summary.csv")
    innovation = read_csv("results/innovations/summary.csv")
    extension = read_csv("results/extension/summary.csv")
    heuristic = read_csv("results/innovations/heuristic_evaluation_summary.csv")
    for path in (figure_b(innovation, extension), figure_c(stage, extension), figure_d(heuristic)):
        print(path)
    description = """# 图11～图13：数据来源与读图说明

运行 `python plot_analysis.py` 可重建三张 240 dpi PNG。脚本只读取既有实验 CSV，
并为图11、图12在相同的固定场景上生成新的逐格／逐层分析数据；不改动原有实验 CSV。

| 图 | 数据来源 | 含义和限制 |
| --- | --- | --- |
| 图11 加权启发函数紧度 | `innovation_experiments.striped_grid()`、`extension_experiments.alternating_terrain()`；反向 Dijkstra 得到每格真实最低剩余代价，逐格数据在 `analysis_data/task2_heuristic_tightness.csv`。原扩展节点数用创新／拓展汇总 CSV 核对。 | 点越接近对角虚线，启发值越接近真实代价；所有下界均已检验不高估。条纹图行列下界中位紧度 1.00，交错图两种下界逐格相同。该图不测预处理时间。 |
| 图12 博弈树逐层剪枝 | `stage2.generated_tree(depth=6, branching=4, seed=2026)` 与 `extension_experiments.ordered_tree()`；逐层追踪见 `analysis_data/task3_depth_trace.csv`。根值、总访问数、叶子数、直接剪枝数均与 `stage2/summary.csv`、`extension/summary.csv` 核对。 | 每格是该层实际访问的节点数，颜色用对数色阶；图中不比较追踪器运行时间。事后有利／不利顺序使用真实子树值，构造成本未计入。 |
| 图13 宝藏深度决策与损失 | 直接读取 `results/innovations/heuristic_evaluation_summary.csv`，不重新搜索。 | 每格显示所选首步及固定首步后完整搜索的收益损失；它不是深度截止预测值。四宝藏 H=4、六宝藏 H=6 已搜完宝藏，没有未来估值调用。 |

图11、图12的 CSV 是**新生成的分析记录**，并非旧实验的计时重复；图13完全复用原始汇总。
"""
    (OUTPUT / "图11-13_说明.md").write_text(description, encoding="utf-8")


if __name__ == "__main__":
    main()
