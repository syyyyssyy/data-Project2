"""独立于 stage2.py 的四题创新对照，所有结果来自实际调用。"""

import argparse
import csv
import platform
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from astar import path_cost, search
from experiments import MAZE, WEIGHTED, random_maze
from innovations import (alpha_beta_transposition, maze_deeper_first,
                         treasure_estimated, treasure_memoized, weighted_axis_lower_bound)
from minimax import TreeNode, alpha_beta, pdf_tree
from stage2 import generated_weighted_grid
from treasure_game import GameState, distance_matrix, solve
from treasure_game import VALUES

OUTPUT = Path(__file__).parent / "results" / "innovations"
FIELDS = ("task", "case", "algorithm", "repeat", "seed", "depth", "result", "best_target",
          "expanded", "generated", "visited", "leaves", "pruned", "cache_hits",
          "calls", "elapsed_seconds", "python", "machine", "timing_scope")
GROUP = ("task", "case", "algorithm", "seed", "depth")


def striped_grid(size: int = 25) -> list[list[str | int]]:
    """每隔五列放一条高代价地带，展示行列下界能识别的必经费用。"""
    grid: list[list[str | int]] = [[7 if c % 5 == 4 else 1 for c in range(size)] for _ in range(size)]
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def shared_dag(seed: int = 2026) -> TreeNode:
    """固定随机叶值；每层重复引用上一层节点，产生真正的共享子树。"""
    rng = random.Random(seed)
    layer = [TreeNode(value=rng.randint(-50, 50)) for _ in range(8)]
    for _ in range(5):
        layer = [TreeNode(children=[layer[(3 * i + j) % len(layer)] for j in range(3)])
                 for i in range(8)]
    return TreeNode(children=layer[:4])


def extended_treasure() -> tuple[dict, GameState, dict[str, int]]:
    """六个宝藏的无障碍四方向网格距离；每个标记占据不同坐标。"""
    points = {"A": (0, 0), "B": (7, 7), "T1": (0, 4), "T2": (2, 1),
              "T3": (2, 6), "T4": (5, 2), "T5": (6, 5), "T6": (7, 0)}
    values = {"T1": 9, "T2": 6, "T3": 7, "T4": 7, "T5": 5, "T6": 8}
    matrix = {a: {b: abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) for b, pb in points.items()}
              for a, pa in points.items()}
    return matrix, GameState(remaining=tuple(values)), values


def run(repeats: int = 20, output: Path = OUTPUT) -> tuple[Path, Path]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    machine, python = platform.node(), platform.python_version()

    def measure(task: int, case: str, algorithm: str, repeat: int, seed: int | str,
                depth: int | str, function, *, grid=None, goal=None, weighted=False):
        began = perf_counter()
        result = function()
        elapsed = perf_counter() - began
        if grid is not None and result.path:
            assert result.cost == path_cost(grid, result.path, weighted=weighted, free_goal=weighted)
            assert result.path[-1] == goal
        rows.append(dict(task=task, case=case, algorithm=algorithm, repeat=repeat,
                         seed=seed, depth=depth, result=getattr(result, "cost", getattr(result, "value", getattr(result, "best_value", ""))),
                         best_target=getattr(result, "best_target", ""),
                         expanded=getattr(result, "expanded", ""), generated=getattr(result, "generated", ""),
                         visited=getattr(result, "visited", ""), leaves=getattr(result, "leaves", ""),
                         pruned=getattr(result, "pruned", ""), cache_hits=getattr(result, "cache_hits", ""),
                         calls=(result.visited + result.cache_hits) if hasattr(result, "cache_hits") else getattr(result, "visited", ""),
                         elapsed_seconds=elapsed, python=python, machine=machine,
                         timing_scope="full_function_call"))
        return result

    maze_cases = (("pdf_maze", MAZE, (3, 4), ""),
                  ("open_25x25", [["S" if (r, c) == (0, 0) else "E" if (r, c) == (24, 24) else "."
                                    for c in range(25)] for r in range(25)], (24, 24), ""),
                  ("random_20x20", random_maze(20, 20, seed=2026), (19, 19), 2026))
    for case, grid, goal, seed in maze_cases:
        for repeat in range(repeats):
            old = measure(1, case, "A*_FIFO", repeat, seed, "",
                          lambda: search(grid, (0, 0), goal), grid=grid, goal=goal)
            new = measure(1, case, "A*_deeper_tie", repeat, seed, "",
                          lambda: maze_deeper_first(grid, (0, 0), goal), grid=grid, goal=goal)
            assert old.cost == new.cost

    weighted_cases = (("pdf_weighted", WEIGHTED, (4, 4), ""),
                      ("striped_25x25", striped_grid(), (24, 24), ""),
                      ("random_30x30", generated_weighted_grid(30, seed=2027, obstacle_probability=0.25),
                       (29, 29), 2027))
    for case, grid, goal, seed in weighted_cases:
        for repeat in range(repeats):
            old = measure(2, case, "A*_global_min", repeat, seed, "",
                          lambda: search(grid, (0, 0), goal, weighted=True, free_goal=True),
                          grid=grid, goal=goal, weighted=True)
            same_tie = measure(2, case, "A*_global_min_deeper", repeat, seed, "",
                               lambda: search(grid, (0, 0), goal, weighted=True, free_goal=True,
                                              tie_break="deeper"),
                               grid=grid, goal=goal, weighted=True)
            new = measure(2, case, "A*_axis_min", repeat, seed, "",
                          lambda: weighted_axis_lower_bound(grid, (0, 0), goal),
                          grid=grid, goal=goal, weighted=True)
            assert old.cost == new.cost == same_tie.cost

    for case, tree, seed in (("pdf_tree", pdf_tree(), ""), ("shared_dag", shared_dag(), 2026)):
        for repeat in range(repeats):
            old = measure(3, case, "Alpha-Beta", repeat, seed, "",
                          lambda: alpha_beta(tree))
            new = measure(3, case, "Alpha-Beta_TT", repeat, seed, "",
                          lambda: alpha_beta_transposition(tree))
            assert old.value == new.value

    pdf_matrix = distance_matrix()
    extended_matrix, extended_state, extended_values = extended_treasure()
    for case, matrix, state, values, depths in (("pdf_treasure", pdf_matrix, GameState(), None, (2, 4)),
                                                 ("six_treasure_grid", extended_matrix, extended_state,
                                                  extended_values, (4, 6))):
        for depth in depths:
            for repeat in range(repeats):
                uncached = measure(4, case, "Minimax_no_cache", repeat, "", depth,
                                   lambda: treasure_memoized(matrix, depth, state=state,
                                                              values=values or VALUES, cache=False))
                cached = measure(4, case, "Minimax_memo", repeat, "", depth,
                                 lambda: treasure_memoized(matrix, depth, state=state,
                                                            values=values or VALUES))
                assert (uncached.best_target, uncached.best_value) == (cached.best_target, cached.best_value)
                if case == "pdf_treasure":
                    reference = measure(4, case, "Alpha-Beta_reference", repeat, "", depth,
                                        lambda: solve(matrix, depth))
                    assert (cached.best_target, cached.best_value) == (reference.best_target, reference.best_value)

    raw = output / "raw.csv"
    summary = output / "summary.csv"
    with raw.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP)].append(row)
    summed = []
    for key, samples in groups.items():
        for field in ("result", "best_target", "expanded", "generated", "visited", "leaves", "pruned", "cache_hits", "calls"):
            assert len({str(sample[field]) for sample in samples}) == 1, (key, field)
        summed.append(dict(zip(GROUP, key), **{field: samples[0][field] for field in
                              ("result", "best_target", "expanded", "generated", "visited", "leaves", "pruned", "cache_hits", "calls")},
                           repeats=len(samples), mean_seconds=mean(sample["elapsed_seconds"] for sample in samples),
                           median_seconds=median(sample["elapsed_seconds"] for sample in samples)))
    with summary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*GROUP, "result", "best_target", "expanded", "generated",
                                                    "visited", "leaves", "pruned", "cache_hits", "calls",
                                                    "repeats", "mean_seconds", "median_seconds"))
        writer.writeheader()
        writer.writerows(summed)
    return raw, summary


def charts(summary_path: Path, output: Path) -> list[Path]:
    """只读汇总 CSV 绘图，图上节点数与保存的数据一致。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        data = list(csv.DictReader(stream))
    made: list[Path] = []

    def panel(ax, task: str, cases: list[str], algorithms: list[str], metric: str, title: str,
              selected_depth: dict[str, str] | None = None) -> None:
        subset = [row for row in data if row["task"] == task and
                  (selected_depth is None or row["depth"] == selected_depth.get(row["case"]))]
        palette = ["#226b7a", "#df8b39", "#578650"]
        width = 0.72 / len(algorithms)
        for index, algorithm in enumerate(algorithms):
            values = [int(next(row[metric] for row in subset
                               if row["case"] == case and row["algorithm"] == algorithm)) for case in cases]
            bars = ax.barh([position - 0.36 + width * (index + 0.5) for position in range(len(cases))],
                           values, height=width * 0.95, label=algorithm, color=palette[index])
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_yticks(range(len(cases)), cases)
        ax.invert_yaxis()
        ax.set_xlabel("扩展节点数" if metric == "expanded" else "实际求值节点数")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.margins(x=0.16)

    configurations = (
        ("task1_tie_break.png", "1", ["pdf_maze", "open_25x25", "random_20x20"],
         ["A*_FIFO", "A*_deeper_tie"], "expanded", "题一：曼哈顿 A*，同分顺序对比"),
        ("task2_axis_bound.png", "2", ["pdf_weighted", "striped_25x25", "random_30x30"],
         ["A*_global_min", "A*_global_min_deeper", "A*_axis_min"], "expanded",
         "题二：免费终点，地形下界与同分顺序"),
        ("task3_transposition.png", "3", ["pdf_tree", "shared_dag"],
         ["Alpha-Beta", "Alpha-Beta_TT"], "visited", "题三：共享子树置换表"),
        ("task4_memoization.png", "4", ["pdf_treasure", "six_treasure_grid"],
         ["Minimax_no_cache", "Minimax_memo"], "visited", "题四：PDF H=4 与六宝藏 H=6"),
    )
    for filename, task, cases, algorithms, metric, title in configurations:
        fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
        selected_depth = {"pdf_treasure": "4", "six_treasure_grid": "6"} if task == "4" else None
        panel(ax, task, cases, algorithms, metric, title, selected_depth)
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)
    return made


def run_heuristic_evaluation(repeats: int = 50, output: Path = OUTPUT) -> tuple[Path, Path, Path]:
    """题四另存深度受限估值实验；完整搜索只用于事后评价首步质量。"""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    pdf_matrix = distance_matrix()
    extended_matrix, extended_state, extended_values = extended_treasure()
    cases = (("pdf_treasure", pdf_matrix, GameState(), VALUES, (1, 2, 3, 4)),
             ("six_treasure_grid", extended_matrix, extended_state, extended_values, (1, 2, 3, 4, 6)))
    rows: list[dict] = []
    for case, matrix, state, values, depths in cases:
        full_depth = len(state.remaining)
        exact = treasure_memoized(matrix, full_depth, state=state, values=values)
        actual_by_target = {}
        for target in state.remaining:
            source = state.a_position if state.player == "A" else state.b_position
            gain = values[target] - matrix[source][target]
            rest = tuple(item for item in state.remaining if item != target)
            child = (GameState(target, state.b_position, rest, state.score_a + gain, state.score_b, "B")
                     if state.player == "A" else
                     GameState(state.a_position, target, rest, state.score_a, state.score_b + gain, "A"))
            actual_by_target[target] = treasure_memoized(matrix, full_depth - 1,
                                                          state=child, values=values).best_value
        for depth in depths:
            for repeat in range(1, repeats + 1):
                for algorithm, use_estimate in (("current_score", False), ("future_estimate", True)):
                    began = perf_counter()
                    result = treasure_estimated(matrix, depth, state=state, values=values,
                                                use_estimate=use_estimate)
                    elapsed = perf_counter() - began
                    actual = actual_by_target[result.best_target]
                    rows.append(dict(experiment="task4_heuristic_evaluation", task=4, case=case,
                                     algorithm=algorithm, seed="", depth=depth, repeat=repeat,
                                     estimate_value=result.best_value, best_target=result.best_target,
                                     full_optimal_target=exact.best_target, full_optimal_value=exact.best_value,
                                     realized_value=actual, regret=exact.best_value - actual,
                                     decision_matches_full=result.best_target == exact.best_target,
                                     visited=result.visited, leaves=result.leaves, pruned=result.pruned,
                                     estimate_calls=result.estimate_calls, elapsed_seconds=elapsed))
                    if depth == full_depth:
                        assert (result.best_target, result.best_value) == (exact.best_target, exact.best_value)
                    if case == "pdf_treasure" and not use_estimate and depth in (2, 4):
                        reference = solve(matrix, depth)
                        assert (result.best_target, result.best_value) == (reference.best_target, reference.best_value)
    fields = tuple(rows[0])
    raw_path = output / "heuristic_evaluation_raw.csv"
    summary_path = output / "heuristic_evaluation_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["algorithm"], row["depth"])].append(row)
    summaries = []
    stable_fields = tuple(field for field in fields if field not in ("repeat", "elapsed_seconds"))
    for samples in groups.values():
        for field in stable_fields:
            assert len({str(sample[field]) for sample in samples}) == 1, field
        summaries.append({field: samples[0][field] for field in stable_fields} |
                         {"repeats": len(samples),
                          "mean_seconds": mean(sample["elapsed_seconds"] for sample in samples),
                          "median_seconds": median(sample["elapsed_seconds"] for sample in samples)})
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    chart_path = chart_heuristic_evaluation(summary_path, output)
    return raw_path, summary_path, chart_path


def chart_heuristic_evaluation(summary_path: Path, output: Path) -> Path:
    """从真实 CSV 绘制首步造成的完整搜索收益损失。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, case in zip(axes, ("pdf_treasure", "six_treasure_grid")):
        subset = [row for row in rows if row["case"] == case]
        depths = sorted({int(row["depth"]) for row in subset})
        for algorithm, color, label in (("current_score", "#226b7a", "仅当前收益"),
                                        ("future_estimate", "#df8b39", "加入未来估值")):
            selected = [next(row for row in subset if int(row["depth"]) == depth and
                             row["algorithm"] == algorithm) for depth in depths]
            ax.plot(depths, [int(row["regret"]) for row in selected], marker="o", label=label, color=color)
            for depth, row in zip(depths, selected):
                ax.annotate(row["best_target"], (depth, int(row["regret"])),
                            xytext=(0, 8 if algorithm == "current_score" else -15),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=case, xlabel="搜索深度 H", ylabel="完整搜索下的收益损失")
        ax.set_xticks(depths)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.legend(fontsize=8)
    fig.suptitle("题四：深度受限首步的决策质量；PDF 四宝藏与拓展六宝藏")
    path = output / "task4_heuristic_evaluation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent algorithm innovation comparisons")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    raw, summary = run(args.repeats, args.output)
    images = charts(summary, args.output)
    heuristic_raw, heuristic_summary, heuristic_chart = run_heuristic_evaluation(args.repeats, args.output)
    print(f"Raw: {raw}\nSummary: {summary}\nCharts: {', '.join(map(str, images))}")
    print(f"Heuristic evaluation: {heuristic_raw}; {heuristic_summary}; {heuristic_chart}")


if __name__ == "__main__":
    main()
