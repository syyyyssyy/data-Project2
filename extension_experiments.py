"""拓展实验：只补测缺失组合；旧实验以 coverage.csv 引用，不改写原数据。"""

import argparse
import csv
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from astar import path_cost, search
from innovation_experiments import extended_treasure
from innovations import weighted_axis_lower_bound
from minimax import TreeNode, alpha_beta, minimax
from stage2 import generated_tree
from treasure_game import GameState, VALUES, distance_matrix, solve

ROOT = Path(__file__).parent
OUTPUT = ROOT / "results" / "extension"
RAW_FIELDS = ("experiment_id", "task", "case", "algorithm", "depth", "seed", "repeat",
              "result", "best_target", "expanded", "generated", "visited", "leaves", "pruned",
              "elapsed_seconds", "python", "machine", "run_utc", "timing_scope")
GROUP_FIELDS = ("experiment_id", "task", "case", "algorithm", "depth", "seed")
METRICS = ("result", "best_target", "expanded", "generated", "visited", "leaves", "pruned")


def narrow_corridor(size: int = 25) -> list[list[str]]:
    """隔行单向蛇形通路；奇数行只开一个连接口，所以起终点间没有岔路。"""
    if size < 5 or size % 4 != 1:
        raise ValueError("Corridor size must be at least 5 and congruent to 1 modulo 4")
    grid = [["X"] * size for _ in range(size)]
    for row in range(0, size, 2):
        grid[row] = ["."] * size
    for row in range(1, size, 2):
        grid[row][size - 1 if (row // 2) % 2 == 0 else 0] = "."
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def alternating_terrain(size: int = 25) -> list[list[str | int]]:
    """高低代价交错：每行每列均有低价格，检验行列最低值下界何时变弱。"""
    grid: list[list[str | int]] = [[1 if (r + c) % 2 == 0 else 9 for c in range(size)]
                                   for r in range(size)]
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def ordered_tree(node: TreeNode, maximizing: bool, favorable: bool) -> tuple[TreeNode, int]:
    """用完整子树真实值构造最有利/最不利顺序；这是事后控制变量，不是实用排序器。"""
    if not node.children:
        if node.value is None:
            raise ValueError("Leaf must have a value")
        return TreeNode(value=node.value), node.value
    children = [ordered_tree(child, not maximizing, favorable) for child in node.children]
    children.sort(key=lambda pair: pair[1], reverse=(maximizing == favorable))
    value = (max if maximizing else min)(score for _, score in children)
    return TreeNode(children=[child for child, _ in children]), value


def eight_treasure() -> tuple[dict, GameState, dict[str, int]]:
    """固定 8×8 无障碍四方向网格；沿用六宝藏坐标，再加入两个明确标记。"""
    points = {"A": (0, 0), "B": (7, 7), "T1": (0, 4), "T2": (2, 1),
              "T3": (2, 6), "T4": (5, 2), "T5": (6, 5), "T6": (7, 0),
              "T7": (3, 3), "T8": (4, 6)}
    values = {"T1": 9, "T2": 6, "T3": 7, "T4": 7, "T5": 5, "T6": 8,
              "T7": 10, "T8": 4}
    matrix = {a: {b: abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) for b, pb in points.items()}
              for a, pa in points.items()}
    return matrix, GameState(remaining=tuple(values)), values


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(repeats: int = 20, output: Path = OUTPUT) -> tuple[Path, Path, Path, Path]:
    """计时只比较本次同环境调用；全部结果先验证，再写入独立 CSV。"""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    python, machine = platform.python_version(), platform.node()
    run_utc = datetime.now(timezone.utc).isoformat()

    def measure(task: int, case: str, algorithm: str, depth: int | str, function,
                repeat: int, *, grid=None, weighted=False, goal=None):
        began = perf_counter()
        result = function()
        elapsed = perf_counter() - began
        if grid is not None and result.path:
            assert result.path[-1] == goal
            assert path_cost(grid, result.path, weighted=weighted, free_goal=weighted) == result.cost
        rows.append(dict(experiment_id=f"EXT-Q{task}", task=task, case=case, algorithm=algorithm,
                         depth=depth, seed=2026 if task == 3 else "", repeat=repeat,
                         result=getattr(result, "cost", getattr(result, "value", getattr(result, "best_value", ""))),
                         best_target=getattr(result, "best_target", ""),
                         expanded=getattr(result, "expanded", ""), generated=getattr(result, "generated", ""),
                         visited=getattr(result, "visited", ""), leaves=getattr(result, "leaves", ""),
                         pruned=getattr(result, "pruned", ""), elapsed_seconds=elapsed,
                         python=python, machine=machine, run_utc=run_utc,
                         timing_scope="full_function_call"))
        return result

    # 题一：旧 CSV 已有同尺寸空旷地图；这里只补单通路地图。
    corridor = narrow_corridor()
    for repeat in range(1, repeats + 1):
        old = measure(1, "Q1-01_corridor_25", "A*_FIFO", "", lambda: search(corridor, (0, 0), (24, 24)),
                      repeat, grid=corridor, goal=(24, 24))
        new = measure(1, "Q1-01_corridor_25", "A*_deeper_tie", "",
                      lambda: search(corridor, (0, 0), (24, 24), tie_break="deeper"),
                      repeat, grid=corridor, goal=(24, 24))
        assert old.cost == new.cost

    # 题二：条纹地图和三种策略已有记录；补测行列低价格无处不在的交错地图。
    grid = alternating_terrain()
    for repeat in range(1, repeats + 1):
        outcomes = []
        for name, function in (
            ("Dijkstra_h0", lambda: search(grid, (0, 0), (24, 24), weighted=True, free_goal=True,
                                           heuristic="zero")),
            ("A*_global_min", lambda: search(grid, (0, 0), (24, 24), weighted=True, free_goal=True)),
            ("A*_global_min_deeper", lambda: search(grid, (0, 0), (24, 24), weighted=True, free_goal=True,
                                                     tie_break="deeper")),
            ("A*_axis_min", lambda: weighted_axis_lower_bound(grid, (0, 0), (24, 24))),
        ):
            result = measure(2, "Q2-02_alternating_25", name, "", function, repeat,
                             grid=grid, weighted=True, goal=(24, 24))
            outcomes.append(result.cost)
        assert len(set(outcomes)) == 1

    # 题三：保持叶值、形状和根值，依真实子树值事后构造两种访问顺序。
    original = generated_tree(depth=6, branching=4, seed=2026)
    favorable, exact = ordered_tree(original, True, True)
    unfavorable, reverse_exact = ordered_tree(original, True, False)
    assert exact == reverse_exact == minimax(original).value
    for case, tree in (("Q3-01_natural", original), ("Q3-02_favorable", favorable),
                       ("Q3-03_unfavorable", unfavorable)):
        for repeat in range(1, repeats + 1):
            plain = measure(3, case, "Minimax", 6, lambda: minimax(tree), repeat)
            pruned = measure(3, case, "Alpha-Beta", 6, lambda: alpha_beta(tree), repeat)
            assert plain.value == pruned.value == exact

    # 题四：每个宝藏数量先在同一 H=4 比较，再在 H=宝藏数看完整搜索。
    matrix4 = distance_matrix()
    matrix6, state6, values6 = extended_treasure()
    matrix8, state8, values8 = eight_treasure()
    games = (("Q4-01_pdf_4", matrix4, GameState(), VALUES, (4,)),
             ("Q4-02_grid_6", matrix6, state6, values6, (4, 6)),
             ("Q4-03_grid_8", matrix8, state8, values8, (4, 8)))
    for case, matrix, state, values, depths in games:
        for depth in depths:
            for repeat in range(1, repeats + 1):
                outcomes = []
                for algorithm, order in (("minimax", "natural"), ("alpha_beta", "natural"),
                                         ("alpha_beta", "gain")):
                    result = measure(4, case, f"{algorithm}_{order}", depth,
                                     lambda algorithm=algorithm, order=order: solve(
                                         matrix, depth, algorithm=algorithm, order=order,
                                         state=state, values=values), repeat)
                    outcomes.append(result.best_value)
                assert len(set(outcomes)) == 1

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in GROUP_FIELDS)].append(row)
    summary = []
    for key, samples in groups.items():
        for field in METRICS:
            assert len({str(sample[field]) for sample in samples}) == 1, (key, field)
        summary.append(dict(zip(GROUP_FIELDS, key), **{field: samples[0][field] for field in METRICS},
                            repeats=len(samples), mean_seconds=mean(x["elapsed_seconds"] for x in samples),
                            median_seconds=median(x["elapsed_seconds"] for x in samples)))
    raw_path, summary_path = output / "raw.csv", output / "summary.csv"
    write_csv(raw_path, rows, RAW_FIELDS)
    write_csv(summary_path, summary, (*GROUP_FIELDS, *METRICS, "repeats", "mean_seconds", "median_seconds"))
    cases_path, coverage_path = make_index(output, summary)
    return raw_path, summary_path, cases_path, coverage_path


def make_index(output: Path, summary: list[dict]) -> tuple[Path, Path]:
    """列明新测、复用和仅测试案例；复用项必须能在原汇总 CSV 中找到。"""
    cases = [
        ("Q1-01_corridor_25", 1, "25×25 蛇形唯一通路；S=(0,0)，E=(24,24)", "", "本次新测"),
        ("Q1-02_open_25", 1, "25×25 无障碍多路径；S=(0,0)，E=(24,24)", "", "复用创新实验"),
        ("Q2-01_striped_25", 2, "25×25 每第5列权值7，其余1；E免费", "", "复用创新实验"),
        ("Q2-02_alternating_25", 2, "25×25 棋盘交错权值1/9；E免费", "", "本次新测"),
        ("Q3-01_natural", 3, "深度6、分支4、叶值随机整数[-100,100]；原顺序", 2026, "本次新测"),
        ("Q3-02_favorable", 3, "同一棵树；MAX高值先、MIN低值先，真实值事后排序", 2026, "本次新测"),
        ("Q3-03_unfavorable", 3, "同一棵树；MAX低值先、MIN高值先，真实值事后排序", 2026, "本次新测"),
        ("Q4-01_pdf_4", 4, "PDF四宝藏；H=4", "", "本次新测"),
        ("Q4-02_grid_6", 4, "8×8无障碍网格六宝藏；H=4/6", "", "本次新测"),
        ("Q4-03_grid_8", 4, "同网格再加T7=(3,3)/10和T8=(4,6)/4；H=4/8", "", "本次新测"),
        ("EDGE-01", 1, "迷宫终点不可达", "", "仅正确性测试"),
        ("EDGE-02", 2, "加权地图终点不可达", "", "仅正确性测试"),
        ("EDGE-03", 1, "多条等长最短路", "", "仅正确性测试"),
        ("EDGE-04", 4, "所有宝藏即时收益为负且仍必须选", "", "仅正确性测试"),
        ("EDGE-05", 4, "两个最优首步并列", "", "仅正确性测试"),
        ("EDGE-06", 4, "H大于剩余宝藏数，以及目标不可达报错", "", "仅正确性测试"),
    ]
    cases_path = output / "cases.csv"
    write_csv(cases_path, [dict(case=case, task=task, construction=construction, seed=seed, status=status)
                           for case, task, construction, seed, status in cases],
              ("case", "task", "construction", "seed", "status"))
    coverage = [dict(case=row["case"], task=row["task"], algorithm=row["algorithm"],
                     depth=row["depth"], status="measured", source="results/extension/summary.csv",
                     source_case=row["case"], repeats=row["repeats"]) for row in summary]
    with (ROOT / "results" / "innovations" / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    for case, source_case, algorithms in (
        ("Q1-02_open_25", "open_25x25", {"A*_FIFO", "A*_deeper_tie"}),
        ("Q2-01_striped_25", "striped_25x25", {"A*_global_min", "A*_global_min_deeper", "A*_axis_min"}),
    ):
        found = [row for row in old if row["case"] == source_case and row["algorithm"] in algorithms]
        assert {row["algorithm"] for row in found} == algorithms
        coverage.extend(dict(case=case, task=row["task"], algorithm=row["algorithm"], depth=row["depth"],
                             status="reused", source="results/innovations/summary.csv",
                             source_case=source_case, repeats=row["repeats"]) for row in found)
    coverage.extend(dict(case=f"EDGE-{number:02d}", task=next(row[1] for row in cases if row[0] == f"EDGE-{number:02d}"),
                         algorithm="见测试", depth="", status="test_only", source="tests/test_extension.py",
                         source_case=f"test_edge_{number:02d}", repeats="") for number in range(1, 7))
    coverage_path = output / "coverage.csv"
    write_csv(coverage_path, coverage, ("case", "task", "algorithm", "depth", "status", "source", "source_case", "repeats"))
    return cases_path, coverage_path


def charts(summary_path: Path, output: Path) -> list[Path]:
    """四张中文图只读取 CSV；跨旧实验的图只比较确定性的节点数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        new = list(csv.DictReader(stream))
    with (ROOT / "results" / "innovations" / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    made = []

    def grouped(filename, title, labels, series, ylabel):
        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        width = 0.8 / len(series)
        for index, (name, values) in enumerate(series):
            bars = ax.bar([i - 0.4 + width * (index + 0.5) for i in range(len(labels))], values,
                          width=width * 0.95, label=name)
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_xticks(range(len(labels)), labels)
        ax.set(title=title, ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(value for _, values in series for value in values) * 1.18)
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    def find(rows, case, algorithm, depth=""):
        return next(row for row in rows if row["case"] == case and row["algorithm"] == algorithm
                    and str(row["depth"]) == str(depth))

    grouped("task1_structure.png", "题一：25×25 单通路与空旷地图；旧/新 CSV 的确定性扩展数",
            ["蛇形单通路", "空旷多路径"],
            [(name, [int(find(new, "Q1-01_corridor_25", name)["expanded"]),
                     int(find(old, "open_25x25", name)["expanded"])])
             for name in ("A*_FIFO", "A*_deeper_tie")], "扩展节点数")
    grouped("task2_terrain.png", "题二：25×25 条纹与交错地形；免费终点、确定性扩展数",
            ["高价条纹", "高低交错"],
            [(name, [int(find(old, "striped_25x25", name)["expanded"]),
                     int(find(new, "Q2-02_alternating_25", name)["expanded"])])
             for name in ("A*_global_min", "A*_global_min_deeper", "A*_axis_min")], "扩展节点数")
    grouped("task3_order.png", "题三：同一棵深度6、分支4的树；seed=2026，完整子树重排",
            ["原顺序", "有利顺序", "不利顺序"],
            [(name, [int(find(new, case, name, 6)["visited"]) for case in
                     ("Q3-01_natural", "Q3-02_favorable", "Q3-03_unfavorable")])
             for name in ("Minimax", "Alpha-Beta")], "访问节点数")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, depth_pairs, title in (
        (axes[0], (("Q4-02_grid_6", 4), ("Q4-03_grid_8", 4)), "固定 H=4"),
        (axes[1], (("Q4-02_grid_6", 6), ("Q4-03_grid_8", 8)), "搜完全部宝藏"),
    ):
        for index, (name, label, color) in enumerate((
            ("alpha_beta_natural", "自然顺序", "#226b7a"),
            ("alpha_beta_gain", "即时收益排序", "#df8b39"),
        )):
            values = [int(find(new, case, name, depth)["visited"]) for case, depth in depth_pairs]
            bars = ax.bar([position - 0.2 + index * 0.4 for position in range(2)], values,
                          width=0.38, label=label, color=color)
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_xticks(range(2), [f"6个 H={depth_pairs[0][1]}", f"8个 H={depth_pairs[1][1]}"])
        ax.set(title=title, ylabel="访问节点数")
        ax.set_ylim(0, max(int(find(new, case, "alpha_beta_natural", depth)["visited"])
                           for case, depth in depth_pairs) * 1.2)
        ax.legend(fontsize=8)
    fig.suptitle("题四：无障碍 8×8 网格，宝藏数、深度与收益排序")
    path = output / "task4_growth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled extension scenarios")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    files = run(args.repeats, args.output)
    images = charts(files[1], args.output)
    print("CSV:", *(str(path) for path in files))
    print("Charts:", *(str(path) for path in images))


if __name__ == "__main__":
    main()
