"""Stage-two measured comparisons and charts sourced from their CSV files."""

import argparse
import csv
import platform
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from astar import search
from experiments import MAZE, WEIGHTED, random_maze
from minimax import TreeNode, alpha_beta, minimax, pdf_tree
from treasure_game import distance_matrix, save_matrix_csv, solve

ROOT = Path(__file__).parent / "results" / "stage2"
RAW_FIELDS = ("experiment_id", "task", "case", "algorithm", "parameter", "depth", "order", "seed",
              "repeat", "repeats_planned", "timing_statistic", "result", "best_target", "path_steps",
              "expanded", "generated", "visited", "leaves", "pruned", "seconds", "python", "machine")
GROUP_FIELDS = ("experiment_id", "task", "case", "algorithm", "parameter", "depth", "order", "seed")


def generated_tree(depth: int = 6, branching: int = 4, seed: int = 2026) -> TreeNode:
    """Full tree; leaves drawn left-to-right from uniform integers [-100, 100]."""
    rng = random.Random(seed)

    def make(level: int) -> TreeNode:
        return TreeNode(value=rng.randint(-100, 100)) if level == 0 else TreeNode(children=[make(level - 1) for _ in range(branching)])

    return make(depth)


def generated_weighted_grid(size: int, *, seed: int, obstacle_probability: float = 0.15,
                            maximum_weight: int = 9,
                            goal: tuple[int, int] | None = None) -> list[list[str | int]]:
    """固定种子生成地形；顶行和目标列留一条可达走廊。"""
    if size < 2 or not 0 <= obstacle_probability < 1 or maximum_weight < 1:
        raise ValueError("Invalid weighted grid parameters")
    goal = goal or (size - 1, size - 1)
    if not (0 <= goal[0] < size and 0 <= goal[1] < size) or goal == (0, 0):
        raise ValueError("Goal must be inside the grid and differ from start")
    rng = random.Random(seed)
    grid: list[list[str | int]] = [
        ["X" if rng.random() < obstacle_probability else rng.randint(1, maximum_weight)
         for _ in range(size)] for _ in range(size)
    ]
    for column in range(goal[1] + 1):
        if grid[0][column] == "X":
            grid[0][column] = rng.randint(1, maximum_weight)
    for row in range(goal[0] + 1):
        if grid[row][goal[1]] == "X":
            grid[row][goal[1]] = rng.randint(1, maximum_weight)
    grid[0][0], grid[goal[0]][goal[1]] = "S", "E"
    return grid


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(repeats: int = 20, seed: int = 2026, output: Path = ROOT) -> tuple[Path, Path]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    matrix = distance_matrix()
    save_matrix_csv(matrix, output / "distance_matrix.csv")
    rows: list[dict] = []
    machine = platform.node()
    python = platform.python_version()
    cases = [
        ("pdf_maze", MAZE, (0, 0), (3, 4), False, False, ""),
        ("random_maze_20x20", random_maze(20, 20, seed=seed), (0, 0), (19, 19), False, False, seed),
        ("pdf_weighted", WEIGHTED, (0, 0), (4, 4), True, True, ""),
        # 复用 test_astar.py 的自设地形：高权值直路与免费终点。
        ("weighted_detour", [["S", 9, "E"], [1, 1, 1]], (0, 0), (0, 2), True, True, ""),
        # 复用“发现更便宜路径”案例；按题二规则让进入 E 免费。
        ("weighted_better_path", [["S", 8, 1], [1, 1, 1], [1, 1, "E"]],
         (0, 0), (2, 2), True, True, ""),
        ("weighted_random_20x20", generated_weighted_grid(20, seed=seed),
         (0, 0), (19, 19), True, True, seed),
        ("weighted_random_30x30", generated_weighted_grid(30, seed=seed + 1, obstacle_probability=0.25),
         (0, 0), (29, 29), True, True, seed + 1),
        ("weighted_random_40x40", generated_weighted_grid(40, seed=seed + 2, obstacle_probability=0.10,
                                                           maximum_weight=5),
         (0, 0), (39, 39), True, True, seed + 2),
        ("weighted_interior_30x30", generated_weighted_grid(30, seed=seed + 3,
                                                             obstacle_probability=0.15, maximum_weight=3,
                                                             goal=(15, 15)),
         (0, 0), (15, 15), True, True, seed + 3),
    ]

    def add(experiment_id: str, task: int, case: str, algorithm: str, parameter: str,
            depth: int | str, order: str, case_seed: int | str, repeat: int, result) -> None:
        rows.append(dict(experiment_id=experiment_id, task=task, case=case, algorithm=algorithm,
                         parameter=parameter, depth=depth, order=order, seed=case_seed,
                         repeat=repeat, repeats_planned=repeats, timing_statistic="arithmetic_mean",
                         result=getattr(result, "cost", getattr(result, "value", getattr(result, "best_value", ""))),
                         best_target=getattr(result, "best_target", ""),
                         path_steps=(len(result.path) - 1 if result.path else "") if hasattr(result, "path") else "",
                         expanded=getattr(result, "expanded", ""), generated=getattr(result, "generated", ""),
                         visited=getattr(result, "visited", ""), leaves=getattr(result, "leaves", ""),
                         pruned=getattr(result, "pruned", ""), seconds=result.elapsed_seconds,
                         python=python, machine=machine))

    for case, grid, start, goal, weighted, free_goal, case_seed in cases:
        task = 2 if weighted else 1
        for repeat in range(1, repeats + 1):
            outcomes = []
            for heuristic in ("zero", "auto"):
                result = search(grid, start, goal, weighted=weighted, free_goal=free_goal, heuristic=heuristic)
                outcomes.append(result.cost)
                add(f"E{task}-{case}", task, case, "A*", result.heuristic, "", "", case_seed, repeat, result)
            if outcomes[0] != outcomes[1]:
                raise AssertionError(f"A* heuristic changed optimum for {case}")

    tree_cases = (("pdf_tree", pdf_tree(), ""),
                  ("full_tree_d6_b4", generated_tree(seed=seed), seed))
    for case, tree, case_seed in tree_cases:
        for repeat in range(1, repeats + 1):
            outcomes = []
            for algorithm, order, function, kwargs in (
                ("Minimax", "left_to_right", minimax, {}),
                ("Alpha-Beta", "left_to_right", alpha_beta, {}),
                ("Alpha-Beta", "first_leaf", alpha_beta, {"ordering": "first_leaf"}),
            ):
                result = function(tree, **kwargs)
                outcomes.append(result.value)
                add(f"E3-{case}", 3, case, algorithm, "", 6 if case.startswith("full") else 3,
                    order, case_seed, repeat, result)
            if len(set(outcomes)) != 1:
                raise AssertionError(f"Tree strategies changed optimum for {case}")

    for depth in (2, 4):
        for repeat in range(1, repeats + 1):
            outcomes = []
            for algorithm, order in (("Minimax", "natural"), ("Alpha-Beta", "natural"),
                                     ("Alpha-Beta", "reverse"), ("Alpha-Beta", "gain")):
                result = solve(matrix, depth, algorithm=algorithm.lower().replace("-", "_"), order=order)
                outcomes.append(result.best_value)
                add(f"E4-H{depth}", 4, "pdf_treasure", algorithm, "", depth, order, "", repeat, result)
            if len(set(outcomes)) != 1:
                raise AssertionError(f"Treasure strategies changed optimum at H={depth}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    summary = []
    for key, samples in groups.items():
        first = samples[0]
        for metric in ("result", "best_target", "path_steps", "expanded", "generated", "visited", "leaves", "pruned"):
            if any(sample[metric] != first[metric] for sample in samples):
                raise AssertionError(f"Non-deterministic {metric}: {key}")
        summary.append(dict(zip(GROUP_FIELDS, key), **{field: first[field] for field in
                            ("result", "best_target", "path_steps", "expanded", "generated", "visited", "leaves", "pruned")},
                            repeats=len(samples), mean_seconds=mean(sample["seconds"] for sample in samples)))
    raw_path, summary_path = output / "raw.csv", output / "summary.csv"
    write_csv(raw_path, rows, RAW_FIELDS)
    write_csv(summary_path, summary, (*GROUP_FIELDS, "result", "best_target", "path_steps", "expanded", "generated", "visited", "leaves", "pruned", "repeats", "mean_seconds"))
    return raw_path, summary_path


def charts(summary_path: Path, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        data = list(csv.DictReader(stream))
    made = []

    def bars(filename: str, title: str, labels: list[str], values: list[int], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(12 if len(values) > 4 else 9, 4.8), layout="constrained")
        palette = ["#226b7a", "#df8b39", "#578650", "#8d6383"]
        artists = ax.bar(labels, values, color=[palette[index % len(palette)] for index in range(len(values))])
        ax.bar_label(artists, padding=3)
        if len(values) > 4:
            ax.tick_params(axis="x", labelsize=9)
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    maze = [r for r in data if r["case"] == "pdf_maze"]
    bars("task1_expanded.png", "题一：PDF 5×5 迷宫，h=0 与曼哈顿距离", [r["parameter"] for r in maze],
         [int(r["expanded"]) for r in maze], "扩展节点数")
    weighted = [r for r in data if r["task"] == "2"]
    cases = list(dict.fromkeys(r["case"] for r in weighted))
    baseline = [int(next(r["expanded"] for r in weighted if r["case"] == case and r["parameter"] == "zero")) for case in cases]
    guided = [int(next(r["expanded"] for r in weighted if r["case"] == case and r["parameter"] != "zero")) for case in cases]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    positions = list(range(len(cases)))
    left = ax.barh([p - 0.2 for p in positions], baseline, height=0.38, label="h=0", color="#226b7a")
    right = ax.barh([p + 0.2 for p in positions], guided, height=0.38, label="免费终点下界", color="#df8b39")
    ax.bar_label(left, padding=3)
    ax.bar_label(right, padding=3)
    ax.set_yticks(positions, [case.replace("weighted_", "") for case in cases])
    ax.invert_yaxis()
    ax.set(title="题二：PDF、自设小地图及固定种子大地图；进入 E 免费", xlabel="扩展节点数")
    ax.set_xlim(0, max(baseline + guided) * 1.15)
    ax.legend()
    path = output / "task2_expanded.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)
    tree = [r for r in data if r["case"] == "full_tree_d6_b4"]
    labels = [r["algorithm"] + "\n" + r["order"] for r in tree]
    bars("task3_visited.png", f"题三：完整树，种子 {tree[0]['seed']}，深度 6、分支 4", labels,
         [int(r["visited"]) for r in tree], "访问节点数")
    bars("task3_pruned.png", f"题三：种子 {tree[0]['seed']}，跳过的直接子分支", labels,
         [int(r["pruned"]) for r in tree], "剪枝分支数")

    treasure = [r for r in data if r["case"] == "pdf_treasure"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for index, depth in enumerate((2, 4)):
        subset = [r for r in treasure if int(r["depth"]) == depth]
        labels = [r["algorithm"] + "\n" + r["order"] for r in subset]
        values = [int(r["visited"]) for r in subset]
        artists = axes[index].bar(labels, values, color=["#226b7a", "#df8b39", "#8d6383", "#578650"])
        axes[index].bar_label(artists, padding=3)
        axes[index].set(title=f"H={depth}，首步 {subset[0]['best_target']}，收益差 {subset[0]['result']}", ylabel="访问节点数")
        axes[index].set_ylim(0, max(values) * 1.18)
    fig.suptitle("题四：PDF 宝藏地图，深度与访问顺序对比")
    path = output / "task4_depths.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-two comparisons and CSV-driven charts")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    raw, summary = run(args.repeats, args.seed, args.output)
    images = charts(summary, args.output)
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Raw: {raw}; Summary: {summary}; Charts: {', '.join(map(str, images))}")


if __name__ == "__main__":
    main()
