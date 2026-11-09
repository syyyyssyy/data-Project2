"""Reproducible comparison runs; CSV rows contain measured results only."""

import csv
import random
from pathlib import Path
from statistics import median

from astar import search
from minimax import alpha_beta, minimax, pdf_tree
from treasure_game import distance_matrix, solve


MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
WEIGHTED = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]


def random_maze(rows: int, columns: int, *, seed: int, obstacle_probability: float = 0.2) -> list[list[str]]:
    if rows < 2 or columns < 2 or not 0 <= obstacle_probability < 1:
        raise ValueError("Invalid random maze parameters")
    rng = random.Random(seed)
    grid = [["X" if rng.random() < obstacle_probability else "." for _ in range(columns)] for _ in range(rows)]
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def _record(rows: list[dict], *, case: str, algorithm: str, heuristic: str = "",
            depth: int | str = "", order: str = "", repeat: int, result) -> None:
    rows.append({
        "case": case, "algorithm": algorithm, "heuristic": heuristic, "depth": depth,
        "order": order, "repeat": repeat,
        "cost_or_value": getattr(result, "cost", getattr(result, "value", getattr(result, "best_value", ""))),
        "best_target": getattr(result, "best_target", ""),
        "path_steps": len(result.path) - 1 if hasattr(result, "path") and result.path else "",
        "expanded": getattr(result, "expanded", ""), "generated": getattr(result, "generated", ""),
        "visited": getattr(result, "visited", ""), "leaves": getattr(result, "leaves", ""),
        "pruned": getattr(result, "pruned", ""), "elapsed_seconds": result.elapsed_seconds,
    })


def run_comparisons(*, repeats: int = 5, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows: list[dict] = []
    cases = (("pdf_maze", MAZE, (0, 0), (3, 4), False, False),
             ("pdf_weighted", WEIGHTED, (0, 0), (4, 4), True, True),
             (f"random_10x10_seed_{seed}", random_maze(10, 10, seed=seed), (0, 0), (9, 9), False, False))
    for case, grid, start, goal, weighted, free_goal in cases:
        for heuristic in ("zero", "auto"):
            for repeat in range(repeats):
                result = search(grid, start, goal, weighted=weighted, free_goal=free_goal, heuristic=heuristic)
                _record(rows, case=case, algorithm="astar", heuristic=result.heuristic, repeat=repeat, result=result)
    for order_name, order_fn in (("natural", None), ("reverse", lambda items: list(reversed(items)))):
        for algorithm, function in (("minimax", minimax), ("alpha_beta", alpha_beta)):
            for repeat in range(repeats):
                result = function(pdf_tree(), order=order_fn)
                _record(rows, case="pdf_tree", algorithm=algorithm, order=order_name, repeat=repeat, result=result)
    matrix = distance_matrix()
    for depth in (2, 4):
        for order in ("natural", "reverse"):
            for algorithm in ("minimax", "alpha_beta"):
                for repeat in range(repeats):
                    result = solve(matrix, depth, algorithm=algorithm, order=order)
                    _record(rows, case="pdf_treasure", algorithm=algorithm, depth=depth, order=order, repeat=repeat, result=result)
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in ("case", "algorithm", "heuristic", "depth", "order"))
        groups.setdefault(key, []).append(row["elapsed_seconds"])
    summaries = [dict(zip(("case", "algorithm", "heuristic", "depth", "order"), key),
                      repeats=len(times), median_seconds=median(times)) for key, times in groups.items()]
    return rows, summaries


def save_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        raise ValueError("No measured rows to save")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
