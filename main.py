"""Command line entry points for all four PDF tasks and comparisons."""

import argparse
import json
from pathlib import Path

from astar import path_cost, search
from experiments import MAZE, WEIGHTED, run_comparisons, save_csv
from minimax import alpha_beta, minimax, pdf_tree
from treasure_game import distance_matrix, format_matrix, save_matrix_csv, solve


def run_task(task: str, *, export: bool = False, tree_input: dict | None = None) -> None:
    if task in ("1", "all"):
        result = search(MAZE, (0, 0), (3, 4))
        print("Task 1 output:")
        print(result.path)
        print(f"Task 1 stats: steps={result.cost}, expanded={result.expanded}, generated={result.generated}")
        assert path_cost(MAZE, result.path) == result.cost
    if task in ("2", "all"):
        result = search(WEIGHTED, (0, 0), (4, 4), weighted=True, free_goal=True)
        print("Task 2 output:")
        print(-1 if not result.path else int(result.cost))
        print(f"Task 2 stats: path={result.path}, expanded={result.expanded}, generated={result.generated}")
        assert path_cost(WEIGHTED, result.path, weighted=True, free_goal=True) == result.cost
    if task in ("3", "all"):
        for name, function in (("Minimax", minimax), ("Alpha-Beta", alpha_beta)):
            result = function(tree_input if tree_input is not None else pdf_tree())
            print(f"Task 3 {name} output:")
            print(result.value)
            print(f"Task 3 {name} stats: visited={result.visited}, leaves={result.leaves}, pruned={result.pruned}")
    if task in ("4", "all"):
        matrix = distance_matrix()
        print("Task 4 distance matrix:\n" + format_matrix(matrix))
        if export:
            path = Path(__file__).parent / "results" / "distance_matrix.csv"
            save_matrix_csv(matrix, path)
            print(f"Saved {path}")
        for depth in (2, 4):
            for algorithm in ("minimax", "alpha_beta"):
                result = solve(matrix, depth, algorithm=algorithm)
                print(f"Task 4 H={depth} {algorithm} output:")
                print(json.dumps({"best_target": result.best_target, "best_value": result.best_value}))
                print(f"Task 4 H={depth} {algorithm} stats: visited={result.visited}, leaves={result.leaves}, pruned={result.pruned}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contemporary AI experiment 2")
    parser.add_argument("--task", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--experiments", action="store_true", help="run comparisons and save measured CSV files")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-matrix", action="store_true")
    parser.add_argument("--tree-json", type=Path, help="题三输入文件：PDF 样例的 tree_structure JSON")
    args = parser.parse_args()
    if args.tree_json and args.task not in ("3", "all"):
        parser.error("--tree-json 仅适用于 --task 3 或 all")
    tree_input = None
    if args.tree_json:
        with args.tree_json.open(encoding="utf-8-sig") as stream:
            tree_input = json.load(stream)
    run_task(args.task, export=args.export_matrix, tree_input=tree_input)
    if args.experiments:
        rows, summaries = run_comparisons(repeats=args.repeats, seed=args.seed)
        results_dir = Path(__file__).parent / "results"
        save_csv(rows, results_dir / "raw_runs.csv")
        save_csv(summaries, results_dir / "median_times.csv")
        print(f"Saved {len(rows)} measured runs and {len(summaries)} group medians to {results_dir}")


if __name__ == "__main__":
    main()
