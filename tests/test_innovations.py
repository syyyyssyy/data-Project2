import unittest

from astar import path_cost, search
from innovations import (alpha_beta_transposition, maze_deeper_first,
                         treasure_estimated, treasure_future_estimate,
                         treasure_memoized, weighted_axis_lower_bound)
from innovation_experiments import extended_treasure, run, run_heuristic_evaluation, shared_dag
from minimax import TreeNode, alpha_beta, pdf_tree
from stage2 import generated_weighted_grid
from treasure_game import GameState, distance_matrix, solve


class InnovationTests(unittest.TestCase):
    def test_deeper_first_preserves_shortest_path(self):
        grid = [["."] * 15 for _ in range(15)]
        grid[0][0], grid[-1][-1] = "S", "E"
        old = search(grid, (0, 0), (14, 14))
        new = maze_deeper_first(grid, (0, 0), (14, 14))
        self.assertEqual((old.cost, new.cost), (28, 28))
        self.assertEqual(path_cost(grid, new.path), new.cost)
        self.assertLess(new.expanded, old.expanded)
        with self.assertRaises(ValueError):
            search(grid, (0, 0), (14, 14), tie_break="bad")

    def test_axis_bound_matches_dijkstra_on_weighted_inputs(self):
        grids = [generated_weighted_grid(15, seed=seed, goal=(10, 9)) for seed in range(4)]
        stripes = [[7 if c in (4, 8, 12) else 1 for c in range(15)] for _ in range(15)]
        stripes[0][0], stripes[-1][-1] = "S", "E"
        grids.append(stripes)
        for grid in grids:
            goal = next((r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == "E")
            baseline = search(grid, (0, 0), goal, weighted=True, free_goal=True, heuristic="zero")
            new = weighted_axis_lower_bound(grid, (0, 0), goal)
            self.assertEqual(new.cost, baseline.cost)
            self.assertEqual(path_cost(grid, new.path, weighted=True, free_goal=True), baseline.cost)
        blocked = [["S", 1, 1], ["X", "X", "X"], [1, 1, "E"]]
        self.assertEqual(weighted_axis_lower_bound(blocked, (0, 0), (2, 2)).path, [])

    def test_transposition_table_on_shared_tree(self):
        shared = TreeNode(children=[TreeNode(3), TreeNode(5), TreeNode(7)])
        root = TreeNode(children=[TreeNode(children=[shared, shared]),
                                  TreeNode(children=[shared, shared])])
        old, new = alpha_beta(root), alpha_beta_transposition(root)
        self.assertEqual(new.value, old.value)
        self.assertGreater(new.cache_hits, 0)
        self.assertLessEqual(new.visited, old.visited)
        self.assertEqual(alpha_beta_transposition(pdf_tree()).value, 5)
        dag = shared_dag(seed=2026)
        self.assertEqual(alpha_beta(dag).value, alpha_beta_transposition(dag).value)
        for seed in range(10):
            dag = shared_dag(seed=seed)
            self.assertEqual(alpha_beta(dag).value, alpha_beta_transposition(dag).value)

    def test_treasure_cache_and_extended_case(self):
        matrix = distance_matrix()
        for depth in (2, 4):
            old = solve(matrix, depth)
            new = treasure_memoized(matrix, depth)
            self.assertEqual((new.best_target, new.best_value),
                             (old.best_target, old.best_value))
            if depth == 4:
                self.assertGreater(new.cache_hits, 0)
        names = tuple(f"T{i}" for i in range(1, 7))
        coordinates = {"A": (0, 0), "B": (6, 6), **{name: (i, (i * 2) % 7) for i, name in enumerate(names)}}
        extended = {a: {b: abs(coordinates[a][0] - coordinates[b][0]) +
                             abs(coordinates[a][1] - coordinates[b][1]) for b in coordinates}
                    for a in coordinates}
        values = dict(zip(names, (9, 6, 7, 7, 5, 8)))
        state = GameState(remaining=names)
        plain = treasure_memoized(extended, 6, state=state, values=values, cache=False)
        cached = treasure_memoized(extended, 6, state=state, values=values)
        self.assertEqual((cached.best_target, cached.best_value),
                         (plain.best_target, plain.best_value))
        self.assertLess(cached.visited, plain.visited)
        self.assertGreater(cached.cache_hits, 0)
        changed = GameState(score_a=11, score_b=3)
        shifted = treasure_memoized(matrix, 4, state=changed)
        self.assertEqual(shifted.best_value, treasure_memoized(matrix, 4).best_value + 8)

    def test_treasure_heuristic_cutoff_and_full_depth(self):
        matrix = distance_matrix()
        initial = GameState()
        self.assertEqual(treasure_future_estimate(matrix, GameState(remaining=())), 0)
        for depth in (0, 1, 2, 3, 4):
            baseline = treasure_estimated(matrix, depth, use_estimate=False)
            reference = solve(matrix, depth)
            self.assertEqual((baseline.best_target, baseline.best_value),
                             (reference.best_target, reference.best_value))
            improved = treasure_estimated(matrix, depth)
            self.assertEqual((improved.best_target, improved.best_value),
                             (treasure_estimated(matrix, depth).best_target,
                              treasure_estimated(matrix, depth).best_value))
            if depth == 4:
                self.assertEqual((improved.best_target, improved.best_value), ("T3", 3))
                self.assertEqual(improved.estimate_calls, 0)
            elif depth < 4:
                self.assertGreater(improved.estimate_calls, 0)
        shifted = GameState(score_a=11, score_b=3)
        self.assertEqual(treasure_estimated(matrix, 2, state=shifted).best_value,
                         treasure_estimated(matrix, 2, state=initial).best_value + 8)
        extended, state, values = extended_treasure()
        exact = treasure_memoized(extended, 6, state=state, values=values)
        estimated = treasure_estimated(extended, 6, state=state, values=values)
        self.assertEqual((estimated.best_target, estimated.best_value),
                         (exact.best_target, exact.best_value))
        self.assertEqual(treasure_estimated(matrix, 3, use_estimate=False).best_target, "T2")
        self.assertEqual(treasure_estimated(matrix, 3).best_target, "T3")
        self.assertEqual(treasure_estimated(extended, 1, state=state, values=values,
                                            use_estimate=False).best_target, "T1")
        self.assertEqual(treasure_estimated(extended, 1, state=state, values=values).best_target, "T2")
        with self.assertRaises(ValueError):
            treasure_estimated(matrix, -1)

    def test_heuristic_evaluation_csv(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            raw, summary, chart = run_heuristic_evaluation(2, Path(directory))
            with raw.open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            with summary.open(encoding="utf-8-sig", newline="") as stream:
                groups = list(csv.DictReader(stream))
            self.assertEqual(len(groups), 18)
            self.assertEqual(len(samples), 36)
            self.assertTrue(chart.exists())
            self.assertTrue(all(int(row["regret"]) >= 0 for row in groups))

    def test_innovation_csv_is_separate_from_stage_two(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            raw, summary = run(repeats=2, output=Path(directory))
            with raw.open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            with summary.open(encoding="utf-8-sig", newline="") as stream:
                groups = list(csv.DictReader(stream))
            self.assertEqual(len(groups), 29)
            self.assertEqual(len(samples), 58)
            self.assertEqual({row["task"] for row in groups}, {"1", "2", "3", "4"})


if __name__ == "__main__":
    unittest.main()
