import tempfile
import unittest
from pathlib import Path

from minimax import alpha_beta, minimax
from astar import path_cost, search
from stage2 import generated_tree, generated_weighted_grid, run


class StageTwoTests(unittest.TestCase):
    def test_generated_weighted_grid_is_repeatable_and_reachable(self):
        grid = generated_weighted_grid(20, seed=2026)
        self.assertEqual(grid, generated_weighted_grid(20, seed=2026))
        self.assertEqual((grid[0][0], grid[-1][-1]), ("S", "E"))
        self.assertEqual(len(grid), 20)
        self.assertEqual(len(grid[0]), 20)
        self.assertTrue(all(grid[0][column] != "X" for column in range(20)))
        self.assertTrue(all(grid[row][-1] != "X" for row in range(20)))
        baseline = search(grid, (0, 0), (19, 19), weighted=True, free_goal=True, heuristic="zero")
        guided = search(grid, (0, 0), (19, 19), weighted=True, free_goal=True)
        self.assertEqual(baseline.cost, guided.cost)
        self.assertEqual(guided.cost, path_cost(grid, guided.path, weighted=True, free_goal=True))
        interior = generated_weighted_grid(30, seed=2029, maximum_weight=3, goal=(15, 15))
        self.assertEqual(interior[15][15], "E")
        self.assertTrue(all(interior[0][column] != "X" for column in range(16)))
        self.assertTrue(all(interior[row][15] != "X" for row in range(16)))
        with self.assertRaises(ValueError):
            generated_weighted_grid(30, seed=2029, goal=(30, 15))

    def test_seeded_tree_and_ordering_preserve_value(self):
        tree = generated_tree(depth=4, branching=3, seed=2026)
        self.assertEqual(minimax(tree).value, alpha_beta(tree).value)
        self.assertEqual(minimax(tree).value, alpha_beta(tree, ordering="first_leaf").value)
        self.assertEqual(minimax(generated_tree(depth=4, branching=3, seed=2026)).value, minimax(tree).value)

    def test_stage_two_csv_has_measured_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, summary = run(repeats=2, seed=2026, output=Path(directory))
            import csv
            with raw.open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            with summary.open(encoding="utf-8-sig", newline="") as stream:
                groups = list(csv.DictReader(stream))
            self.assertEqual(len(samples), 64)
            self.assertEqual(len(groups), 32)
            weighted = [row for row in groups if row["task"] == "2"]
            self.assertEqual({row["case"] for row in weighted},
                             {"pdf_weighted", "weighted_detour", "weighted_better_path",
                              "weighted_random_20x20", "weighted_random_30x30", "weighted_random_40x40",
                              "weighted_interior_30x30"})
            self.assertTrue(all(int(row["repeats"]) == 2 for row in groups))


if __name__ == "__main__":
    unittest.main()
