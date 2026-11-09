"""拓展场景的正确性和边界规则；性能结论以 CSV 实测为准。"""

import csv
import tempfile
import unittest
from math import inf
from pathlib import Path

from astar import search, weighted_cost
from extension_experiments import (alternating_terrain, eight_treasure, narrow_corridor,
                                   ordered_tree, run)
from minimax import alpha_beta, minimax
from stage2 import generated_tree
from treasure_game import GameState, distance_matrix, solve, take


class ExtensionTests(unittest.TestCase):
    def test_controlled_scenarios_and_csv(self):
        corridor = narrow_corridor()
        fifo = search(corridor, (0, 0), (24, 24))
        deeper = search(corridor, (0, 0), (24, 24), tie_break="deeper")
        self.assertEqual((fifo.cost, deeper.cost), (336, 336))
        weighted = alternating_terrain()
        self.assertEqual(search(weighted, (0, 0), (24, 24), weighted=True, free_goal=True,
                                heuristic="zero").cost,
                         search(weighted, (0, 0), (24, 24), weighted=True, free_goal=True).cost)
        root = generated_tree(seed=2026)
        favorable, value = ordered_tree(root, True, True)
        unfavorable, other = ordered_tree(root, True, False)
        self.assertEqual(value, other)
        self.assertEqual(alpha_beta(favorable).value, minimax(unfavorable).value)
        matrix, state, values = eight_treasure()
        self.assertEqual(solve(matrix, 4, state=state, values=values).best_value,
                         solve(matrix, 4, state=state, values=values, order="gain").best_value)
        with tempfile.TemporaryDirectory() as directory:
            raw, summary, cases, coverage = run(1, Path(directory))
            with raw.open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            with summary.open(encoding="utf-8-sig", newline="") as stream:
                groups = list(csv.DictReader(stream))
            with coverage.open(encoding="utf-8-sig", newline="") as stream:
                index = list(csv.DictReader(stream))
            self.assertEqual(len(samples), len(groups))
            self.assertTrue(cases.exists())
            self.assertEqual({row["status"] for row in index}, {"measured", "reused", "test_only"})

    def test_edge_01(self):
        grid = [["S", "X", "E"]]
        self.assertEqual(search(grid, (0, 0), (0, 2)).path, [])
        self.assertEqual(search(grid, (0, 0), (0, 2), tie_break="deeper").cost, inf)

    def test_edge_02(self):
        grid = [["S", "X", "E"]]
        self.assertEqual(weighted_cost((0, 0), (0, 2), grid), -1)

    def test_edge_03(self):
        grid = [["S", "."], [".", "E"]]
        results = [search(grid, (0, 0), (1, 1), tie_break=mode) for mode in ("fifo", "deeper")]
        self.assertTrue(all(result.cost == 2 and len(result.path) == 3 for result in results))

    def test_edge_04(self):
        names = ("A", "B", "T1", "T2")
        matrix = {a: {b: 0 if a == b else 10 for b in names} for a in names}
        state = GameState(remaining=("T1", "T2"))
        values = {"T1": 1, "T2": 1}
        result = solve(matrix, 2, state=state, values=values)
        self.assertIsNotNone(result.best_target)
        self.assertEqual(result.best_value, 0)
        self.assertEqual(take(state, result.best_target, matrix, values=values).score_a, -9)

    def test_edge_05(self):
        names = ("A", "B", "T1", "T2")
        matrix = {a: {b: 0 if a == b else 1 for b in names} for a in names}
        state = GameState(remaining=("T1", "T2"))
        values = {"T1": 5, "T2": 5}
        self.assertEqual(solve(matrix, 1, state=state, values=values).best_target, "T1")
        self.assertEqual(solve(matrix, 1, state=state, values=values, order="reverse").best_target, "T2")

    def test_edge_06(self):
        matrix = distance_matrix()
        self.assertEqual(solve(matrix, 8).best_value, solve(matrix, 4).best_value)
        self.assertEqual(solve(matrix, 8).best_target, solve(matrix, 4).best_target)
        unreachable = {a: dict(row) for a, row in matrix.items()}
        unreachable["A"]["T1"] = inf
        with self.assertRaisesRegex(ValueError, "unreachable"):
            take(GameState(), "T1", unreachable)


if __name__ == "__main__":
    unittest.main()
