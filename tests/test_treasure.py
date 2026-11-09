import unittest

from astar import search
from treasure_game import GRID, NAMES, POSITIONS, VALUES, GameState, distance_matrix, solve, take


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = distance_matrix()

    def test_map_and_matrix(self):
        self.assertEqual((len(GRID), len(GRID[0])), (7, 7))
        self.assertEqual(POSITIONS["T1"], (0, 6))
        self.assertEqual(POSITIONS["T2"], (2, 2))
        self.assertEqual(POSITIONS["T3"], (4, 4))
        self.assertEqual(POSITIONS["T4"], (6, 0))
        self.assertEqual(VALUES, {"T1": 9, "T2": 6, "T3": 7, "T4": 7})
        for first in NAMES:
            self.assertEqual(self.matrix[first][first], 0)
            for second in NAMES:
                self.assertEqual(self.matrix[first][second], self.matrix[second][first])
                self.assertEqual(self.matrix[first][second], search(GRID, POSITIONS[first], POSITIONS[second], weighted=True, heuristic="zero").cost)

    def test_state_and_gain(self):
        original = GameState()
        moved = take(original, "T2", self.matrix)
        self.assertEqual(moved.a_position, "T2")
        self.assertEqual(moved.score_a, VALUES["T2"] - self.matrix["A"]["T2"])
        self.assertEqual(moved.player, "B")
        self.assertNotIn("T2", moved.remaining)
        self.assertIn("T2", original.remaining)
        with self.assertRaises(ValueError):
            take(moved, "T2", self.matrix)

    def test_pdf_depths_and_reproducibility(self):
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            plain = solve(self.matrix, depth, algorithm="minimax")
            pruned = solve(self.matrix, depth, algorithm="alpha_beta")
            self.assertEqual((plain.best_target, plain.best_value), (target, value))
            self.assertEqual((pruned.best_target, pruned.best_value), (target, value))
            self.assertEqual((solve(self.matrix, depth).best_target, solve(self.matrix, depth).best_value), (target, value))

    def test_depth_and_order(self):
        self.assertEqual(solve(self.matrix, 0).best_value, 0)
        self.assertIsNone(solve(self.matrix, 0).best_target)
        for depth in (1, 2, 3, 4):
            for order in ("natural", "reverse", "gain"):
                a = solve(self.matrix, depth, algorithm="minimax", order=order)
                b = solve(self.matrix, depth, algorithm="alpha_beta", order=order)
                self.assertEqual((a.best_target, a.best_value), (b.best_target, b.best_value))


if __name__ == "__main__":
    unittest.main()
