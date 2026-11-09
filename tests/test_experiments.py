import unittest

from experiments import random_maze, run_comparisons


class ExperimentTests(unittest.TestCase):
    def test_repeatable_inputs_and_raw_samples(self):
        self.assertEqual(random_maze(10, 10, seed=42), random_maze(10, 10, seed=42))
        rows, summaries = run_comparisons(repeats=2, seed=42)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(summaries), 18)
        self.assertTrue(all(row["elapsed_seconds"] >= 0 for row in rows))
        self.assertTrue(all(row["repeats"] == 2 for row in summaries))


if __name__ == "__main__":
    unittest.main()
