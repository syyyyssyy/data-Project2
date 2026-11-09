"""核对 PDF 要求的简洁返回类型和命令行独立结果行。"""

import ast
import io
import json
import unittest
from contextlib import redirect_stdout

from astar import maze_path, weighted_cost
from experiments import MAZE, WEIGHTED
from main import run_task
from minimax import alpha_beta_score, pdf_tree
from treasure_game import best_treasure_choice, distance_matrix, format_matrix


class OutputContractTests(unittest.TestCase):
    @staticmethod
    def lines_for(task: str) -> list[str]:
        stream = io.StringIO()
        with redirect_stdout(stream):
            run_task(task)
        return stream.getvalue().splitlines()

    def test_simple_python_interfaces(self):
        self.assertEqual(maze_path((0, 0), (3, 4), MAZE)[-1], (3, 4))
        self.assertEqual(weighted_cost((0, 0), (4, 4), WEIGHTED), 7)
        self.assertEqual(alpha_beta_score(pdf_tree()), 5)
        sample = {"tree_structure": {"is_maximizing_player": True,
                                     "children": [{"value": 3}, {"value": 5}]}}
        self.assertEqual(alpha_beta_score(sample), 5)
        matrix = distance_matrix()
        self.assertEqual(best_treasure_choice(matrix, 2), {"best_target": "T2", "best_value": -1})
        self.assertEqual(best_treasure_choice(matrix, 4, algorithm="minimax"),
                         {"best_target": "T3", "best_value": 3})
        self.assertEqual(len(format_matrix(matrix).splitlines()), 7)  # 表头＋六行

    def test_cli_results_are_separate_lines(self):
        first = self.lines_for("1")
        self.assertEqual(first[0], "Task 1 output:")
        self.assertEqual(ast.literal_eval(first[1]), maze_path((0, 0), (3, 4), MAZE))
        self.assertTrue(first[2].startswith("Task 1 stats:"))
        second = self.lines_for("2")
        self.assertEqual(second[0], "Task 2 output:")
        self.assertEqual(int(second[1]), 7)
        self.assertTrue(second[2].startswith("Task 2 stats:"))
        third = self.lines_for("3")
        self.assertEqual([int(third[index + 1]) for index, line in enumerate(third)
                          if line.endswith(" output:")], [5, 5])
        fourth = self.lines_for("4")
        outputs = [json.loads(fourth[index + 1]) for index, line in enumerate(fourth)
                   if line.endswith(" output:")]
        self.assertEqual(outputs, [{"best_target": "T2", "best_value": -1}] * 2 +
                         [{"best_target": "T3", "best_value": 3}] * 2)


if __name__ == "__main__":
    unittest.main()
