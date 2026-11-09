import unittest

from astar import manhattan, maze_path, path_cost, search, weighted_cost


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


class AStarTests(unittest.TestCase):
    def test_pdf_maze(self):
        result = search(MAZE, (0, 0), (3, 4))
        self.assertEqual(result.cost, 7)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (3, 4))
        self.assertEqual(path_cost(MAZE, result.path), 7)

    def test_pdf_weighted(self):
        result = search(WEIGHTED, (0, 0), (4, 4), weighted=True, free_goal=True)
        self.assertEqual(weighted_cost((0, 0), (4, 4), WEIGHTED), 7)
        self.assertEqual(path_cost(WEIGHTED, result.path, weighted=True, free_goal=True), 7)
        self.assertEqual(search(WEIGHTED, (0, 0), (4, 4), weighted=True, free_goal=True, heuristic="zero").cost, 7)

    def test_start_goal_and_invalid(self):
        self.assertEqual(maze_path((0, 0), (0, 0), MAZE), [(0, 0)])
        for start, goal in (((-1, 0), (0, 0)), ((0, 0), (9, 0)), ((0, 3), (0, 0))):
            with self.assertRaises(ValueError):
                search(MAZE, start, goal)

    def test_unreachable_and_equal_routes(self):
        self.assertEqual(maze_path((0, 0), (2, 2), [["S", "X", "E"], ["X", "X", "X"], [".", ".", "."]]), [])
        result = search([["S", "."], [".", "E"]], (0, 0), (1, 1))
        self.assertEqual(result.cost, 2)
        self.assertEqual(path_cost([["S", "."], [".", "E"]], result.path), 2)

    def test_weighted_detour_and_free_goal(self):
        grid = [["S", 9, "E"], [1, 1, 1]]
        result = search(grid, (0, 0), (0, 2), weighted=True, free_goal=True)
        self.assertEqual(result.cost, 3)
        self.assertEqual(path_cost(grid, result.path, weighted=True, free_goal=True), 3)
        self.assertEqual(search([["S", "E"]], (0, 0), (0, 1), weighted=True, free_goal=True).cost, 0)
        with self.assertRaises(ValueError):
            search(grid, (0, 0), (0, 2), weighted=True, free_goal=True, heuristic="manhattan")

    def test_better_discovered_path(self):
        grid = [["S", 8, 1], [1, 1, 1], [1, 1, "E"]]
        a = search(grid, (0, 0), (2, 2), weighted=True)
        d = search(grid, (0, 0), (2, 2), weighted=True, heuristic="zero")
        self.assertEqual(a.cost, d.cost)
        self.assertEqual(a.cost, path_cost(grid, a.path, weighted=True))

    def test_invalid_path(self):
        with self.assertRaises(ValueError):
            path_cost(MAZE, [(0, 0), (-1, 0), (0, 1)])


if __name__ == "__main__":
    unittest.main()
