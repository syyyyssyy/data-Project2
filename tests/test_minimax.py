import unittest
import json

from minimax import TreeNode, alpha_beta, minimax, pdf_tree, tree_from_dict


class MinimaxTests(unittest.TestCase):
    def test_pdf_json_input(self):
        sample = json.loads('''{"tree_structure":{"is_maximizing_player":true,"children":[
          {"is_maximizing_player":false,"children":[
            {"is_maximizing_player":true,"children":[{"value":3},{"value":5}]},
            {"is_maximizing_player":true,"children":[{"value":6},{"value":9}]}
          ]},
          {"is_maximizing_player":false,"children":[
            {"is_maximizing_player":true,"children":[{"value":6},{"value":9}]},
            {"is_maximizing_player":true,"children":[{"value":1},{"value":2}]}
          ]}
        ]}}''')
        self.assertEqual(alpha_beta(sample).value, 5)
        self.assertEqual(minimax(sample).value, 5)
        self.assertEqual(alpha_beta(tree_from_dict(sample)).value, 5)

    def test_invalid_json_player_flag(self):
        with self.assertRaises(ValueError):
            alpha_beta({"tree_structure": {"is_maximizing_player": False, "children": [{"value": 3}]}})

    def test_pdf_tree(self):
        self.assertEqual(minimax(pdf_tree()).value, 5)
        self.assertEqual(alpha_beta(pdf_tree()).value, 5)

    def test_leaf_and_invalid(self):
        self.assertEqual(minimax(TreeNode(7)).value, 7)
        self.assertEqual(alpha_beta(TreeNode(7)).value, 7)
        with self.assertRaises(ValueError):
            minimax(TreeNode())
        root = TreeNode()
        root.children.append(root)
        with self.assertRaises(ValueError):
            alpha_beta(root)

    def test_pruning_and_order(self):
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(5), TreeNode(7)]),
            TreeNode(children=[TreeNode(2), TreeNode(100)]),
        ])
        plain, pruned = minimax(root), alpha_beta(root)
        self.assertEqual(plain.value, pruned.value)
        self.assertEqual(pruned.value, 5)
        self.assertLess(pruned.leaves, plain.leaves)
        reverse = lambda children: list(reversed(children))
        self.assertEqual(alpha_beta(root, order=reverse).value, 5)
        ordered = alpha_beta(root, ordering="first_leaf")
        self.assertEqual(ordered.value, plain.value)
        self.assertLessEqual(ordered.visited, plain.visited)


if __name__ == "__main__":
    unittest.main()
