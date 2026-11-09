"""题三博弈树：MAX/MIN 交替递归，默认从左到右搜索。"""

from dataclasses import dataclass, field
from math import inf
from time import perf_counter
from typing import Callable


@dataclass
class TreeNode:
    """叶子使用 value，内部节点使用 children；根节点默认由 MAX 决策。"""
    value: int | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class TreeResult:
    """visited 含内部节点和叶子；pruned 统计被跳过的直接子分支。"""
    value: int
    visited: int
    leaves: int
    pruned: int
    elapsed_seconds: float


def tree_from_dict(data: dict) -> TreeNode:
    """将 PDF 的 JSON 树（含 tree_structure 外层）转换为 TreeNode。"""
    if not isinstance(data, dict):
        raise ValueError("Tree input must be a JSON object")
    if "tree_structure" in data:
        if set(data) != {"tree_structure"}:
            raise ValueError("tree_structure wrapper cannot contain other fields")
        data = data["tree_structure"]

    def build(item: dict, maximizing: bool) -> TreeNode:
        if not isinstance(item, dict):
            raise ValueError("Every tree node must be a JSON object")
        if "children" in item:
            if set(item) != {"children", "is_maximizing_player"}:
                raise ValueError("An internal node needs children and is_maximizing_player")
            if item["is_maximizing_player"] is not maximizing:
                raise ValueError("MAX and MIN must alternate, starting with MAX")
            children = item["children"]
            if not isinstance(children, list) or not children:
                raise ValueError("An internal node needs a nonempty children list")
            return TreeNode(children=[build(child, not maximizing) for child in children])
        if set(item) != {"value"} or not isinstance(item["value"], int) or isinstance(item["value"], bool):
            raise ValueError("A leaf needs one integer value")
        return TreeNode(value=item["value"])

    return build(data, True)


def _first_leaf(node: TreeNode) -> int:
    """用子树最左叶值估计先搜哪个孩子；只影响顺序，不代替真正的递归值。"""
    seen: set[int] = set()
    while isinstance(node, TreeNode) and node.children:
        if id(node) in seen:
            raise ValueError("Cycle in game tree")
        seen.add(id(node))
        node = node.children[0]
    if not isinstance(node, TreeNode) or not isinstance(node.value, int) or isinstance(node.value, bool):
        raise ValueError("A leaf must have an integer value")
    return node.value


def _solve(root: TreeNode, *, pruning: bool, order: Callable[[list[TreeNode]], list[TreeNode]] | None = None,
           ordering: str = "natural") -> TreeResult:
    """共用递归核心：pruning=False 是 Minimax，True 是 Alpha-Beta。"""
    if ordering not in ("natural", "first_leaf") or (order is not None and ordering != "natural"):
        raise ValueError("Use one supported ordering method")
    began = perf_counter()
    visited = leaves = pruned = 0
    active: set[int] = set()

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> int:
        nonlocal visited, leaves, pruned
        if not isinstance(node, TreeNode):
            raise ValueError("Every child must be a TreeNode")
        identity = id(node)
        if identity in active:
            raise ValueError("Cycle in game tree")
        visited += 1
        if not node.children:
            # 到叶子才能读取具体分数；父层会按 MAX 或 MIN 规则回传它。
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise ValueError("A leaf must have an integer value")
            leaves += 1
            return node.value
        if node.value is not None:
            raise ValueError("Internal nodes must not have a value")
        if ordering == "first_leaf":
            # 搜索值不变；先搜更可能有利的子树，可能让界限更早收紧。
            children = sorted(node.children, key=_first_leaf, reverse=maximizing)
        else:
            children = list(node.children) if order is None else order(list(node.children))
        if len(children) != len(node.children) or {id(x) for x in children} != {id(x) for x in node.children}:
            raise ValueError("Order callback must return a permutation of children")
        active.add(identity)
        try:
            # MAX 从负无穷开始取最大；MIN 从正无穷开始取最小。
            best = -inf if maximizing else inf
            for index, child in enumerate(children):
                value = visit(child, not maximizing, alpha, beta)
                best = max(best, value) if maximizing else min(best, value)
                if pruning:
                    # alpha 是 MAX 已有的保底值，beta 是 MIN 已有的上界。
                    if maximizing:
                        alpha = max(alpha, best)
                    else:
                        beta = min(beta, best)
                    if alpha >= beta:
                        # 当前分支已不可能改变祖先的选择，剩余孩子可以不访问。
                        pruned += len(children) - index - 1
                        break
            return int(best)
        finally:
            active.remove(identity)

    value = visit(root, True, -inf, inf)
    return TreeResult(value, visited, leaves, pruned, perf_counter() - began)


def minimax(root: TreeNode | dict, *, order: Callable[[list[TreeNode]], list[TreeNode]] | None = None,
            ordering: str = "natural") -> TreeResult:
    """遍历全部必要子树，作为正确性和性能对照基线。"""
    if isinstance(root, dict):
        root = tree_from_dict(root)
    return _solve(root, pruning=False, order=order, ordering=ordering)


def alpha_beta(root: TreeNode | dict, *, order: Callable[[list[TreeNode]], list[TreeNode]] | None = None,
               ordering: str = "natural") -> TreeResult:
    """返回与 Minimax 相同的根值，并跳过不会影响该值的子分支。"""
    if isinstance(root, dict):
        root = tree_from_dict(root)
    return _solve(root, pruning=True, order=order, ordering=ordering)


def alpha_beta_score(root: TreeNode | dict) -> int:
    """题三的简洁输出接口：按从左到右顺序返回 MAX 的最佳整数得分。"""
    return alpha_beta(root).value


def pdf_tree() -> TreeNode:
    """按 PDF 的具体节点定义构造样例树；shared 被两个 MIN 节点引用。"""
    shared = TreeNode(children=[TreeNode(6), TreeNode(9)])
    return TreeNode(children=[
        TreeNode(children=[TreeNode(children=[TreeNode(3), TreeNode(5)]), shared]),
        TreeNode(children=[shared, TreeNode(children=[TreeNode(1), TreeNode(2)])]),
    ])
