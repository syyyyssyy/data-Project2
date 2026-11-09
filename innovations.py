"""四题的独立拓展；标准入口和原有实验保持默认行为。"""

from dataclasses import dataclass, replace
from math import inf
from time import perf_counter
from typing import Mapping

from astar import Grid, Position, SearchResult, _validate, _weight, search
from minimax import TreeNode, tree_from_dict
from treasure_game import GameState, VALUES


def maze_deeper_first(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """题一：f 相同时优先 g 较大的节点；仍使用曼哈顿距离。"""
    return search(grid, start, goal, tie_break="deeper")


def weighted_axis_lower_bound(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """题二：按必经的行/列边界最低进入代价，构造可采纳启发函数。"""
    began = perf_counter()
    _validate(grid, start, goal, True)
    rows, cols = len(grid), len(grid[0])
    row_min = [inf] * rows
    col_min = [inf] * cols
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "X":
                cost = _weight(grid[r][c])
                row_min[r] = min(row_min[r], cost)
                col_min[c] = min(col_min[c], cost)
    # 整行/列不可通行时该边界没有有效地形下界；用 0 避免 inf-inf。
    row_min = [0 if value == inf else value for value in row_min]
    col_min = [0 if value == inf else value for value in col_min]
    row_prefix = [0]
    col_prefix = [0]
    for cost in row_min:
        row_prefix.append(row_prefix[-1] + cost)
    for cost in col_min:
        col_prefix.append(col_prefix[-1] + cost)

    def axis_lower_bound(point: Position, destination: Position) -> float:
        r, c = point
        gr, gc = destination
        # 每条必须跨过的行/列边界至少支付目标行/列中的最低进入费。
        # 水平移动只跨列，垂直移动只跨行，因此两项可相加。
        row_cost = (row_prefix[gr + 1] - row_prefix[r + 1] if r < gr
                    else row_prefix[r] - row_prefix[gr])
        col_cost = (col_prefix[gc + 1] - col_prefix[c + 1] if c < gc
                    else col_prefix[c] - col_prefix[gc])
        # 每条实际路径的正常进入费 >= 必经边界下界；免费终点只少付一次。
        return max(row_cost + col_cost - _weight(grid[gr][gc]), 0)

    result = search(grid, start, goal, weighted=True, free_goal=True,
                    heuristic=axis_lower_bound, tie_break="deeper")
    return replace(result, elapsed_seconds=perf_counter() - began, heuristic="axis_minimum_lower_bound")


@dataclass(frozen=True)
class CachedTreeResult:
    value: int
    visited: int
    leaves: int
    pruned: int
    cache_hits: int
    elapsed_seconds: float


def alpha_beta_transposition(root: TreeNode | dict) -> CachedTreeResult:
    """题三：只缓存完全求值的子树；被剪枝后的界限不能当作精确值复用。"""
    began = perf_counter()
    if isinstance(root, dict):
        root = tree_from_dict(root)
    exact_cache: dict[tuple[int, bool], int] = {}
    active: set[int] = set()
    visited = leaves = pruned = cache_hits = 0

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> tuple[int, bool]:
        nonlocal visited, leaves, pruned, cache_hits
        if not isinstance(node, TreeNode):
            raise ValueError("Every child must be a TreeNode")
        identity = id(node)
        if identity in active:
            raise ValueError("Cycle in game tree")
        key = (identity, maximizing)
        if key in exact_cache:
            cache_hits += 1
            return exact_cache[key], True
        visited += 1
        if not node.children:
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise ValueError("A leaf must have an integer value")
            leaves += 1
            exact_cache[key] = node.value
            return node.value, True
        if node.value is not None:
            raise ValueError("Internal nodes must not have a value")
        best = -inf if maximizing else inf
        exact = True
        active.add(identity)
        try:
            for index, child in enumerate(node.children):
                value, child_exact = visit(child, not maximizing, alpha, beta)
                exact &= child_exact
                best = max(best, value) if maximizing else min(best, value)
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    pruned += len(node.children) - index - 1
                    exact = False
                    break
            answer = int(best)
            if exact:
                exact_cache[key] = answer
            return answer, exact
        finally:
            active.remove(identity)

    answer, _ = visit(root, True, -inf, inf)
    return CachedTreeResult(answer, visited, leaves, pruned, cache_hits, perf_counter() - began)


@dataclass(frozen=True)
class CachedGameResult:
    best_target: str | None
    best_value: int
    visited: int
    leaves: int
    cache_hits: int
    elapsed_seconds: float
    depth: int


@dataclass(frozen=True)
class EstimatedGameResult:
    """深度受限博弈结果；best_value 是估值，不能当作实际最终收益。"""

    best_target: str | None
    best_value: int
    visited: int
    leaves: int
    pruned: int
    estimate_calls: int
    elapsed_seconds: float
    depth: int


def treasure_future_estimate(matrix: Mapping[str, Mapping[str, int | float]], state: GameState,
                             *, values: Mapping[str, int] = VALUES) -> int:
    """按当前玩家的即时收益贪心模拟剩余行动，返回预测的增量收益差。"""
    a, b, player = state.a_position, state.b_position, state.player
    remaining = list(state.remaining)
    future = 0
    while remaining:
        source = a if player == "A" else b
        for target in remaining:
            if target not in values or source not in matrix or target not in matrix[source]:
                raise ValueError("Missing treasure value or distance")
        target = max(remaining, key=lambda item: values[item] - matrix[source][item])
        distance = matrix[source][target]
        if distance == inf:
            raise ValueError(f"Target {target} is unreachable from {source}")
        gain = values[target] - distance
        future += gain if player == "A" else -gain
        if player == "A":
            a = target
            player = "B"
        else:
            b = target
            player = "A"
        remaining.remove(target)
    return int(future)


def treasure_estimated(matrix: Mapping[str, Mapping[str, int | float]], depth: int,
                       *, state: GameState | None = None, values: Mapping[str, int] = VALUES,
                       use_estimate: bool = True) -> EstimatedGameResult:
    """题四拓展：深度截止时用当前收益差加预测未来收益，沿用 Alpha-Beta。"""
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("Depth must be a nonnegative integer")
    current = state if state is not None else GameState()
    if current.player not in ("A", "B") or any(target not in values for target in current.remaining):
        raise ValueError("Invalid game state or treasure values")
    began = perf_counter()
    visited = leaves = pruned = estimate_calls = 0

    def visit(node: GameState, turns: int, alpha: float, beta: float) -> tuple[int, str | None]:
        nonlocal visited, leaves, pruned, estimate_calls
        visited += 1
        if turns == 0 or not node.remaining:
            leaves += 1
            future = 0
            if turns == 0 and node.remaining and use_estimate:
                estimate_calls += 1
                future = treasure_future_estimate(matrix, node, values=values)
            return node.value + future, None
        maximizing = node.player == "A"
        best = -inf if maximizing else inf
        best_target = None
        for index, target in enumerate(node.remaining):
            source = node.a_position if maximizing else node.b_position
            if source not in matrix or target not in matrix[source]:
                raise ValueError("Missing distance")
            distance = matrix[source][target]
            if distance == inf:
                raise ValueError(f"Target {target} is unreachable from {source}")
            gain = values[target] - distance
            rest = tuple(item for item in node.remaining if item != target)
            child = (GameState(target, node.b_position, rest, node.score_a + gain, node.score_b, "B")
                     if maximizing else
                     GameState(node.a_position, target, rest, node.score_a, node.score_b + gain, "A"))
            score, _ = visit(child, turns - 1, alpha, beta)
            if best_target is None or (score > best if maximizing else score < best):
                best, best_target = score, target
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                pruned += len(node.remaining) - index - 1
                break
        return int(best), best_target

    value, target = visit(current, depth, -inf, inf)
    return EstimatedGameResult(target, value, visited, leaves, pruned,
                               estimate_calls, perf_counter() - began, depth)


def treasure_memoized(matrix: Mapping[str, Mapping[str, int | float]], depth: int,
                      *, state: GameState | None = None, values: Mapping[str, int] = VALUES,
                      cache: bool = True) -> CachedGameResult:
    """题四：缓存剩余行动的收益差，不把历史累计分数放进键。"""
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("Depth must be a nonnegative integer")
    current = state if state is not None else GameState()
    if current.player not in ("A", "B") or any(target not in values for target in current.remaining):
        raise ValueError("Invalid game state or treasure values")
    began = perf_counter()
    memo: dict[tuple[str, str, tuple[str, ...], str, int], int] = {}
    visited = leaves = cache_hits = 0

    def continuation(a: str, b: str, remaining: tuple[str, ...], player: str, turns: int) -> int:
        nonlocal visited, leaves, cache_hits
        key = (a, b, remaining, player, turns)
        if cache and key in memo:
            cache_hits += 1
            return memo[key]
        visited += 1
        if turns == 0 or not remaining:
            leaves += 1
            if cache:
                memo[key] = 0
            return 0
        best = -inf if player == "A" else inf
        for target in remaining:
            source = a if player == "A" else b
            distance = matrix[source][target]
            if distance == inf:
                raise ValueError(f"Target {target} is unreachable from {source}")
            gain = values[target] - distance
            rest = tuple(name for name in remaining if name != target)
            future = continuation(target if player == "A" else a,
                                  target if player == "B" else b,
                                  rest, "B" if player == "A" else "A", turns - 1)
            candidate = (gain if player == "A" else -gain) + future
            best = max(best, candidate) if player == "A" else min(best, candidate)
        answer = int(best)
        if cache:
            memo[key] = answer
        return answer

    if depth == 0 or not current.remaining:
        visited = leaves = 1
        return CachedGameResult(None, current.value, visited, leaves, 0, perf_counter() - began, depth)
    best = -inf if current.player == "A" else inf
    best_target = None
    for target in current.remaining:
        source = current.a_position if current.player == "A" else current.b_position
        distance = matrix[source][target]
        if distance == inf:
            raise ValueError(f"Target {target} is unreachable from {source}")
        gain = values[target] - distance
        rest = tuple(name for name in current.remaining if name != target)
        future = continuation(target if current.player == "A" else current.a_position,
                              target if current.player == "B" else current.b_position,
                              rest, "B" if current.player == "A" else "A", depth - 1)
        candidate = (gain if current.player == "A" else -gain) + future
        if best_target is None or (candidate > best if current.player == "A" else candidate < best):
            best, best_target = candidate, target
    visited += 1
    return CachedGameResult(best_target, int(current.value + best), visited, leaves,
                            cache_hits, perf_counter() - began, depth)
