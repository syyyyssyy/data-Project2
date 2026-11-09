"""题四：先用 A* 求关键点距离，再用对抗搜索决定 A 的首个宝藏。"""

import csv
from dataclasses import dataclass
from itertools import combinations
from math import inf
from pathlib import Path
from time import perf_counter
from typing import Mapping

from astar import search

GRID = (
    ("A", 1, 1, "X", 2, 1, "T1"),
    (1, "X", 6, "X", 1, "X", 1),
    (1, 1, "T2", 1, 1, 1, 1),
    ("X", 1, "X", 8, "X", 1, "X"),
    (1, 1, 1, 1, "T3", 1, 1),
    (1, "X", 9, "X", 1, "X", 1),
    ("T4", 1, 1, "X", 1, 1, "B"),
)
NAMES = ("A", "B", "T1", "T2", "T3", "T4")
POSITIONS = {cell: (r, c) for r, row in enumerate(GRID) for c, cell in enumerate(row) if cell in NAMES}
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TARGETS = tuple(VALUES)


def distance_matrix() -> dict[str, dict[str, int | float]]:
    """对六个关键点两两运行 A*，得到后续博弈可直接查用的移动代价。"""
    matrix: dict[str, dict[str, int | float]] = {name: {name: 0} for name in NAMES}
    for first, second in combinations(NAMES, 2):
        # 本题六个关键点的进入代价同为 1；反向走同一路径时，首尾权值相抵，代价相等。
        # 两个方向都实际搜索，以便发现地图或代价规则实现中的不对称错误。
        forward = search(GRID, POSITIONS[first], POSITIONS[second], weighted=True)
        backward = search(GRID, POSITIONS[second], POSITIONS[first], weighted=True)
        if forward.cost != backward.cost:
            raise ValueError(f"Distance asymmetry between {first} and {second}")
        matrix[first][second] = matrix[second][first] = forward.cost
    return matrix


def format_matrix(matrix: dict[str, dict[str, int | float]]) -> str:
    lines = ["     " + " ".join(f"{name:>4}" for name in NAMES)]
    for name in NAMES:
        lines.append(f"{name:>4} " + " ".join(f"{matrix[name][other]:>4}" for other in NAMES))
    return "\n".join(lines)


def save_matrix_csv(matrix: dict[str, dict[str, int | float]], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("from", *NAMES))
        for name in NAMES:
            writer.writerow((name, *(matrix[name][other] for other in NAMES)))


@dataclass(frozen=True)
class GameState:
    """不可变状态：位置、尚未选择的宝藏、双方累计收益及轮到谁行动。"""
    a_position: str = "A"
    b_position: str = "B"
    remaining: tuple[str, ...] = TARGETS
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def value(self) -> int:
        """A 要最大化、B 要最小化的收益差。"""
        return self.score_a - self.score_b


def take(state: GameState, target: str, matrix: dict[str, dict[str, int | float]],
         *, values: Mapping[str, int] = VALUES) -> GameState:
    """执行一次选择，返回新状态；原状态不会被递归分支修改。"""
    if target not in state.remaining or state.player not in ("A", "B"):
        raise ValueError("Invalid player or already claimed target")
    source = state.a_position if state.player == "A" else state.b_position
    distance = matrix[source][target]
    if distance == inf:
        raise ValueError(f"Target {target} is unreachable from {source}")
    # 只收集明确选中的 target；经过其他宝藏并不自动领取。
    gain = values[target] - int(distance)
    remaining = tuple(name for name in state.remaining if name != target)
    if state.player == "A":
        return GameState(target, state.b_position, remaining, state.score_a + gain, state.score_b, "B")
    return GameState(state.a_position, target, remaining, state.score_a, state.score_b + gain, "A")


@dataclass(frozen=True)
class GameResult:
    best_target: str | None
    best_value: int
    visited: int
    leaves: int
    pruned: int
    elapsed_seconds: float
    depth: int
    algorithm: str
    order: str


def solve(matrix: dict[str, dict[str, int | float]], depth: int, *, algorithm: str = "alpha_beta",
          state: GameState | None = None, order: str = "natural",
          values: Mapping[str, int] = VALUES) -> GameResult:
    """搜索 depth 次后续行动，返回当前玩家在根节点的最优首步与收益差。"""
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("Depth must be a nonnegative integer")
    if algorithm not in ("minimax", "alpha_beta") or order not in ("natural", "reverse", "gain"):
        raise ValueError("Unknown algorithm or traversal order")
    state = state or GameState()
    began = perf_counter()
    visited = leaves = pruned = 0

    def evaluate(current: GameState, remaining_depth: int, alpha: float, beta: float) -> tuple[int, str | None]:
        nonlocal visited, leaves, pruned
        visited += 1
        # H 表示还能选择几次宝藏；到 0 或宝藏取完时，用当前累计收益差评价。
        if remaining_depth == 0 or not current.remaining:
            leaves += 1
            return current.value, None
        maximizing = current.player == "A"
        best = -inf if maximizing else inf
        best_target = None
        if order == "gain":
            # 实验用排序：优先看即时收益高的目标；最终值仍由完整递归决定。
            source = current.a_position if maximizing else current.b_position
            options = tuple(sorted(current.remaining, key=lambda target: values[target] - matrix[source][target], reverse=True))
        else:
            options = current.remaining if order == "natural" else tuple(reversed(current.remaining))
        for index, target in enumerate(options):
            # 每选择一个宝藏，行动次数减 1；take 同时移动玩家、累计收益并换手。
            value, _ = evaluate(take(current, target, matrix, values=values), remaining_depth - 1, alpha, beta)
            # 严格大于/小于才替换，因而并列时保留当前遍历顺序中的第一个目标。
            if best_target is None or (value > best if maximizing else value < best):
                best, best_target = value, target
            if algorithm == "alpha_beta":
                # 与题三相同的界限更新；界限交叉后，剩余选择不影响根节点决策。
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    pruned += len(options) - index - 1
                    break
        return int(best), best_target

    value, target = evaluate(state, depth, -inf, inf)
    return GameResult(target, value, visited, leaves, pruned, perf_counter() - began, depth, algorithm, order)


def best_treasure_choice(matrix: dict[str, dict[str, int | float]], depth: int,
                         *, algorithm: str = "alpha_beta") -> dict[str, str | int | None]:
    """题四的简洁输出接口：返回 PDF 样例中的两个字段。"""
    result = solve(matrix, depth, algorithm=algorithm)
    return {"best_target": result.best_target, "best_value": result.best_value}
