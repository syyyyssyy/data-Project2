"""四方向网格 A*：题一使用单位步长，题二使用进入格子的地形代价。"""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter
from typing import Callable, Sequence

Position = tuple[int, int]
Grid = Sequence[Sequence[object]]
Heuristic = Callable[[Position, Position], float]


@dataclass(frozen=True)
class SearchResult:
    """搜索结果；expanded 是有效出队数，generated 是入队数。"""
    path: list[Position]
    cost: int | float
    expanded: int
    generated: int
    elapsed_seconds: float
    heuristic: str


def manhattan(a: Position, b: Position) -> int:
    """忽略障碍时，四方向移动至少需要的步数。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero(a: Position, b: Position) -> int:
    return 0


def _weight(cell: object) -> int:
    """统一解释地形权值：数字为自身权值，起终点与宝藏格为 1。"""
    if isinstance(cell, bool):
        raise ValueError("Boolean terrain is invalid")
    if isinstance(cell, int) and cell > 0:
        return cell
    if isinstance(cell, str) and cell.isdecimal() and int(cell) > 0:
        return int(cell)
    if cell in ("S", "E", "A", "B") or (isinstance(cell, str) and cell.startswith("T") and cell[1:].isdecimal()):
        return 1
    raise ValueError(f"Invalid weighted terrain: {cell!r}")


def _validate(grid: Grid, start: Position, goal: Position, weighted: bool) -> None:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Grid must be nonempty and rectangular")
    for name, point in (("start", start), ("goal", goal)):
        if (not isinstance(point, tuple) or len(point) != 2 or
                any(not isinstance(v, int) or isinstance(v, bool) for v in point)):
            raise ValueError(f"{name} must be a (row, column) integer tuple")
        r, c = point
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            raise ValueError(f"{name} is out of bounds")
        if grid[r][c] == "X":
            raise ValueError(f"{name} is an obstacle")
    if weighted:
        for row in grid:
            for cell in row:
                if cell != "X":
                    _weight(cell)


def path_cost(grid: Grid, path: Sequence[Position], *, weighted: bool = False,
              free_goal: bool = False) -> int:
    """逐步检查路径并重新计算代价，独立验证 A* 给出的结果。"""
    if not path:
        return 0
    _validate(grid, path[0], path[-1], weighted)
    total = 0
    for previous, current in zip(path, path[1:]):
        if (not isinstance(current, tuple) or len(current) != 2 or
                any(not isinstance(v, int) or isinstance(v, bool) for v in current) or
                not (0 <= current[0] < len(grid) and 0 <= current[1] < len(grid[0])) or
                manhattan(previous, current) != 1 or grid[current[0]][current[1]] == "X"):
            raise ValueError("Path has a non-adjacent step or crosses an obstacle")
        total += 0 if free_goal and current == path[-1] else (_weight(grid[current[0]][current[1]]) if weighted else 1)
    return total


def search(grid: Grid, start: Position, goal: Position, *, weighted: bool = False,
           free_goal: bool = False, heuristic: str | Heuristic = "auto",
           tie_break: str = "fifo") -> SearchResult:
    """求最小代价路径；无解时路径为空，代价为无穷大。"""
    _validate(grid, start, goal, weighted)
    if tie_break not in ("fifo", "deeper"):
        raise ValueError("tie_break must be 'fifo' or 'deeper'")
    if free_goal and not weighted:
        raise ValueError("free_goal requires weighted terrain")
    if heuristic == "auto":
        # 非终点的每一步至少付出 minimum。若进入 E 免费，则最多只有最后一步不付费，
        # 所以 (曼哈顿距离 - 1) × minimum 仍是不会高估真实剩余代价的下界。
        minimum = min((_weight(cell) for row in grid for cell in row if cell != "X"), default=1) if weighted else 1
        h: Heuristic = (lambda a, b: max(manhattan(a, b) - (1 if free_goal else 0), 0) * minimum)
        name = "free_goal_lower_bound" if free_goal else "manhattan_lower_bound"
    elif heuristic in ("zero", "h0"):
        h, name = zero, "zero"
    elif heuristic == "manhattan":
        if weighted and free_goal:
            raise ValueError("Manhattan is inadmissible when entering the goal is free")
        minimum = min((_weight(cell) for row in grid for cell in row if cell != "X"), default=1) if weighted else 1
        h, name = lambda a, b: manhattan(a, b) * minimum, "manhattan"
    elif callable(heuristic):
        h, name = heuristic, getattr(heuristic, "__name__", "custom")
    else:
        raise ValueError(f"Unknown heuristic: {heuristic!r}")
    began = perf_counter()
    serial = count()
    # 堆元素依次为 (f=g+h, 同分优先级, 入队序号, g, 坐标)。默认同分优先级为 0。
    open_set: list[tuple[float, int, int, int, Position]] = [(h(start, goal), 0, next(serial), 0, start)]
    # best_g 保存目前找到的最低到达代价；parent 只记录该最低代价路径上的前驱。
    best_g: dict[Position, int] = {start: 0}
    parent: dict[Position, Position] = {}
    expanded, generated = 0, 1
    while open_set:
        _, _, _, g, current = heappop(open_set)
        # 堆中可能还留有此坐标先前较贵的条目；它已被更优路径替代，无需扩展。
        if g != best_g[current]:
            continue
        expanded += 1
        if current == goal:
            # 从终点沿 parent 逆向回到起点，再翻转成题目要求的起点到终点顺序。
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(path, g, expanded, generated, perf_counter() - began, name)
        r, c = current
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])) or grid[nr][nc] == "X":
                continue
            neighbor = (nr, nc)
            # 题一每步代价为 1；题二支付“进入邻格”的权值，进入 E 的最后一步为 0。
            step = 0 if weighted and free_goal and neighbor == goal else (_weight(grid[nr][nc]) if weighted else 1)
            proposed = g + step
            # 只有找到更便宜的路径才更新前驱并重新入队，保证后续按最新 g 搜索。
            if proposed < best_g.get(neighbor, inf):
                best_g[neighbor], parent[neighbor] = proposed, current
                secondary = -proposed if tie_break == "deeper" else 0
                heappush(open_set, (proposed + h(neighbor, goal), secondary, next(serial), proposed, neighbor))
                generated += 1
    return SearchResult([], inf, expanded, generated, perf_counter() - began, name)


def maze_path(start: Position, goal: Position, grid: Grid) -> list[Position]:
    """题一接口：直接返回包含起终点的最短路径。"""
    return search(grid, start, goal).path


def weighted_cost(start: Position, goal: Position, grid: Grid) -> int:
    """题二接口：直接返回最小代价；无解时返回 -1。"""
    result = search(grid, start, goal, weighted=True, free_goal=True)
    return -1 if not result.path else int(result.cost)
