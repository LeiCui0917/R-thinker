from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple


COMPASS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIRECTIONS = ["N", "E", "S", "W"]
CARDINAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE_DIR = {"N": "S", "S": "N", "E": "W", "W": "E"}
TUNNEL_PAIRS = (("W", "E"), ("N", "S"))

CONNECTIVITY_MAX_ROUNDS = 64
ZERO_ONE_BFS_INF = 10**9


def default_start_goal_zones(maze_w: int, maze_h: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """返回默认起点/终点 3x3 区域定义（供 builder 与 env 共享）。"""
    spawn_zone = {
        (x, y)
        for x in range(1, 4)
        for y in range(1, 4)
        if 0 < x < maze_w - 1 and 0 < y < maze_h - 1
    }
    goal_zone = {
        (x, y)
        for x in range(maze_w - 4, maze_w - 1)
        for y in range(maze_h - 4, maze_h - 1)
        if 0 < x < maze_w - 1 and 0 < y < maze_h - 1
    }
    return spawn_zone, goal_zone


def build_maze_grid(
    maze_size: Tuple[int, int],
    has_loops: bool,
    loop_probability: float,
    rng: random.Random,
) -> tuple[List[List[int]], int, int]:
    """生成迷宫网格。

    逻辑顺序：先生成正常迷宫，再在 builder 内打通起点/终点区域。

    返回：
    - maze_cells：二维网格（0=通路，1=墙）
    - maze_w：网格宽度
    - maze_h：网格高度
    """

    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    if len(maze_size) != 2:
        raise ValueError(f"maze_size must be a tuple of length 2, got {maze_size!r}")
    if maze_size[0] <= 0 or maze_size[1] <= 0:
        raise ValueError(f"maze_size values must be positive, got {maze_size!r}")
    if not (0.0 <= float(loop_probability) <= 1.0):
        raise ValueError(f"loop_probability must be in [0,1], got {loop_probability!r}")

    maze_w = 2 * maze_size[0] + 1
    maze_h = 2 * maze_size[1] + 1

    # 初始全墙，后续通过 DFS 与打洞规则开通路径。
    maze_cells = [[1 for _ in range(maze_h)] for _ in range(maze_w)]

    def is_within_bound(x: int, y: int) -> bool:
        """判断坐标是否落在网格内。"""
        return 0 <= x < maze_w and 0 <= y < maze_h

    def ensure_all_open_connected(start: tuple[int, int] = (1, 1)) -> None:
        """确保所有通路格与起点连通（必要时最小代价开墙连接）。"""
        # 复杂度与设计约束：
        # - 这里使用 0-1 BFS（开墙代价=1，走通路代价=0）来寻找“最少开墙”的连通修复路径，
        #   目标是在保持迷宫结构的前提下，避免出现割裂通路岛。
        # - 这是生成阶段的一次性修复，优先保证结构合法性与稳定性，而不是绝对最短执行时间。
        w, h = maze_w, maze_h
        if w <= 2 or h <= 2:
            return

        sx, sy = start
        if not (0 <= sx < w and 0 <= sy < h):
            return
        if maze_cells[sx][sy] != 0:
            sx, sy = 1, 1
            if not (0 <= sx < w and 0 <= sy < h) or maze_cells[sx][sy] != 0:
                return

        def is_inner(x: int, y: int) -> bool:
            return 0 < x < w - 1 and 0 < y < h - 1

        def neighbors(x: int, y: int):
            for dx, dy in CARDINAL_STEPS:
                nx, ny = x + dx, y + dy
                if is_inner(nx, ny):
                    yield nx, ny

        def reachable_from_start() -> set[tuple[int, int]]:
            if maze_cells[sx][sy] != 0:
                return set()
            q = deque([(sx, sy)])
            seen: set[tuple[int, int]] = {(sx, sy)}
            while q:
                x, y = q.popleft()
                for nx, ny in neighbors(x, y):
                    if (nx, ny) in seen:
                        continue
                    if maze_cells[nx][ny] != 0:
                        continue
                    seen.add((nx, ny))
                    q.append((nx, ny))
            return seen

        def all_open_cells() -> set[tuple[int, int]]:
            out: set[tuple[int, int]] = set()
            for x in range(1, w - 1):
                col = maze_cells[x]
                for y in range(1, h - 1):
                    if col[y] == 0:
                        out.add((x, y))
            return out

        for _ in range(CONNECTIVITY_MAX_ROUNDS):
            reachable = reachable_from_start()
            all_open = all_open_cells()
            unreachable = all_open - reachable
            if not unreachable:
                return

            dist = [[ZERO_ONE_BFS_INF for _ in range(h)] for _ in range(w)]
            prev: list[list[tuple[int, int] | None]] = [[None for _ in range(h)] for _ in range(w)]
            dq: deque[tuple[int, int]] = deque()
            for x, y in reachable:
                dist[x][y] = 0
                dq.appendleft((x, y))

            target: tuple[int, int] | None = None
            while dq:
                x, y = dq.popleft()
                if (x, y) in unreachable and maze_cells[x][y] == 0:
                    target = (x, y)
                    break
                base = dist[x][y]
                for nx, ny in neighbors(x, y):
                    step_cost = 0 if maze_cells[nx][ny] == 0 else 1
                    nd = base + step_cost
                    if nd < dist[nx][ny]:
                        dist[nx][ny] = nd
                        prev[nx][ny] = (x, y)
                        if step_cost == 0:
                            dq.appendleft((nx, ny))
                        else:
                            dq.append((nx, ny))

            if target is None:
                return

            cx, cy = target
            while prev[cx][cy] is not None:
                if maze_cells[cx][cy] != 0:
                    maze_cells[cx][cy] = 0
                cx, cy = prev[cx][cy]

            tx, ty = target
            maze_cells[tx][ty] = 0

    def create_loops_constrained(percent: float) -> None:
        """受约束打洞增加环路，避免破坏迷宫走廊结构。"""
        # 复杂度与设计约束：
        # - 限制 2x2 全通块，是为了保留“走廊/拐点”拓扑，避免地图过度空旷导致策略退化。
        # - 同时检查局部孤立墙岛，减少视觉和结构噪声，保持可解释性。
        candidates = []
        for x in range(1, maze_w - 1):
            for y in range(1, maze_h - 1):
                if maze_cells[x][y] != 1:
                    continue
                for d1, d2 in TUNNEL_PAIRS:
                    dx1, dy1 = COMPASS[d1]
                    dx2, dy2 = COMPASS[d2]
                    if maze_cells[x + dx1][y + dy1] == 0 and maze_cells[x + dx2][y + dy2] == 0:
                        candidates.append((x, y))
                        break

        def would_create_2x2_open_square(cx: int, cy: int) -> bool:
            for ox in (-1, 0):
                for oy in (-1, 0):
                    xs = [cx + ox, cx + ox + 1]
                    ys = [cy + oy, cy + oy + 1]
                    ok = True
                    for xx in xs:
                        for yy in ys:
                            if not (0 <= xx < maze_w and 0 <= yy < maze_h):
                                ok = False
                                break
                            cell_open = (maze_cells[xx][yy] == 0) or (xx == cx and yy == cy)
                            if not cell_open:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        return True
            return False

        def creates_isolated_wall_island(cx: int, cy: int) -> bool:
            """检查开洞后是否在局部制造 1~2 格孤立墙岛。"""
            def is_open_cell(xx: int, yy: int) -> bool:
                if not (0 <= xx < maze_w and 0 <= yy < maze_h):
                    return False
                if xx == cx and yy == cy:
                    return True
                return maze_cells[xx][yy] == 0

            def is_wall_cell(xx: int, yy: int) -> bool:
                if not (0 <= xx < maze_w and 0 <= yy < maze_h):
                    return True
                return maze_cells[xx][yy] == 1 and not (xx == cx and yy == cy)

            neighbors = []
            for d in ("N", "E", "S", "W"):
                dx, dy = COMPASS[d]
                nx, ny = cx + dx, cy + dy
                if is_wall_cell(nx, ny):
                    neighbors.append((nx, ny))

            def has_single_wall_island() -> bool:
                for wx, wy in neighbors:
                    open_count = 0
                    for d in DIRECTIONS:
                        dx, dy = COMPASS[d]
                        if is_open_cell(wx + dx, wy + dy):
                            open_count += 1
                    if open_count == 4:
                        return True
                return False

            def has_double_wall_island() -> bool:
                for wx, wy in neighbors:
                    for d in DIRECTIONS:
                        dx, dy = COMPASS[d]
                        px, py = wx + dx, wy + dy
                        if not is_wall_cell(px, py):
                            continue

                        other_dirs = [dd for dd in DIRECTIONS if dd != d]

                        ok_pair = True
                        for dd in other_dirs:
                            ddx, ddy = COMPASS[dd]
                            if not is_open_cell(wx + ddx, wy + ddy):
                                ok_pair = False
                                break

                        if ok_pair:
                            opp = OPPOSITE_DIR[d]
                            for dd in (cand for cand in DIRECTIONS if cand != opp):
                                ddx, ddy = COMPASS[dd]
                                if not is_open_cell(px + ddx, py + ddy):
                                    ok_pair = False
                                    break

                        if ok_pair:
                            return True
                return False

            if has_single_wall_island():
                return True
            if has_double_wall_island():
                return True
            return False

        target = int(len(candidates) * percent)
        if target <= 0:
            return
        rng.shuffle(candidates)
        opened = 0
        for x, y in candidates:
            if opened >= target:
                break
            if would_create_2x2_open_square(x, y):
                continue
            if creates_isolated_wall_island(x, y):
                continue
            maze_cells[x][y] = 0
            opened += 1

    # DFS 主干：从固定起点 (1,1) 挖掘走廊。
    start_x, start_y = 1, 1
    maze_cells[start_x][start_y] = 0
    stack = [(start_x, start_y)]

    while stack:
        x, y = stack[-1]
        neighbors = []

        for direction in DIRECTIONS:
            dx, dy = COMPASS[direction]
            nx, ny = x + 2 * dx, y + 2 * dy
            if is_within_bound(nx, ny) and maze_cells[nx][ny] == 1:
                neighbors.append((direction, nx, ny))

        if neighbors:
            direction, nx, ny = rng.choice(neighbors)
            dx, dy = COMPASS[direction]
            wx, wy = x + dx, y + dy
            maze_cells[wx][wy] = 0
            maze_cells[nx][ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    if has_loops:
        create_loops_constrained(loop_probability)

    spawn_zone, goal_zone = default_start_goal_zones(maze_w, maze_h)
    for x, y in spawn_zone:
        maze_cells[x][y] = 0
    for x, y in goal_zone:
        maze_cells[x][y] = 0

    # 最终兜底：保证不存在与起点割裂的通路区域。
    ensure_all_open_connected(start=(1, 1))

    return maze_cells, maze_w, maze_h
