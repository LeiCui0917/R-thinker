"""实时迷宫规则说明

1. 决策机制：双方按回合提交目标 junction（坐标），环境执行对应路径。
2. 决策预算：每方成功提交一次动作计 1 次；达到 `max_steps` 后不能继续行动。
3. 动态阻塞：对手当前位置与对手历史轨迹会形成阻塞（起点区/终点区除外）。
4. 动作语义：合法动作是“下一跳目标 junction”，格式为坐标对 `(x, y)`。
5. 胜负判定：先到达终点区者胜；双方同时到达为平局；双方预算耗尽为超时。
6. 可扩展性：地图规模、环路密度、决策节奏均可通过参数控制。

状态帧格式说明：
- `frame_state()` 输出墙路底图、双方位置方向、起终点锚点坐标、轨迹与折叠连通边。
- `collapsed_connectivity` 将起点区折叠为 `(2,2)`、终点区折叠为 `(28,28)`，用于前端/Agent 拓扑观察。

实时迷宫环境类
"""

import random
import time
from collections import deque
from typing import Tuple, List, Dict, Optional
from .maze_builder import build_maze_grid, default_start_goal_zones, COMPASS, DIRECTIONS
from .maze_pathing import _compute_junctions, compute_next_junction_paths



# 迷宫生成与起终点区域打通由 `maze_builder.py` 负责；本文件只消费生成结果并处理运行时规则。

# =============================
# RealtimeMazeEnv
# =============================

class RealtimeMazeEnv:
    """实时迷宫环境（Revised）。

    消费 builder 生成的地图与区域定义，处理 super-cell 区域语义、
    junction-target 动作、动态阻塞与终局判定。
    """

    ACTIONS = DIRECTIONS

    # -----------------------------
    # (0) 初始化
    # -----------------------------
    def __init__(
        self,
        maze_size,
        loop_probability: float,
        step_duration: float,
        trail_release_seconds: float = 0.0,
        max_steps=400,
        seed=None,
    ):
        self.random = random.Random(seed)

        if isinstance(maze_size, int):
            maze_size = (maze_size, maze_size)

        # 环境策略：默认启用环路。
        self.has_loops = True

        # 环路密度由显式参数控制。
        self.loop_prob = max(0.0, min(1.0, float(loop_probability)))

        self.maze_cells, self.W, self.H = build_maze_grid(
            maze_size=maze_size,
            has_loops=self.has_loops,
            loop_probability=self.loop_prob,
            rng=self.random,
        )
        self.spawn_zone, self.goal_zone = default_start_goal_zones(self.W, self.H)
        self.ascii_map = "\n".join(
            "".join('#' if self.maze_cells[x][y] == 1 else '.' for x in range(self.W))
            for y in range(self.H)
        )

        # 3x3 区域锚点固定为中心坐标。
        self.spawn_super_cell: Tuple[int, int] = (2, 2)
        self.goal_super_cell: Tuple[int, int] = (self.W - 3, self.H - 3)

        # 统一锚点字段。
        self.start_cell: Tuple[int, int] = self.spawn_super_cell
        self.goal_cell: Tuple[int, int] = self.goal_super_cell
        self.red_start, self.blue_start = (3, 2), (2, 3)

        self.max_steps = int(max_steps)
        self.step_duration = float(step_duration)
        self.trail_release_seconds = max(0.0, float(trail_release_seconds))
        self.red_steps = 0
        self.blue_steps = 0

        self.red: Tuple[int, int] = self.red_start
        self.blue: Tuple[int, int] = self.blue_start

        # 已走轨迹集合（用于动态阻塞）。
        self.red_visited: set[Tuple[int, int]] = set()
        self.blue_visited: set[Tuple[int, int]] = set()
        self.red_visit_events: deque[tuple[Tuple[int, int], float]] = deque()
        self.blue_visit_events: deque[tuple[Tuple[int, int], float]] = deque()
        self.red_visit_counts: dict[Tuple[int, int], int] = {}
        self.blue_visit_counts: dict[Tuple[int, int], int] = {}

        # 有序轨迹（保留历史顺序，供状态展示与分析）。
        self.red_trace: list[Tuple[int, int]] = []
        self.blue_trace: list[Tuple[int, int]] = []

        # Pause control: time-based trail release should not elapse during pause.
        self._paused = False
        self._pause_started = None
        self._pause_total = 0.0

        self.reset()

    # =========================================================
    # 1. State Extraction and Combination
    # =========================================================
    def render_ascii(self) -> str:
        """渲染 ASCII 迷宫视图（仅显示墙路与双方当前位置/轨迹）。"""
        self._evict_expired_trails()
        grid = [
            [
                '#' if self.maze_cells[x][y] == 1 else '.'
                for x in range(self.W)
            ]
            for y in range(self.H)
        ]

        # 轨迹只覆盖通路格（'.'）。
        for (x, y) in self.red_visited:
            if 0 <= x < self.W and 0 <= y < self.H and grid[y][x] == '.':
                grid[y][x] = 'R'
        for (x, y) in self.blue_visited:
            if 0 <= x < self.W and 0 <= y < self.H and grid[y][x] == '.':
                grid[y][x] = 'B'

        rx, ry = self.red
        bx, by = self.blue
        if 0 <= rx < self.W and 0 <= ry < self.H:
            grid[ry][rx] = 'R'
        if 0 <= bx < self.W and 0 <= by < self.H:
            grid[by][bx] = 'B'

        return "\n".join("".join(row) for row in grid)

    def frame_state(self) -> Dict[str, object]:
        """输出统一状态帧（供前端与 Agent 使用）。"""
        self._evict_expired_trails()
        rx, ry = self.red
        cx, cy = self.blue
        goal_x, goal_y = self.goal_cell

        red_visited_visible = [
            {"x": x, "y": y}
            for (x, y) in self.red_visited
            if (x, y) not in self.spawn_zone
        ]
        blue_visited_visible = [
            {"x": x, "y": y}
            for (x, y) in self.blue_visited
            if (x, y) not in self.spawn_zone
        ]
        red_trace_visible = [
            {"x": x, "y": y}
            for (x, y) in self.red_trace
            if (x, y) not in self.spawn_zone
        ]
        blue_trace_visible = [
            {"x": x, "y": y}
            for (x, y) in self.blue_trace
            if (x, y) not in self.spawn_zone
        ]

        return {
            "w": self.W,
            "h": self.H,
            "map": self.ascii_map,
            "red": {"x": rx, "y": ry, "dir": self.red_dir},
            "blue": {"x": cx, "y": cy, "dir": self.blue_dir},
            "goal": {"x": goal_x, "y": goal_y},
            "start_zone": [{"x": self.spawn_super_cell[0], "y": self.spawn_super_cell[1], "id": "start"}],
            "goal_zone": [{"x": self.goal_super_cell[0], "y": self.goal_super_cell[1], "id": "goal"}],
            # 同时保留完整区域点集，供需要几何信息的模块使用。
            "start_zone_cells": [{"x": x, "y": y} for (x, y) in sorted(self.spawn_zone)],
            "goal_zone_cells": [{"x": x, "y": y} for (x, y) in sorted(self.goal_zone)],
            "collapsed_connectivity": self._collapsed_connectivity_edges(),
            "visited": {
                "red": red_visited_visible,
                "blue": blue_visited_visible,
            },
            "trace": {
                "red": red_trace_visible,
                "blue": blue_trace_visible,
            },
        }

    def _shortest_dirs_within_zone(
        self,
        zone: set[tuple[int, int]],
        start_cell: tuple[int, int],
        goal_cell: tuple[int, int],
    ) -> list[str] | None:
        """在同一区域内部求最短方向序列（BFS）。"""
        if start_cell == goal_cell:
            return []
        if start_cell not in zone or goal_cell not in zone:
            return None

        q = deque([start_cell])
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen: set[tuple[int, int]] = {start_cell}

        while q:
            cx, cy = q.popleft()
            for d in self.ACTIONS:
                dx, dy = COMPASS[d]
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if nxt not in zone or nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = ((cx, cy), d)
                if nxt == goal_cell:
                    dirs_rev: list[str] = []
                    cur = nxt
                    while cur != start_cell:
                        parent, step_dir = prev[cur]
                        dirs_rev.append(step_dir)
                        cur = parent
                    dirs_rev.reverse()
                    return dirs_rev
                q.append(nxt)
        return None

    def _zone_frontier_cells(self, zone: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """返回 zone 中用于出区枚举的 5 个前沿格。"""
        if zone is self.spawn_zone:
            return sorted((x, y) for (x, y) in zone if x == 3 or y == 3)
        if zone is self.goal_zone:
            edge_x = self.W - 4
            edge_y = self.H - 4
            return sorted((x, y) for (x, y) in zone if x == edge_x or y == edge_y)
        return sorted(zone)

    def _enumerate_zone_exits(
        self,
        role: str,
        zone: set[tuple[int, int]],
        src: tuple[int, int],
    ) -> list[tuple[tuple[int, int], list[str]]]:
        """枚举 zone 内前沿格的全部可出区点，返回 (exit_cell, zone_prefix_dirs)。"""
        if src not in zone:
            return []

        frontier_cells = self._zone_frontier_cells(zone)
        out: list[tuple[tuple[int, int], list[str]]] = []
        seen_exits: set[tuple[int, int]] = set()
        for zx, zy in frontier_cells:
            zone_prefix = self._shortest_dirs_within_zone(zone, src, (zx, zy))
            if zone_prefix is None:
                continue
            for d in self.ACTIONS:
                dx, dy = COMPASS[d]
                exit_cell = (zx + dx, zy + dy)
                if exit_cell in zone:
                    continue
                if self._is_blocked(role, exit_cell):
                    continue
                if exit_cell in seen_exits:
                    continue
                seen_exits.add(exit_cell)
                out.append((exit_cell, zone_prefix + [d]))
        return out

    def _current_zone(self, cell: tuple[int, int]) -> set[tuple[int, int]] | None:
        if cell in self.spawn_zone:
            return self.spawn_zone
        if cell in self.goal_zone:
            return self.goal_zone
        return None

    def _blocked_cells_for_role(self, role: str) -> set[tuple[int, int]]:
        self._evict_expired_trails()
        shared_zone_cells = set(self.spawn_zone) | set(self.goal_zone)
        if role == "red":
            blocked_cells = set(self.blue_visited) - shared_zone_cells
            blocked_cells.add(self.blue)
        else:
            blocked_cells = set(self.red_visited) - shared_zone_cells
            blocked_cells.add(self.red)
        return blocked_cells - shared_zone_cells

    def _zone_target_paths(
        self,
        role: str,
        zone: set[tuple[int, int]],
        src: tuple[int, int],
        ascii_map: str,
        blocked_cells: set[tuple[int, int]],
        targets: set[tuple[int, int]] | None,
    ) -> dict[tuple[int, int], list[str]]:
        best: dict[tuple[int, int], tuple[tuple[int, int, int], list[str]]] = {}
        for exit_cell, zone_prefix in self._enumerate_zone_exits(role, zone, src):
            next_paths = compute_next_junction_paths(
                ascii_map,
                exit_cell,
                blocked_cells=blocked_cells,
                targets=targets,
            )
            for cand_target, cand_path in next_paths.items():
                if not cand_path or cand_target in zone:
                    continue
                normalized_target = self.goal_super_cell if cand_target in self.goal_zone else cand_target
                score = (len(zone_prefix) + len(cand_path), cand_target[1], cand_target[0])
                full_path = zone_prefix + cand_path
                prev = best.get(normalized_target)
                if prev is None or score < prev[0]:
                    best[normalized_target] = (score, full_path)
        return {k: v[1] for k, v in best.items()}

    def _collapsed_connectivity_edges(self) -> list[str]:
        """构建折叠连通边集合（起点区折叠为起点坐标，终点区折叠为终点坐标）。"""
        # 先在原始 junction 图上求连通，再将端点按 super-cell 规则折叠。
        junctions = _compute_junctions(self.ascii_map)
        if not junctions:
            return []

        targets = set(self.spawn_zone) | set(self.goal_zone)
        edges: set[frozenset[tuple[int, int]]] = set()
        for j in junctions:
            next_paths = compute_next_junction_paths(
                self.ascii_map,
                j,
                targets=targets,
            )
            for target in next_paths.keys():
                if j in self.spawn_zone:
                    a = self.start_cell
                elif j in self.goal_zone:
                    a = self.goal_cell
                else:
                    a = j

                if target in self.spawn_zone:
                    b = self.start_cell
                elif target in self.goal_zone:
                    b = self.goal_cell
                else:
                    b = target

                if a == b:
                    continue
                edges.add(frozenset((a, b)))

        out: list[str] = []
        for edge in edges:
            a, b = tuple(edge)
            left = f"({a[0]},{a[1]})"
            right = f"({b[0]},{b[1]})"

            out.append(f"{left}-{right}")
        return out

    def _is_blocked(self, role: str, cell: Tuple[int, int]) -> bool:
        """统一阻塞判定：墙 + 对手当前位置 + 对手已走轨迹（起终点区除外）。"""
        self._evict_expired_trails()
        x, y = cell
        if not (0 <= x < self.W and 0 <= y < self.H):
            return True
        if self.maze_cells[x][y] != 0:
            return True
        if cell in self.spawn_zone or cell in self.goal_zone:
            return False
        if role == "red":
            if cell == self.blue:
                return True
            if cell in self.blue_visited:
                return True
        else:
            if cell == self.red:
                return True
            if cell in self.red_visited:
                return True
        return False

    # =========================================================
    # 2. Move Legality
    # =========================================================
    def _solver_snapshot(self) -> dict[str, tuple[int, int] | str]:
        self._evict_expired_trails()
        return {
            "map": self.ascii_map,
            "red": self.red,
            "blue": self.blue,
        }

    def get_legal_actions(self, role: str) -> List[tuple[int, int]]:
        # junction-target 语义：返回“下一跳目标 junction”候选（坐标对）。
        solver = self._solver_snapshot()
        sx, sy = solver[role]

        targets: set[tuple[int, int]] | None = set(self.goal_zone)
        blocked_cells = self._blocked_cells_for_role(role)

        opts: list[tuple[int, int]] = []

        src = (sx, sy)
        src_zone = self._current_zone(src)

        if src_zone is not None:
            # 在 super-cell 内：汇总所有可出区点可达外部路口，去重后全量返回。
            zone_paths = self._zone_target_paths(
                role=role,
                zone=src_zone,
                src=src,
                ascii_map=solver["map"],
                blocked_cells=blocked_cells,
                targets=targets,
            )
            ordered_targets = sorted(zone_paths.keys(), key=lambda p: (p[1], p[0]))
            for tx, ty in ordered_targets:
                opts.append((tx, ty))
        else:
            next_paths = compute_next_junction_paths(
                solver["map"],
                (sx, sy),
                blocked_cells=blocked_cells,
                targets=targets,
            )

            for (tx, ty), path in next_paths.items():
                if not path:
                    continue
                if (tx, ty) in self.spawn_zone:
                    continue
                if (tx, ty) in self.goal_zone:
                    tx, ty = self.goal_super_cell
                opts.append((tx, ty))

        return opts

    # =========================================================
    # 3. Move Execution
    # =========================================================
    def apply_action(self, role: str, move) -> bool:
        # 每方独立决策预算：每次成功动作计 1 次。
        if role == "red":
            if self.red_steps >= self.max_steps:
                return False
            src = self.red
        else:
            if self.blue_steps >= self.max_steps:
                return False
            src = self.blue
        blocked_cells = self._blocked_cells_for_role(role)

        target = (int(move[0]), int(move[1]))

        solver = self._solver_snapshot()
        path = None

        src_zone = self._current_zone(src)
        if src_zone is not None:
            zone_paths = self._zone_target_paths(
                role=role,
                zone=src_zone,
                src=src,
                ascii_map=solver["map"],
                blocked_cells=blocked_cells,
                targets=set(self.goal_zone),
            )
            path = zone_paths.get(target)
        else:
            next_paths = compute_next_junction_paths(
                solver["map"],
                src,
                blocked_cells=blocked_cells,
                targets=set(self.goal_zone),
            )

            path = next_paths.get(target)
            if not path and target == self.goal_super_cell:
                goal_candidates = [(t, p) for t, p in next_paths.items() if t in self.goal_zone and p]
                if goal_candidates:
                    goal_candidates.sort(key=lambda it: (len(it[1]), it[0][1], it[0][0]))
                    path = goal_candidates[0][1]

        if not path:
            return False

        step_fn = self._step_red if role == "red" else self._step_blue
        for step_dir in path:
            dx, dy = COMPASS[step_dir]
            px, py = self.red if role == "red" else self.blue
            dst = (px + dx, py + dy)

            if self._is_blocked(role, dst):
                return False
            step_fn(step_dir, dst)

        if role == "red":
            self.red_steps += 1
        else:
            self.blue_steps += 1
        return True

    def _step_red(self, action: str, dst: tuple[int, int] | None = None) -> None:
        """执行红方单步并更新轨迹。"""
        if dst is None:
            rx, ry = self.red
            dx, dy = COMPASS[action]
            dst = (rx + dx, ry + dy)
        self.red = dst
        self._record_visit("red", dst)
        self.red_trace.append(dst)
        self.red_dir = action

    def _step_blue(self, action: str, dst: tuple[int, int] | None = None) -> None:
        """执行蓝方单步并更新轨迹。"""
        if dst is None:
            cx, cy = self.blue
            dx, dy = COMPASS[action]
            dst = (cx + dx, cy + dy)
        self.blue = dst
        self._record_visit("blue", dst)
        self.blue_trace.append(dst)
        self.blue_dir = action

    def _record_visit(self, role: str, cell: tuple[int, int]) -> None:
        if role == "red":
            visited = self.red_visited
            events = self.red_visit_events
            counts = self.red_visit_counts
        else:
            visited = self.blue_visited
            events = self.blue_visit_events
            counts = self.blue_visit_counts

        visited.add(cell)
        if self.trail_release_seconds <= 0.0:
            return

        events.append((cell, self._now_effective()))
        counts[cell] = counts.get(cell, 0) + 1

    def _evict_expired_for_role(self, role: str, now_ts: float) -> None:
        if role == "red":
            events = self.red_visit_events
            counts = self.red_visit_counts
            visited = self.red_visited
        else:
            events = self.blue_visit_events
            counts = self.blue_visit_counts
            visited = self.blue_visited

        release_before = now_ts - self.trail_release_seconds
        while events and events[0][1] <= release_before:
            cell, _ = events.popleft()
            old_count = counts.get(cell, 0)
            if old_count <= 1:
                counts.pop(cell, None)
                visited.discard(cell)
            else:
                counts[cell] = old_count - 1

    def _evict_expired_trails(self) -> None:
        if self.trail_release_seconds <= 0.0:
            return
        now_ts = self._now_effective()
        self._evict_expired_for_role("red", now_ts)
        self._evict_expired_for_role("blue", now_ts)

    # =========================================================
    # 4. Win/Loss and Status Judgment
    # =========================================================
    def check_game_status(self) -> Dict[str, Optional[object]]:
        red_success = self.red in self.goal_zone
        blue_success = self.blue in self.goal_zone

        status: Dict[str, Optional[object]] = {
            "game_over": False,
            "red_success": False,
            "blue_success": False,
            "winner": None,
            "reason": None,
            "event": None,
            "red_steps": self.red_steps,
            "blue_steps": self.blue_steps,
        }

        if red_success:
            status["red_success"] = True
        if blue_success:
            status["blue_success"] = True

        if red_success and blue_success:
            status.update({"game_over": True, "winner": "draw", "reason": "both_reached", "event": "draw"})
            return status
        if red_success:
            status.update({"game_over": True, "winner": "red", "reason": "red_reached", "event": "red_goal"})
            return status
        if blue_success:
            status.update({"game_over": True, "winner": "blue", "reason": "blue_reached", "event": "blue_goal"})
            return status
        # 超时：双方都耗尽决策预算。
        if (self.red_steps >= self.max_steps) and (self.blue_steps >= self.max_steps):
            status.update({"game_over": True, "winner": None, "reason": "timeout", "event": "timeout"})
            return status

        return status

    def get_termination_reason(self, status: Optional[Dict[str, object]] = None) -> str:
        """返回终局事件字符串（`red_goal`/`blue_goal`/`draw`/`timeout`）。"""
        if not status:
            status = self.check_game_status()
        event = status.get("event")
        return str(event) if event is not None else "running"

    # =========================================================
    # 5. Reset Game
    # =========================================================
    def reset(self) -> None:
        """重置对局状态到初始局面。"""
        self.red_steps = 0
        self.blue_steps = 0
        self.red_dir: Optional[str] = None
        self.blue_dir: Optional[str] = None
        self._init_red()
        self._init_blue()
        self.red_visited.clear()
        self.blue_visited.clear()
        self.red_visit_events.clear()
        self.blue_visit_events.clear()
        self.red_visit_counts.clear()
        self.blue_visit_counts.clear()
        self._record_visit("red", self.red)
        self._record_visit("blue", self.blue)

        self.red_trace = [self.red]
        self.blue_trace = [self.blue]

    def _init_red(self) -> None:
        """初始化红方位置。"""
        self.red = self.red_start

    def _init_blue(self) -> None:
        """初始化蓝方位置。"""
        self.blue = self.blue_start

    # =========================================================
    # 6. Pause / Resume Time Basis
    # =========================================================
    def _now_effective(self) -> float:
        """Effective monotonic time excluding accumulated pause duration."""
        t = time.monotonic()
        paused_total = self._pause_total + (
            max(0.0, t - self._pause_started)
            if self._paused and self._pause_started is not None
            else 0.0
        )
        return t - paused_total

    def pause(self) -> None:
        if not self._paused:
            self._paused = True
            self._pause_started = time.monotonic()

    def resume(self) -> None:
        if self._paused:
            if self._pause_started is not None:
                self._pause_total += max(0.0, time.monotonic() - self._pause_started)
            self._paused = False
            self._pause_started = None
