import heapq
import math
import re
import time


class RuleAgent:
    """
    Rule Agent (Maze): D* Lite over collapsed junction graph.
    - Build graph from frame['collapsed_connectivity']
    - Incrementally maintain shortest-path values toward frame['goal']
    - Pick best target from current legal junction moves
    """

    _EDGE_RE = re.compile(
        r"^\((?P<x1>-?\d+),(?P<y1>-?\d+)\)-\((?P<x2>-?\d+),(?P<y2>-?\d+)\)$"
    )

    def __init__(self, env, delay_s: float = 0.0):
        self.env = env
        self.delay_s = max(0.0, float(delay_s or 0.0))
        self.last_action_source = "rule"
        self._adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self._graph_sig: tuple[str, ...] = ()

        self._g: dict[tuple[int, int], float] = {}
        self._rhs: dict[tuple[int, int], float] = {}
        self._open: list[tuple[tuple[float, float], tuple[int, int]]] = []

        self._s_start: tuple[int, int] | None = None
        self._s_goal: tuple[int, int] | None = None
        self._s_last: tuple[int, int] | None = None
        self._km: float = 0.0

    def _extract_target(self, opt):
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            try:
                return (int(opt[0]), int(opt[1]))
            except Exception:
                return None
        return None

    def _coord_from_dict(self, obj):
        if not isinstance(obj, dict):
            return None
        try:
            return (int(obj['x']), int(obj['y']))
        except Exception:
            return None

    def _parse_edges(self, edge_lines):
        adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
        sig_parts: list[str] = []
        for s in edge_lines or []:
            line = str(s or '').strip()
            m = self._EDGE_RE.match(line)
            if not m:
                continue
            a = (int(m.group('x1')), int(m.group('y1')))
            b = (int(m.group('x2')), int(m.group('y2')))
            if a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
            left, right = (a, b) if a <= b else (b, a)
            sig_parts.append(f"{left}->{right}")
        return adj, tuple(sorted(sig_parts))

    def _h(self, a, b):
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _g_val(self, s):
        return float(self._g.get(s, math.inf))

    def _rhs_val(self, s):
        return float(self._rhs.get(s, math.inf))

    def _calculate_key(self, s):
        base = min(self._g_val(s), self._rhs_val(s))
        if self._s_start is None:
            return (math.inf, math.inf)
        return (base + self._h(self._s_start, s) + self._km, base)

    def _push_open(self, s):
        heapq.heappush(self._open, (self._calculate_key(s), s))

    def _pop_open(self):
        while self._open:
            key, node = heapq.heappop(self._open)
            if key == self._calculate_key(node):
                return key, node
        return None, None

    def _top_open_key(self):
        while self._open:
            key, node = self._open[0]
            if key == self._calculate_key(node):
                return key
            heapq.heappop(self._open)
        return (math.inf, math.inf)

    def _update_vertex(self, u):
        if self._s_goal is None:
            return
        if u != self._s_goal:
            best = math.inf
            for s in self._adj.get(u, ()):
                best = min(best, 1.0 + self._g_val(s))
            self._rhs[u] = best
        if self._g_val(u) != self._rhs_val(u):
            self._push_open(u)

    def _reset_planner(self, start, goal):
        self._s_start = start
        self._s_goal = goal
        self._s_last = start
        self._km = 0.0
        self._g = {}
        self._rhs = {}
        self._open = []
        self._rhs[goal] = 0.0
        self._push_open(goal)

    def _compute_shortest_path(self, max_expands=20000):
        if self._s_start is None:
            return
        expands = 0
        while (
            self._top_open_key() < self._calculate_key(self._s_start)
            or self._rhs_val(self._s_start) != self._g_val(self._s_start)
        ):
            expands += 1
            if expands > max_expands:
                break

            old_key, u = self._pop_open()
            if u is None:
                break

            new_key = self._calculate_key(u)
            if old_key < new_key:
                self._push_open(u)
                continue

            if self._g_val(u) > self._rhs_val(u):
                self._g[u] = self._rhs_val(u)
                for p in self._adj.get(u, ()):
                    self._update_vertex(p)
            else:
                self._g[u] = math.inf
                self._update_vertex(u)
                for p in self._adj.get(u, ()):
                    self._update_vertex(p)

    def _ensure_plan(self, start, goal, frame):
        adj, sig = self._parse_edges(frame.get('collapsed_connectivity', []))
        graph_changed = (sig != self._graph_sig)
        goal_changed = (goal != self._s_goal)

        if graph_changed:
            self._adj = adj
            self._graph_sig = sig

        if graph_changed or goal_changed or self._s_start is None:
            self._reset_planner(start, goal)
            self._compute_shortest_path()
            return

        if self._s_last is not None and start != self._s_last:
            self._km += self._h(self._s_last, start)
        self._s_start = start
        self._s_last = start
        self._compute_shortest_path()

    def get_action(self, frame=None, role=None, legal_moves=None):
        self.last_action_source = "rule"
        if not legal_moves:
            return None

        candidates = []
        for opt in legal_moves:
            t = self._extract_target(opt)
            if t is not None:
                candidates.append(t)

        if not candidates:
            return None

        move = None
        if frame:
            is_blue = (role == 'blue')
            agent_state = frame.get('blue') if is_blue else frame.get('red')
            start = self._coord_from_dict(agent_state)
            goal = self._coord_from_dict(frame.get('goal'))

            if start is not None and goal is not None:
                self._ensure_plan(start, goal, frame)

                def cost_to_goal(t):
                    if t == goal:
                        return 0.0
                    return self._g_val(t)

                move = min(candidates, key=lambda t: (1.0 + cost_to_goal(t), self._h(t, goal), t[1], t[0]))

        # Fallback when frame/graph info is unavailable.
        if move is None:
            move = candidates[0]

        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        return move
