"""Agent/utils/maze_translator.py

【模块：翻译器（Maze）】

职责：
- 将实时迷宫 `frame` 翻译为供提示词使用的文本描述。
- 输出只包含两类信息：
    1) 路口连通图（无向边）
    2) 红蓝双方轨迹（路口级压缩表示）

说明：
- 本文件偏“展示/提示词辅助”，不参与动作决策。
- 当前版本按项目约定仅使用 `collapsed_connectivity` 作为图来源。
"""


class MazeTranslator:
    """把迷宫帧翻译成自然语言描述。"""

    DIRS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0),
    }

    def _zone_cells_for(self, frame: dict, zone_name: str) -> set[tuple[int, int]]:
        """读取某类区域（start/goal）覆盖的网格集合。"""

        def _parse_zone_cells(key: str) -> list[tuple[int, int]]:
            zone = frame[key]
            out = [(int(cell["x"]), int(cell["y"])) for cell in zone]
            return sorted(set(out), key=lambda p: (p[1], p[0]))

        cells = _parse_zone_cells(f"{zone_name}_zone_cells")
        return set(cells)

    def translate(self, frame: dict, role: str | None = None) -> str:
        """将迷宫帧翻译成多行文本描述。"""

        ascii_map = frame["map"]

        red = frame["red"]
        blue = frame["blue"]
        start_cells = self._zone_cells_for(frame, "start")
        goal_cells = self._zone_cells_for(frame, "goal")

        start_anchor = (int(frame["start_zone"][0]["x"]), int(frame["start_zone"][0]["y"]))
        goal_anchor = (int(frame["goal_zone"][0]["x"]), int(frame["goal_zone"][0]["y"]))

        def collapse_node(cell: tuple[int, int]) -> tuple[int, int]:
            if cell in start_cells:
                return start_anchor
            if cell in goal_cells:
                return goal_anchor
            return cell

        def format_node(node: tuple[int, int]) -> str:
            return f"({node[0]},{node[1]})"

        def trace_to_xy_list(key: str) -> list[tuple[int, int]]:
            pts = frame["trace"][key]
            return [(int(p["x"]), int(p["y"])) for p in pts]

        def fmt_now_pos(label: str, obj: dict) -> str | None:
            x, y = int(obj["x"]), int(obj["y"])
            node = collapse_node((x, y))
            return f"{label} at {format_node(node)} Now"

        red_now = fmt_now_pos("Red Red", red)
        blue_now = fmt_now_pos("Blue Red", blue)

        rows = (ascii_map or "").strip().split("\n")
        h = len(rows)
        w = len(rows[0]) if h > 0 else 0

        def is_open_static(px: int, py: int) -> bool:
            return 0 <= px < w and 0 <= py < h and rows[py][px] != "#"

        def neighbors_static(px: int, py: int) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for dx, dy in self.DIRS.values():
                nx, ny = px + dx, py + dy
                if is_open_static(nx, ny):
                    out.append((nx, ny))
            return out

        junctions_static = {
            (x, y)
            for y in range(h)
            for x in range(w)
            if is_open_static(x, y) and len(neighbors_static(x, y)) >= 3
        }

        def compress_to_junction_sequence(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
            goals_local = set(goal_cells)
            seq: list[tuple[int, int]] = []
            for cell in path:
                if cell in junctions_static or cell in goals_local:
                    collapsed = collapse_node(cell)
                    if not seq or seq[-1] != collapsed:
                        seq.append(collapsed)
            return seq

        def fmt_junction_path(label: str, seq: list[tuple[int, int]]) -> str:
            s = " -> ".join(format_node(n) for n in seq)
            return f"{label} trajectory (junction-to-junction): {s}"

        lines: list[str] = []
        lines.append(f"{red_now}; {blue_now}")

        lines.append(f"Start zone anchor: {format_node(start_anchor)}; Goal zone anchor: {format_node(goal_anchor)}")
        lines.append(fmt_junction_path("Red", compress_to_junction_sequence(trace_to_xy_list("red"))))
        lines.append(fmt_junction_path("Blue", compress_to_junction_sequence(trace_to_xy_list("blue"))))

        graph_edges_text: list[str] = []
        # 图来源固定：仅使用环境提前折叠好的 collapsed_connectivity。
        collapsed = frame["collapsed_connectivity"]

        for edge in collapsed:
            text = edge.strip()
            graph_edges_text.append(text)

        lines.append("Junction Connectivity (undirected, includes start/goal anchor coordinates):")
        lines.extend(graph_edges_text)

        return "\n".join(lines)
