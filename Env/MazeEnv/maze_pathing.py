from __future__ import annotations

from typing import Any

_JUNC_DIRS: tuple[tuple[str, int, int], ...] = (
    ("N", 0, -1),
    ("E", 1, 0),
    ("S", 0, 1),
    ("W", -1, 0),
)


def _parse_used_edges(used_edges: Any) -> set[frozenset[tuple[int, int]]]:
    if not isinstance(used_edges, list):
        return set()
    out: set[frozenset[tuple[int, int]]] = set()
    for edge in used_edges:
        if not isinstance(edge, dict):
            continue
        a = edge.get("a")
        b = edge.get("b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        try:
            ax, ay = int(a.get("x")), int(a.get("y"))
            bx, by = int(b.get("x")), int(b.get("y"))
        except Exception:
            continue
        if abs(ax - bx) + abs(ay - by) != 1:
            continue
        out.add(frozenset(((ax, ay), (bx, by))))
    return out


def _parse_ascii_map(
    ascii_map: str,
    *,
    blocked_edges: set[frozenset[tuple[int, int]]] | None = None,
    blocked_cells: set[tuple[int, int]] | None = None,
):
    rows = (ascii_map or "").strip("\n").split("\n") if ascii_map else []
    h = len(rows)
    w = len(rows[0]) if h > 0 else 0
    blocked_edges = blocked_edges or set()
    blocked_cells = blocked_cells or set()

    def is_open(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h and rows[y][x] != "#" and (x, y) not in blocked_cells

    def edge_open(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return frozenset((a, b)) not in blocked_edges

    return w, h, is_open, edge_open


def _compute_junctions(
    ascii_map: str,
    *,
    blocked_edges: set[frozenset[tuple[int, int]]] | None = None,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """Junction = open cell with degree >= 3."""
    w, h, is_open, edge_open = _parse_ascii_map(
        ascii_map,
        blocked_edges=blocked_edges,
        blocked_cells=blocked_cells,
    )
    junctions: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            if not is_open(x, y):
                continue
            deg = 0
            for _, dx, dy in _JUNC_DIRS:
                nx, ny = x + dx, y + dy
                if is_open(nx, ny) and edge_open((x, y), (nx, ny)):
                    deg += 1
            if deg >= 3:
                junctions.add((x, y))
    return junctions


def compute_next_junction_paths(
    ascii_map: str,
    start: tuple[int, int],
    *,
    used_edges: Any | None = None,
    blocked_cells: set[tuple[int, int]] | None = None,
    targets: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], list[str]]:
    """Return legal next targets and their concrete N/E/S/W sequences."""
    blocked_edges = _parse_used_edges(used_edges)
    _, _, is_open, edge_open = _parse_ascii_map(
        ascii_map,
        blocked_edges=blocked_edges,
        blocked_cells=blocked_cells,
    )

    sx, sy = start
    if not is_open(sx, sy):
        return {}

    junctions = _compute_junctions(
        ascii_map,
        blocked_edges=blocked_edges,
        blocked_cells=blocked_cells,
    )

    def neighbors(x: int, y: int):
        out = []
        for d, dx, dy in _JUNC_DIRS:
            nx, ny = x + dx, y + dy
            if is_open(nx, ny) and edge_open((x, y), (nx, ny)):
                out.append((d, nx, ny))
        return out

    out: dict[tuple[int, int], list[str]] = {}
    for first_dir, dx, dy in _JUNC_DIRS:
        x1, y1 = sx + dx, sy + dy
        if not is_open(x1, y1) or not edge_open((sx, sy), (x1, y1)):
            continue

        prev = (sx, sy)
        curr = (x1, y1)
        moves: list[str] = [first_dir]
        visited = {prev}

        while True:
            if curr in visited:
                break
            visited.add(curr)

            if targets is not None and curr in targets:
                out[curr] = moves
                break

            if curr in junctions:
                out[curr] = moves
                break

            cx, cy = curr
            neigh = neighbors(cx, cy)
            if len(neigh) <= 1:
                break
            if len(neigh) >= 3:
                out[curr] = moves
                break

            nxt = None
            for d2, nx, ny in neigh:
                if (nx, ny) != prev:
                    nxt = (d2, nx, ny)
                    break
            if nxt is None:
                break

            d2, nx, ny = nxt
            moves.append(d2)
            prev = curr
            curr = (nx, ny)

    return out
