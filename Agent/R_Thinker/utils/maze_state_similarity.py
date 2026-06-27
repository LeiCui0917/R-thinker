"""Maze 语义树节点相似度。

当前 Maze 逆语义树只支持两类节点：
- 根节点：`<Side> reaches the shared goal zone at (x, y)`
- 普通节点：`(a, b) to (x, y)`

因此本文件只围绕“抽取节点核心坐标 + 根据红蓝双方位置打分”展开。
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


# ============================================================
# 模块一：节点文本解析
# ============================================================

_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_ROOT_RE = re.compile(
    r"reaches\s+the\s+shared\s+goal\s+zone\s+at\s*(\(\s*\d+\s*,\s*\d+\s*\))",
    re.IGNORECASE,
)
_EDGE_RE = re.compile(
    r"^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*to\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$",
    re.IGNORECASE,
)


def _parse_semantic_xy(text: str) -> Optional[Tuple[int, int]]:
    """抽取语义节点的核心坐标。

    约定：
    - 对普通节点 `(a,b) to (x,y)`，取前驱点 `(a,b)` 作为匹配坐标
    - 对根节点，取目标点 `(x,y)`
    """
    text = (text or "").strip()
    if not text:
        return None

    m_edge = _EDGE_RE.match(text)
    if m_edge:
        return int(m_edge.group(1)), int(m_edge.group(2))

    m_root = _ROOT_RE.search(text)
    if m_root:
        m_coord = _COORD_RE.search(m_root.group(1))
        if m_coord:
            return int(m_coord.group(1)), int(m_coord.group(2))

    return None


# ============================================================
# 模块二：状态提取
# ============================================================

def _maze_role_xy(state: Dict, role: str) -> Optional[Tuple[int, int]]:
    """从状态中提取某一方当前所在格子。"""
    if not isinstance(state, dict) or not role:
        return None
    obj = state.get(role) or {}
    if not isinstance(obj, dict):
        return None
    x, y = obj.get("x"), obj.get("y")
    if x is None or y is None:
        return None
    try:
        return int(x), int(y)
    except Exception:
        return None


def _maze_role_blocked_cells(state: Dict, role: str) -> set[Tuple[int, int]]:
    """收集某一方已经占用/访问过的格子。"""
    blocked: set[Tuple[int, int]] = set()
    if not isinstance(state, dict) or not role:
        return blocked

    xy = _maze_role_xy(state, role)
    if xy is not None:
        blocked.add(xy)

    visited = state.get("visited")
    if isinstance(visited, dict):
        role_vis = visited.get(role)
        if isinstance(role_vis, list):
            for p in role_vis:
                if not isinstance(p, dict):
                    continue
                vx, vy = p.get("x"), p.get("y")
                if vx is None or vy is None:
                    continue
                try:
                    blocked.add((int(vx), int(vy)))
                except Exception:
                    continue
    return blocked


def _maze_parent_id(node_id: Optional[str]) -> Optional[str]:
    """返回父节点 id。"""
    if not node_id or not isinstance(node_id, str) or "." not in node_id:
        return None
    return node_id.rsplit(".", 1)[0]


# ============================================================
# 模块三：基础相似度
# ============================================================

def maze_similarity_two_side_weighted(
    state: Dict,
    semantic_text: str,
    threshold: float = 0.0,
    *,
    self_role: str = "red",
    opponent_role: Optional[str] = None,
    w_self: float = 0.6,
    w_opp_far: float = 0.4,
) -> Tuple[float, bool]:
    """计算双边加权相似度。

    记：
    - self_close = 1 / (1 + dist(self, node))
    - opp_far    = dist(opp, node) / (1 + dist(opp, node))
    """
    if not isinstance(state, dict):
        return 0.0, False

    text = (semantic_text or "").strip()
    if not text:
        return 0.0, False

    node_xy = _parse_semantic_xy(text)
    if node_xy is None:
        return 0.0, False

    self_xy = _maze_role_xy(state, self_role)
    if self_xy is None:
        return 0.0, False

    if opponent_role is None:
        opponent_role = "blue" if self_role == "red" else "red"
    opp_xy = _maze_role_xy(state, opponent_role)

    dist_self = abs(self_xy[0] - node_xy[0]) + abs(self_xy[1] - node_xy[1])
    self_close = 1.0 / (1.0 + float(dist_self))

    if opp_xy is None:
        opp_far = 0.0
    else:
        dist_opp = abs(opp_xy[0] - node_xy[0]) + abs(opp_xy[1] - node_xy[1])
        opp_far = float(dist_opp) / (1.0 + float(dist_opp))

    score = float(w_self) * float(self_close) + float(w_opp_far) * float(opp_far)
    return float(score), bool(float(score) >= float(threshold))


# ============================================================
# 模块四：带父节点 blocking 的相似度
# ============================================================

def maze_similarity_with_parent_two_side_weighted(
    state: Dict,
    semantic_text: str,
    *,
    node_id: Optional[str],
    semantic_states: Dict[str, str],
    threshold: float = 0.0,
    self_role: str = "red",
    opponent_role: Optional[str] = None,
    w_self: float = 0.6,
    w_opp_far: float = 0.4,
    extra_blocked_cells: Optional[set[Tuple[int, int]]] = None,
) -> Tuple[float, bool]:
    """在基础相似度上加入 blocking 检查。

    语义：
    - `blocked_cells` 表示对 `self_role` 不可进入的格子
    - 默认来自对手已占用/访问的格子
    - `extra_blocked_cells` 用于补充额外禁入格
    """
    score, _ = maze_similarity_two_side_weighted(
        state,
        semantic_text,
        threshold=0.0,
        self_role=self_role,
        opponent_role=opponent_role,
        w_self=w_self,
        w_opp_far=w_opp_far,
    )
    if float(score) <= 0.0:
        return 0.0, False

    if opponent_role is None:
        opponent_role = "blue" if self_role == "red" else "red"

    blocked_cells = _maze_role_blocked_cells(state, opponent_role)
    if extra_blocked_cells:
        try:
            blocked_cells |= set(extra_blocked_cells)
        except Exception:
            pass

    curr_xy = _parse_semantic_xy(str(semantic_text))
    if blocked_cells and curr_xy is not None and curr_xy in blocked_cells:
        return 0.0, False

    parent_id = _maze_parent_id(node_id)
    if not parent_id:
        return float(score), bool(float(score) >= float(threshold))

    parent_text = semantic_states.get(parent_id)
    if not parent_text:
        return float(score), bool(float(score) >= float(threshold))

    parent_xy = _parse_semantic_xy(str(parent_text))
    if blocked_cells and parent_xy is not None and parent_xy in blocked_cells:
        return 0.0, False

    return float(score), bool(float(score) >= float(threshold))
