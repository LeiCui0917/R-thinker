"""语义树节点相似度检索与打分工具。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .semantic_tree_manager import _parse_tree_nodes


# ============================================================
# 模块一：配置与输入归一化
# ============================================================

def _load_similarity_weights() -> tuple[float, float, float, float]:
    """从配置中读取 Maze/Chess 的默认相似度权重。"""
    defaults = {
        "maze": {"w_self": 0.6, "w_opp_far": 0.4},
        "chess": {"w_self": 0.6, "w_opp_far": 0.4},
    }

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent_config.json"))
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        weights = (raw or {}).get("think_similarity_weights", {}) or {}
        maze_cfg = weights.get("maze", {}) or {}
        chess_cfg = weights.get("chess", {}) or {}
        maze_w_self = float(maze_cfg.get("w_self", defaults["maze"]["w_self"]))
        maze_w_opp_far = float(maze_cfg.get("w_opp_far", defaults["maze"]["w_opp_far"]))
        chess_w_self = float(chess_cfg.get("w_self", defaults["chess"]["w_self"]))
        chess_w_opp_far = float(chess_cfg.get("w_opp_far", defaults["chess"]["w_opp_far"]))
        return maze_w_self, maze_w_opp_far, chess_w_self, chess_w_opp_far
    except Exception:
        return (
            float(defaults["maze"]["w_self"]),
            float(defaults["maze"]["w_opp_far"]),
            float(defaults["chess"]["w_self"]),
            float(defaults["chess"]["w_opp_far"]),
        )


_MAZE_W_SELF_DEFAULT, _MAZE_W_OPP_FAR_DEFAULT, _CHESS_W_SELF_DEFAULT, _CHESS_W_OPP_FAR_DEFAULT = _load_similarity_weights()


def _parse_tree_nodes_canonical(tree_text: str) -> Dict[str, str]:
    """统一复用 semantic_tree_manager 中的树行解析规则。"""
    return _parse_tree_nodes(str(tree_text or ""))


def _as_maze_state(state_or_str: Any) -> Optional[dict]:
    """把 Maze 状态归一化为 dict。"""
    if isinstance(state_or_str, dict):
        return state_or_str
    if isinstance(state_or_str, str):
        s = state_or_str.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# ============================================================
# 模块二：最近节点检索
# ============================================================

def find_nearest_node_chess(
    fen: str,
    tree_text: str,
    *,
    exclude_root: bool = True,
    w_self: float = _CHESS_W_SELF_DEFAULT,
    w_opp_far: float = _CHESS_W_OPP_FAR_DEFAULT,
) -> Tuple[Optional[str], Optional[str], float]:
    """在 Chess 语义树中寻找当前最相似节点。"""
    _ = (w_self, w_opp_far)
    nodes = _parse_tree_nodes_canonical(tree_text)
    if not nodes:
        return None, None, 0.0

    from .chess_state_similarity import chess_semantic_similarity_score

    best_id: Optional[str] = None
    best_desc: Optional[str] = None
    best_score = -1.0
    best_depth = 10**9
    for nid, desc in nodes.items():
        if exclude_root and nid == "0":
            continue
        score = float(chess_semantic_similarity_score(fen, desc))
        depth = nid.count(".")
        if (score > best_score) or (score == best_score and (best_id is None or depth < best_depth)):
            best_id, best_desc, best_score, best_depth = nid, desc, score, depth

    if best_id is None:
        return None, None, 0.0
    return best_id, best_desc, best_score


def find_nearest_node_chess_at_depth(
    fen: str,
    tree_text: str,
    target_depth: int,
    *,
    exclude_root: bool = True,
    w_self: float = _CHESS_W_SELF_DEFAULT,
    w_opp_far: float = _CHESS_W_OPP_FAR_DEFAULT,
) -> Tuple[Optional[str], Optional[str], float]:
    """在指定深度的 Chess 节点中寻找最相似节点。"""
    _ = (w_self, w_opp_far)
    nodes = _parse_tree_nodes_canonical(tree_text)
    if not nodes:
        return None, None, 0.0

    from .chess_state_similarity import chess_semantic_similarity_score

    best_id: Optional[str] = None
    best_desc: Optional[str] = None
    best_score = -1.0
    for nid, desc in nodes.items():
        if exclude_root and nid == "0":
            continue
        if nid.count(".") != int(target_depth):
            continue
        score = float(chess_semantic_similarity_score(fen, desc))
        if score > best_score:
            best_id, best_desc, best_score = nid, desc, score

    if best_id is None:
        return None, None, 0.0
    return best_id, best_desc, best_score


def find_nearest_node(
    state: Any,
    tree_text: str,
    *,
    exclude_root: bool = True,
    source_role: str = "red",
    opponent_role: Optional[str] = None,
    w_self: float = _MAZE_W_SELF_DEFAULT,
    w_opp_far: float = _MAZE_W_OPP_FAR_DEFAULT,
    extra_blocked_cells: Optional[set[tuple[int, int]]] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """在 Maze 语义树中寻找当前最相似节点。"""
    nodes = _parse_tree_nodes_canonical(tree_text)
    if not nodes:
        return None, None, 0.0

    maze_state = _as_maze_state(state)
    if maze_state is None:
        return None, None, 0.0

    from .maze_state_similarity import maze_similarity_with_parent_two_side_weighted

    best_id: Optional[str] = None
    best_desc: Optional[str] = None
    best_score = -1.0
    best_depth = 10**9
    for nid, desc in nodes.items():
        if exclude_root and nid == "0":
            continue
        score, _ = maze_similarity_with_parent_two_side_weighted(
            maze_state,
            desc,
            node_id=nid,
            semantic_states=nodes,
            threshold=0.0,
            self_role=source_role,
            opponent_role=opponent_role,
            w_self=w_self,
            w_opp_far=w_opp_far,
            extra_blocked_cells=extra_blocked_cells,
        )
        depth = nid.count(".")
        if (score > best_score) or (score == best_score and (best_id is None or depth < best_depth)):
            best_id, best_desc, best_score, best_depth = nid, desc, float(score), depth

    if best_id is None:
        return None, None, 0.0
    return best_id, best_desc, float(best_score)


# ============================================================
# 模块三：全量节点打分
# ============================================================

def score_tree_nodes_chess(
    fen: str,
    tree_text: str,
    *,
    exclude_root: bool = True,
    top_k: Optional[int] = None,
    w_self: float = _CHESS_W_SELF_DEFAULT,
    w_opp_far: float = _CHESS_W_OPP_FAR_DEFAULT,
) -> List[Dict[str, Any]]:
    """给 Chess 语义树中的每个节点打分。"""
    _ = (w_self, w_opp_far)
    nodes = _parse_tree_nodes_canonical(tree_text)
    if not nodes:
        return []

    from .chess_state_similarity import chess_semantic_similarity_score

    out: List[Dict[str, Any]] = []
    for nid, desc in nodes.items():
        if exclude_root and nid == "0":
            continue
        score = float(chess_semantic_similarity_score(fen, desc))
        out.append({"id": nid, "score": float(score), "depth": nid.count(".")})

    out.sort(key=lambda r: (float(r.get("score", 0.0)), int(r.get("depth", 0))), reverse=True)
    if top_k is not None:
        try:
            k = int(top_k)
        except Exception:
            k = None
        if k is not None and k >= 0:
            out = out[:k]
    return out


def score_tree_nodes_maze(
    state: Any,
    tree_text: str,
    *,
    exclude_root: bool = True,
    top_k: Optional[int] = None,
    source_role: str = "red",
    opponent_role: Optional[str] = None,
    w_self: float = _MAZE_W_SELF_DEFAULT,
    w_opp_far: float = _MAZE_W_OPP_FAR_DEFAULT,
    extra_blocked_cells: Optional[set[tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """给 Maze 语义树中的每个节点打分。"""
    nodes = _parse_tree_nodes_canonical(tree_text)
    if not nodes:
        return []

    maze_state = _as_maze_state(state)
    if maze_state is None:
        return []

    from .maze_state_similarity import maze_similarity_with_parent_two_side_weighted

    out: List[Dict[str, Any]] = []
    for nid, desc in nodes.items():
        if exclude_root and nid == "0":
            continue
        score, _ = maze_similarity_with_parent_two_side_weighted(
            maze_state,
            desc,
            node_id=nid,
            semantic_states=nodes,
            threshold=0.0,
            self_role=source_role,
            opponent_role=opponent_role,
            w_self=w_self,
            w_opp_far=w_opp_far,
            extra_blocked_cells=extra_blocked_cells,
        )
        out.append({"id": nid, "score": float(score), "depth": nid.count(".")})

    out.sort(key=lambda r: (float(r.get("score", 0.0)), int(r.get("depth", 0))), reverse=True)
    if top_k is not None:
        try:
            k = int(top_k)
        except Exception:
            k = None
        if k is not None and k >= 0:
            out = out[:k]
    return out
