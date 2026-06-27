"""SlowOnly 变体（仅慢模块）。

机制说明：
- 每步只调用 `SlowThinkModule` 更新语义树，不主动让 LLM 直接给动作。
- 当当前状态与语义节点高度匹配时，优先尝试 direct execute。
- 若未命中则仅更新树并返回 None（静默决策）。

用途：
- 作为 ThinkAgent 的消融对照，用于评估“仅慢思考”效果。
"""

from __future__ import annotations

import json
from typing import Any

from Agent.R_Thinker.direct_execute_module import pick_forced_maze_target_from_subgoal_path, pick_forced_uci_from_subgoal
from Agent.R_Thinker.slow_think_module import SlowThinkModule
from Agent.R_Thinker.utils.semantic_tree_manager import apply_tree_patches, format_path_leaf_to_root, merge_tree
from Agent.R_Thinker.utils.tree_similarity import find_nearest_node, find_nearest_node_chess
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total

_EXACT_HIT_SCORE = 0.9999


def _root_desc_chess(player: str) -> str:
    p = (player or "").strip().lower()
    if p == "w":
        opponent_color = "Black"
        self_color = "White"
    elif p == "b":
        opponent_color = "White"
        self_color = "Black"
    else:
        opponent_color = "Opponent"
        self_color = "Our"
    return f"{opponent_color} King is captured by {self_color} piece at (x, n)"


def _maze_root_desc_from_frame(frame: dict, role: str) -> str:
    goal_a = "Red reaches Goal A"
    goal_b = "Blue reaches Goal B"
    try:
        g = frame.get("goal")
        if isinstance(g, dict) and ("x" in g) and ("y" in g):
            goal_a = f"Red reaches Goal A at ({int(g['x'])}, {int(g['y'])})"
    except Exception:
        pass
    try:
        g = frame.get("goal_b")
        if isinstance(g, dict) and ("x" in g) and ("y" in g):
            goal_b = f"Blue reaches Goal B at ({int(g['x'])}, {int(g['y'])})"
    except Exception:
        pass

    r = (role or "").strip().lower()
    if r == "red":
        return goal_a
    if r == "blue":
        return goal_b
    return goal_a


class SlowOnlyMaze:
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, role: str = "red", extra_instruction: str | None = None):
        prompt = str(base_prompt or "")
        if extra_instruction:
            prompt += "\n" + str(extra_instruction)
        self.inner = SlowThinkModule(
            api_setting,
            llm_settings,
            model_name,
            prompt,
            log_file,
            game_type="maze",
            log_agent="so",
            side=role,
        )
        self.role = role
        self._tree = "- 0:"
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
        self.last_action_source = "slow_only"

    def get_action(self, frame: Any, player=None, legal_moves=None):
        """Maze 单步流程：匹配节点→可选直执行→慢模块更新树。"""
        if legal_moves is None and isinstance(player, list):
            legal_moves = player
            player = None
        role = (player if isinstance(player, str) and player.strip() else self.role or "red").strip().lower()
        frame_obj = frame if isinstance(frame, dict) else {}
        state_str = json.dumps(frame_obj, ensure_ascii=False)

        # Root node aligned with ThinkAgent semantics.
        root_desc = _maze_root_desc_from_frame(frame_obj, role)
        self._tree = apply_tree_patches(self._tree, fix_nodes={"0": root_desc})

        opp_role = "blue" if role == "red" else "red"

        # Choose nearest node (including root) as the expansion target.
        nid, desc, score = find_nearest_node(
            state_str,
            self._tree,
            exclude_root=False,
            source_role=role,
            opponent_role=opp_role,
        )
        nid = (nid or "0").strip() or "0"
        desc = (desc or root_desc).strip()

        # Strict order:
        # 1) nearest on current tree
        # 2) threshold gate + direct-execute on current tree
        # 3) only when threshold not met, ask slow and merge tree updates
        if float(score) >= _EXACT_HIT_SCORE:
            try:
                subgoal_path = format_path_leaf_to_root(self._tree, nid)
                forced = pick_forced_maze_target_from_subgoal_path(frame_obj, role, subgoal_path, legal_moves)
                if forced is not None:
                    self.last_action_source = "direct_execute"
                    return forced
            except Exception:
                pass

        result = self.inner.get_guidance(
            state=frame_obj,
            player=role,
            guidance_line_tree_last=self._tree,
            expand_node_id=nid,
            expand_node_desc=desc,
        )
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        new_tree = (result or {}).get("tree", "") or ""
        fix_nodes = (result or {}).get("fix_nodes", {}) or {}
        self._tree = merge_tree(apply_tree_patches(self._tree, fix_nodes=fix_nodes), new_tree)

        return None


class SlowOnlyChess:
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, side: str | None = None, extra_instruction: str | None = None):
        prompt = str(base_prompt or "")
        if extra_instruction:
            prompt += "\n" + str(extra_instruction)
        self.inner = SlowThinkModule(
            api_setting,
            llm_settings,
            model_name,
            prompt,
            log_file,
            game_type="chess",
            log_agent="so",
            side=side,
        )
        self._tree = "- 0:"
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
        self.last_action_source = "slow_only"

    def get_action(self, enhanced_FEN_full: Any, color: str, legal_moves: list):
        """Chess 单步流程：匹配节点→可选直执行→慢模块更新树。"""
        state_str = str(enhanced_FEN_full)
        player = (color or "").strip().lower()

        # Root node aligned with ThinkAgent semantics.
        root_desc = _root_desc_chess(player)
        self._tree = apply_tree_patches(self._tree, fix_nodes={"0": root_desc})

        # Choose nearest node (including root) as the expansion target.
        nid, desc, score = find_nearest_node_chess(
            state_str,
            self._tree,
            exclude_root=False,
        )
        nid = (nid or "0").strip() or "0"
        desc = (desc or root_desc).strip()

        # Strict order:
        # 1) nearest on current tree
        # 2) threshold gate + direct-execute on current tree
        # 3) only when threshold not met, ask slow and merge tree updates
        if float(score) >= _EXACT_HIT_SCORE:
            try:
                subgoal_path = format_path_leaf_to_root(self._tree, nid)
                forced = pick_forced_uci_from_subgoal(state_str, subgoal_path, legal_moves, player)
                if forced is not None:
                    self.last_action_source = "direct_execute"
                    return forced
            except Exception:
                pass

        result = self.inner.get_guidance(
            state=enhanced_FEN_full,
            player=player,
            guidance_line_tree_last=self._tree,
            expand_node_id=nid,
            expand_node_desc=desc,
        )
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        new_tree = (result or {}).get("tree", "") or ""
        fix_nodes = (result or {}).get("fix_nodes", {}) or {}
        self._tree = merge_tree(apply_tree_patches(self._tree, fix_nodes=fix_nodes), new_tree)

        return None
