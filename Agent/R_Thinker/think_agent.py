"""ThinkAgent（四模块协同：快决策 + 双慢更新 + 选点调度）。

整体机制：
1) 快模块：根据当前 subgoal 直接产出动作（前台主路径）。
2) 慢模块-我方：在后台扩展“我方语义树”。
3) 慢模块-敌方：在后台扩展“敌方语义树”。
4) 调度模块：融合两棵树并按 k/(k-1) 规则选点，给快模块提供 subgoal。

单步流程：
- 先基于缓存树选出本步 subgoal。
- 若可 direct-execute 则直接返回动作。
- 否则异步触发慢模块更新，并由快模块完成本步决策。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from .fast_think_module import FastThinkModule
from .slow_think_module import SlowThinkModule
from .direct_execute_module import pick_forced_uci_from_subgoal, pick_forced_maze_target_from_subgoal_path
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage
from .utils.semantic_tree_manager import (
    _parse_tree_nodes,
    apply_tree_patches,
    format_path_leaf_to_root,
    merge_tree,
)
from .utils.tree_similarity import (
    find_nearest_node,
    find_nearest_node_chess,
    find_nearest_node_chess_at_depth,
    score_tree_nodes_chess,
    score_tree_nodes_maze,
)
from .utils.chess_semantic_grounding import ground_chess_semantic_text
from .utils.maze_semantic_grounding import ground_maze_semantic_text


def _get_leaf_node_ids(tree_text: str) -> list[str]:
    nodes = _parse_tree_nodes(tree_text)
    if not nodes:
        return []
    children = set()
    for nid in nodes:
        if "." in nid:
            children.add(nid.rsplit(".", 1)[0])
    return [nid for nid in nodes if nid not in children]


class ThinkAgent:
    """ThinkAgent 主调度器：维护树缓存、触发慢更新并驱动快决策。

    设计目标：
    1) 前台 `get_action` 必须尽快返回，不被慢思考阻塞。
    2) 后台慢模块持续维护两棵语义树（我方/敌方），给下一步提供更好 subgoal。
    3) 通过 direct-execute 规则尽量减少不必要的 LLM 调用。

    你可以把它理解成：
    - 快模块 = "当前这一步怎么走"（即时执行）
    - 慢模块 = "长期结构怎么长"（异步更新）
    - 调度模块 = "这一步更该进攻还是防守"（双树选点）
    """

    def __init__(
        self,
        api_setting: dict,
        llm_settings: dict,
        fast_module_prompt: str,
        slow_module_prompt: str,
        log_file: str,
        model_name: str = "",
        game_type: str = "chess",
        slow_module_opponent_prompt: str = "",
        log_agent: str = "think",
        side: str | None = None,
        enable_self_slow: bool = True,
        enable_opp_slow: bool = True,
        decision_mode_policy: str = "auto",
    ):
        # 基本配置
        self.api_setting = api_setting
        self.llm_settings = llm_settings
        self.fast_prompt_template = fast_module_prompt
        self.slow_prompt_template = slow_module_prompt
        self.slow_opponent_prompt_template = slow_module_opponent_prompt or slow_module_prompt
        self._log_file = log_file
        self.game_type = str(game_type or "chess").lower()
        self._log_agent = str(log_agent or "think")
        self._side = str(side or "").strip().lower() or None
        self.enable_self_slow = bool(enable_self_slow)
        self.enable_opp_slow = bool(enable_opp_slow)
        self.decision_mode_policy = str(decision_mode_policy or "auto").strip().lower()
        if self.decision_mode_policy not in {"auto", "attack_only", "defend_only"}:
            self.decision_mode_policy = "auto"

        # 统一模型名（由调用方注入，严格要求非空）
        self._model_name = str(model_name or "").strip()
        if not self._model_name:
            raise ValueError("ThinkAgent requires explicit non-empty model_name")

        # 派生各模块日志文件路径（统一短名）
        self._fast_log_file = make_log_file(self._log_file, game=self.game_type, side=self._side, agent=self._log_agent, module="fast")
        self._slow_self_log_file = make_log_file(self._log_file, game=self.game_type, side=self._side, agent=self._log_agent, module="slow_self")
        self._slow_opp_log_file = make_log_file(self._log_file, game=self.game_type, side=self._side, agent=self._log_agent, module="slow_opp")
        self._slow_trees_json_file = make_log_file(self._log_file, game=self.game_type, side=self._side, agent=self._log_agent, module="summary", ext=".json")

        # ========== 模块 1：快模块实例 ==========
        self.fast_module = FastThinkModule(
            api_setting=self.api_setting,
            llm_settings=self.llm_settings,
            model_name=self._model_name,
            fast_module_prompt=self.fast_prompt_template,
            log_file=self._fast_log_file,
            game_type=self.game_type,
            log_agent=self._log_agent,
            side=self._side,
        )

        # ========== 模块 2/3：慢模块实例（自我/敌方） ==========
        self.slow_module = (
            SlowThinkModule(
                api_setting=self.api_setting,
                llm_settings=self.llm_settings,
                model_name=self._model_name,
                slow_module_prompt=self.slow_prompt_template,
                log_file=self._slow_self_log_file,
                game_type=self.game_type,
                log_agent=self._log_agent,
                side=self._side,
                log_module="slow_self",
            )
            if self.enable_self_slow
            else None
        )
        self.slow_module_opponent = (
            SlowThinkModule(
                api_setting=self.api_setting,
                llm_settings=self.llm_settings,
                model_name=self._model_name,
                slow_module_prompt=self.slow_opponent_prompt_template,
                log_file=self._slow_opp_log_file,
                game_type=self.game_type,
                log_agent=self._log_agent,
                side=self._side,
                log_module="slow_opp",
            )
            if self.enable_opp_slow
            else None
        )

        # 慢线程控制（保证 get_action 不阻塞）
        self._lock = threading.Lock()
        self._slow_done: bool = True

        # Public usage/token interface for wrapper parity.
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self.last_request_exception = None
        self.last_request_status_code = None

        # 缓存：两棵树（slow 写 / fast 读）
        # Keep a root node (0) in both trees so downstream code always has a fallback.
        self._cached_tree = "- 0:"
        self._cached_opponent_tree = "- 0:"

        # Maze-only (race): remember the shared goal-zone center for root grounding.
        self._maze_goal_a_xy = None  # (x, y)

        # Maze-only: allow the controller to disable direct-execution for one decision
        # after an execution failure (dynamic blocking can invalidate a planned path).
        self._maze_disable_direct_execute_once: bool = False
        self.last_action_source = "fast"

    def _sync_usage_from_modules(self) -> None:
        """聚合 fast/slow 模块 usage 与 token 统计，对外暴露统一口径。

        说明：
        - `last_usage` / `last_request_*` 以 fast 模块为主，因为真正出动作的是 fast。
        - `total_tokens` 统计 fast + slow_self + slow_opp，便于做总成本评估。
        - 任何子模块取值异常都静默跳过，不影响主流程。
        """
        self.last_usage = normalize_usage(getattr(self.fast_module, "last_usage", {}) or {})
        self.last_request_exception = getattr(self.fast_module, "last_request_exception", None)
        self.last_request_status_code = getattr(self.fast_module, "last_request_status_code", None)

        total = 0
        for module in (self.fast_module, self.slow_module, self.slow_module_opponent):
            if module is None:
                continue
            try:
                total += int(getattr(module, "total_tokens", 0) or 0)
            except Exception:
                continue
        self.total_tokens = total

    def token_usage_snapshot(self) -> dict[str, int]:
        """返回稳定的 token 快照，供 wrapper/实验统计读取。

        返回字段：
        - fast: 快模块累计 token
        - slow_self: 我方慢模块累计 token
        - slow_opp: 敌方慢模块累计 token
        - total: 三者之和（下限截断为 0）
        """
        def _tok(module) -> int:
            try:
                return int(getattr(module, "total_tokens", 0) or 0)
            except Exception:
                return 0

        fast = _tok(self.fast_module)
        slow_self = _tok(self.slow_module)
        slow_opp = _tok(self.slow_module_opponent)
        total = int(max(0, fast + slow_self + slow_opp))
        return {
            "fast": fast,
            "slow_self": slow_self,
            "slow_opp": slow_opp,
            "total": total,
        }

    def _root_desc(self, player: str, *, for_opponent_tree: bool) -> str:
        """Return the desired root-node description for the cached tree.

        - self tree: 根描述我方希望达成的终局（通常是对方失败条件）
        - opponent tree: 根描述敌方希望达成的终局（我们需要阻止）

        注意：
        - 这是语义树的“锚点”，用于保证树结构始终可回溯到一个稳定根。
        - maze/chess 的根语义不同，分别按各自游戏机制生成。
        """
        p = (player or "").strip().lower()
        if self.game_type == "maze":
        # maze
            ga_x, ga_y = self._maze_goal_a_xy
            if p == "blue":
                actor = "Red" if for_opponent_tree else "Blue"
            else:
                actor = "Blue" if for_opponent_tree else "Red"
            return f"{actor} reaches the shared goal zone at ({ga_x}, {ga_y})"

        # chess
        if p == "w":
            self_color, opponent_color = "White", "Black"
        else:
            self_color, opponent_color = "Black", "White"
        return f"{opponent_color} King is captured by {self_color} piece at (x, n)"

    def _sync_root_nodes(self, player: str) -> None:
        """Keep root node 0 populated for both cached trees."""
        self_root = self._root_desc(player, for_opponent_tree=False)
        opp_root = self._root_desc(player, for_opponent_tree=True)
        with self._lock:
            self._cached_tree = apply_tree_patches(self._cached_tree, fix_nodes={"0": self_root})
            self._cached_opponent_tree = apply_tree_patches(self._cached_opponent_tree, fix_nodes={"0": opp_root})

    def _state_hits_any_tree_node(
        self,
        *,
        state: Any,
        state_str: str,
        player: str,
        legal_moves: list,
        self_tree_text: str,
        opp_tree_text: str,
    ) -> bool:
        """判断当前状态是否“命中”任一语义树节点。

        目的：
        - 如果当前状态已经落在语义节点上，继续 slow 扩展常常是冗余的。
        - 提前跳过 slow，有助于减少震荡与无效 token 消耗。

        判定方式：
        - Maze: 检查我方/敌方当前位置是否出现在对应树节点描述坐标中。
        - Chess: 用 nearest score 近似“精确命中”（阈值 0.9999）。
        """
        try:
            if self.game_type == "maze":
                p = (player or "").strip().lower()
                opponent_player = "blue" if p == "red" else "red"

                _nid_s, _desc_s, score_s = find_nearest_node(
                    state_str,
                    self_tree_text,
                    exclude_root=False,
                    source_role=player,
                    opponent_role=opponent_player,
                )
                _nid_o, _desc_o, score_o = find_nearest_node(
                    state_str,
                    opp_tree_text,
                    exclude_root=False,
                    source_role=opponent_player,
                    opponent_role=player,
                )
                self_hit = float(score_s) >= 0.9999
                opp_hit = float(score_o) >= 0.9999
                return bool(self_hit or opp_hit)

            # chess: use a conservative threshold for "exact".
            _nid_s, _desc_s, score_s = find_nearest_node_chess(
                state_str,
                self_tree_text,
                exclude_root=False,
            )
            _nid_o, _desc_o, score_o = find_nearest_node_chess(
                state_str,
                opp_tree_text,
                exclude_root=False,
            )
            return (float(score_s) >= 0.9999) or (float(score_o) >= 0.9999)
        except Exception:
            return False

    # ============================================================
    # 模块 2/3：后台慢模块（Slow self / Slow opponent）
    # ============================================================
    def _trigger_slow_if_done(self, state: Any, state_str: str, player: str, legal_moves: list) -> None:
        """慢模块入口：仅在空闲时异步触发一次 slow self/opp 更新。

        关键约束：
        - 非阻塞：此函数只负责“发起后台线程”，不等待 slow 结果。
        - 单飞：通过 `_slow_done` 保证同时最多只有一轮 slow 在跑。
        - 命中即跳过：若当前状态已命中语义节点，则不启动本轮 slow。
        """

        if (not self.enable_self_slow) and (not self.enable_opp_slow):
            return

        with self._lock:
            if not self._slow_done:
                return

            # If current state already matches a node in either tree, skip slow thinking.
            tree_snapshot = self._cached_tree if self.enable_self_slow else "- 0:"
            opp_tree_snapshot = self._cached_opponent_tree if self.enable_opp_slow else "- 0:"

        if self._state_hits_any_tree_node(
            state=state,
            state_str=state_str,
            player=player,
            legal_moves=legal_moves,
            self_tree_text=tree_snapshot,
            opp_tree_text=opp_tree_snapshot,
        ):
            return

        with self._lock:
            if not self._slow_done:
                return
            self._slow_done = False

        def _run_slow(state_inner: Any, state_str_inner: str, player_inner: str, moves_inner: list) -> None:
            """后台慢线程主体：选扩展点 -> 调 slow -> 合并树 -> 写快照。"""
            try:
                # Snapshot cached trees early for consistent expand targets.
                with self._lock:
                    current_tree_snapshot = self._cached_tree if self.enable_self_slow else "- 0:"
                    current_opp_tree_snapshot = self._cached_opponent_tree if self.enable_opp_slow else "- 0:"

                # 取“最相似节点”作为 expand 目标（直接调用 semantic_tree_manager，避免额外封装）
                def _nearest(tree_text: str, *, source_role: str) -> tuple[str, str]:
                    """从给定树里选“本轮最该扩展”的节点（返回 id, desc）。"""
                    if self.game_type == "chess":
                        nid, desc, _ = find_nearest_node_chess(
                            state_str_inner,
                            tree_text,
                            exclude_root=False,
                        )
                    else:
                        # Only consider leaf nodes for slow module expansion
                        p = (source_role or "").strip().lower()
                        opp_role = "blue" if p == "red" else "red"
                        leaf_ids = set(_get_leaf_node_ids(tree_text))
                        # Score only leaf nodes
                        node_scores = score_tree_nodes_maze(
                            state_str_inner,
                            tree_text,
                            exclude_root=False,
                            top_k=None,
                            source_role=source_role,
                            opponent_role=opp_role,
                        )
                        leaf_scores = [n for n in node_scores if n["id"] in leaf_ids]
                        if not leaf_scores:
                            # fallback: use all nodes if no leaves
                            leaf_scores = node_scores
                        # Pick the highest score (desc, id)
                        if not leaf_scores:
                            nid, desc = "0", ""
                        else:
                            best = max(leaf_scores, key=lambda n: (n["score"], -n["depth"]))
                            nid, desc = best["id"], ""  # desc will be filled below
                            # Get desc from tree
                            desc = _parse_tree_nodes(tree_text).get(nid, "")
                    if nid is None:
                        raise RuntimeError("nearest-node returned nid=None (tree contract broken)")
                    if desc is None:
                        raise RuntimeError(f"nearest-node returned desc=None for nid={nid}")
                    return str(nid).strip() or "0", str(desc).strip()

                p = (player_inner or "").strip().lower()
                if self.game_type == "maze":
                    opponent_player = "blue" if p == "red" else "red"
                else:
                    opponent_player = "b" if p == "w" else "w"

                # Compute expand targets from the snapshots.
                target_id, target_desc = "0", ""
                target_id_opp, target_desc_opp = "0", ""
                if self.enable_self_slow:
                    target_id, target_desc = _nearest(current_tree_snapshot, source_role=player_inner)
                if self.enable_opp_slow:
                    target_id_opp, target_desc_opp = _nearest(current_opp_tree_snapshot, source_role=opponent_player)

                # -------- 模块 2/3：并行调用慢模块（self / opponent） --------
                def _call_self() -> dict:
                    """调用我方 slow 模块；若关闭则返回空增量。"""
                    if (not self.enable_self_slow) or (self.slow_module is None):
                        return {"tree": "", "fix_nodes": {}}
                    return self.slow_module.get_guidance(
                        state=state_inner,
                        player=player_inner,
                        guidance_line_tree_last=current_tree_snapshot,
                        expand_node_id=target_id,
                        expand_node_desc=target_desc,
                    )

                def _call_opp() -> dict:
                    """调用敌方 slow 模块；若关闭则返回空增量。"""
                    if (not self.enable_opp_slow) or (self.slow_module_opponent is None):
                        return {"tree": "", "fix_nodes": {}}
                    return self.slow_module_opponent.get_guidance(
                        state=state_inner,
                        player=opponent_player,
                        guidance_line_tree_last=current_opp_tree_snapshot,
                        expand_node_id=target_id_opp,
                        expand_node_desc=target_desc_opp,
                    )

                if self.enable_self_slow and self.enable_opp_slow:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        fut_self = ex.submit(_call_self)
                        fut_opp = ex.submit(_call_opp)
                        result_self = fut_self.result()
                        result_opp = fut_opp.result()
                else:
                    result_self = _call_self()
                    result_opp = _call_opp()

                new_tree_self = result_self["tree"]
                fix_nodes_self = result_self.get("fix_nodes") or {}
                new_tree_opp = result_opp["tree"]
                fix_nodes_opp = result_opp.get("fix_nodes") or {}

                # -------- 模块 4：合并进缓存树 + 写 JSON --------
                with self._lock:
                    if self.enable_self_slow:
                        patched_self = apply_tree_patches(self._cached_tree, fix_nodes=fix_nodes_self)
                        self._cached_tree = merge_tree(patched_self, new_tree_self)
                    if self.enable_opp_slow:
                        patched_opp = apply_tree_patches(self._cached_opponent_tree, fix_nodes=fix_nodes_opp)
                        self._cached_opponent_tree = merge_tree(patched_opp, new_tree_opp)

                    tree_for_log = self._cached_tree
                    opp_tree_for_log = self._cached_opponent_tree

                # Compute per-node similarity scores for logging (debug/explainability).
                # This is intentionally done here (slow thread) so get_action() stays fast.
                node_scores_self = []
                node_scores_opp = []
                try:
                    if self.game_type == "chess":
                        node_scores_self = score_tree_nodes_chess(
                            state_str_inner,
                            tree_for_log,
                            exclude_root=False,
                            top_k=None,
                        )
                        node_scores_opp = score_tree_nodes_chess(
                            state_str_inner,
                            opp_tree_for_log,
                            exclude_root=False,
                            top_k=None,
                        )
                    else:
                        node_scores_self = score_tree_nodes_maze(
                            state_str_inner,
                            tree_for_log,
                            exclude_root=False,
                            top_k=None,
                            source_role=player_inner,
                            opponent_role=opponent_player,
                        )
                        node_scores_opp = score_tree_nodes_maze(
                            state_str_inner,
                            opp_tree_for_log,
                            exclude_root=False,
                            top_k=None,
                            source_role=opponent_player,
                            opponent_role=player_inner,
                        )
                except Exception:
                    # Logging should never break the agent.
                    node_scores_self = []
                    node_scores_opp = []

                self._append_trees_json_snapshot(
                    {
                        "player": player_inner,
                        "slow_self_new_tree_len": len(new_tree_self),
                        "slow_opp_new_tree_len": len(new_tree_opp),
                    },
                    state_str=state_str_inner,
                    node_scores={
                        "self": node_scores_self,
                        "opponent": node_scores_opp,
                    },
                )
            finally:
                with self._lock:
                    self._slow_done = True

        thr = threading.Thread(
            target=_run_slow,
            args=(state, state_str, player, legal_moves),
            daemon=True,
        )
        thr.start()

    # ============================================================
    # 模块 4：存树 + 相似度匹配与选点
    # ============================================================
    def _append_trees_json_snapshot(
        self,
        meta: Optional[Dict[str, Any]] = None,
        state_str: Optional[str] = None,
        node_scores: Optional[Dict[str, Any]] = None,
    ) -> None:
        """把当前缓存的我树/敌树写入 NDJSON 快照。

        说明：
        - 采用“逐行 JSON”而不是大数组，便于流式追加和后处理。
        - 不写入完整 state_str，控制日志体积。
        - 可选携带 node_scores 便于可解释性分析。
        """
        path = getattr(self, "_slow_trees_json_file", "")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "game_type": self.game_type,
            **(meta or {}),
            # Keep the NDJSON small: do not store full `state_str` (maze contains map/visited/trace).
            "state": {},
            "self": {
                "tree": self._cached_tree,
            },
            "opponent": {
                "tree": self._cached_opponent_tree,
            },
        }

        if node_scores:
            try:
                if "self" in node_scores:
                    record["self"]["node_scores"] = node_scores.get("self")
                if "opponent" in node_scores:
                    record["opponent"]["node_scores"] = node_scores.get("opponent")
            except Exception:
                pass

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _prepare_subgoal(
        self,
        state_str: str,
        player: str,
        legal_moves: list,
    ) -> Tuple[str, str, str]:
        """调度选点：基于双树相似度与策略规则产出 subgoal/path/decision_mode。

        输出：
        - subgoal_text: 给 fast 模块的当前步语义目标
        - subgoal_path: 该目标在树中的叶到根路径（用于解释与直执行）
        - decision_mode: ATTACK / DEFEND

        核心思想：
        - 先找我树最近点 S（默认进攻）。
        - 再评估敌树最近点 O（必要时防守）。
        - 最终受 policy（auto/attack_only/defend_only）门控。
        """

        p = (player or "").strip().lower()

        # 1) 先在“我方树”上找当前最相似节点，作为默认进攻目标。
        # chess 还会用该节点深度 k 约束敌树搜索层级（k / k-1）。
        if self.game_type == "chess":
            self_id, self_desc, self_score = find_nearest_node_chess(
                state_str,
                self._cached_tree,
                exclude_root=False,
            )
        else:
            self_id, self_desc, self_score = find_nearest_node(
                state_str,
                self._cached_tree,
                exclude_root=False,
                source_role=player,
                opponent_role=("blue" if p == "red" else "red"),
            )

        self_id = (self_id or "0").strip() or "0"
        k = self_id.count(".") if self.game_type == "chess" else 0
        self_enabled = bool(self.enable_self_slow)
        opp_enabled = bool(self.enable_opp_slow)
        policy = self.decision_mode_policy

        # 2) 再评估“敌方树”是否更值得优先处理（防守切换候选）。
        opp_id, opp_desc, opp_score = None, None, 0.0
        use_opponent = False

        # ===== Opponent evaluation (CHESS) =====
        # 规则：
        # 1) 先按我方节点深度 k，在敌树的 k / (k-1) 层找最相似节点；
        # 2) 若没找到，再回退到整棵敌树搜索；
        # 3) 仅当敌树分数 > 我树分数时，切 DEFEND。
        if opp_enabled and self.game_type == "chess":
            opp_candidates: list[tuple[str, str, float, int]] = []
            for depth in (k, k - 1):
                if depth < 1:
                    continue
                oid, odesc, oscore = find_nearest_node_chess_at_depth(
                    state_str,
                    self._cached_opponent_tree,
                    depth,
                    exclude_root=False,
                )
                if oid and (odesc is not None):
                    opp_candidates.append((str(oid), str(odesc), float(oscore), int(depth)))

            if opp_candidates:
                opp_candidates.sort(key=lambda t: (t[2], -t[3]), reverse=True)
                opp_id, opp_desc, opp_score, _ = opp_candidates[0]
            else:
                oid, odesc, oscore = find_nearest_node_chess(
                    state_str,
                    self._cached_opponent_tree,
                    exclude_root=False,
                )
                if oid and (odesc is not None):
                    opp_id, opp_desc, opp_score = str(oid), str(odesc), float(oscore)

            # chess 切换条件：敌方候选存在，且敌树得分严格高于我树。
            use_opponent = bool(opp_id) and (float(opp_score) > float(self_score))

        # ===== Opponent evaluation (MAZE) =====
        # 规则：
        # 1) 先拿敌树最近点；
        # 2) 解析坐标后比较两段距离：
        #    - dist_self: 我到我方目标点距离
        #    - dist_our_to_opp: 我到敌方目标点距离
        # 3) 仅当敌方更近且足够近(<=2)时，切 DEFEND。
        elif opp_enabled:
            try:
                opponent_player = "blue" if p == "red" else "red"

                oid, odesc, oscore = find_nearest_node(
                    state_str,
                    self._cached_opponent_tree,
                    exclude_root=False,
                    source_role=player,
                    opponent_role=opponent_player,
                )
                if oid and (odesc is not None):
                    opp_id, opp_desc, opp_score = str(oid), str(odesc), float(oscore)

                try:
                    obj = json.loads(state_str)
                except Exception:
                    obj = None

                our_xy = None
                if isinstance(obj, dict):
                    our_obj = (obj.get(player) or {}) if isinstance(obj.get(player), dict) else {}
                    try:
                        our_xy = (int(our_obj.get("x")), int(our_obj.get("y")))
                    except Exception:
                        pass

                s_xy = None
                ms = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", self_desc or "")
                if ms:
                    s_xy = (int(ms.group(1)), int(ms.group(2)))

                o_xy = None
                mo = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", opp_desc or "")
                if mo:
                    o_xy = (int(mo.group(1)), int(mo.group(2)))

                dist_self = float("inf")
                if self_enabled and our_xy is not None and s_xy is not None:
                    dist_self = float(abs(int(our_xy[0]) - int(s_xy[0])) + abs(int(our_xy[1]) - int(s_xy[1])))

                dist_our_to_opp = float("inf")
                if our_xy is not None and o_xy is not None:
                    dist_our_to_opp = float(abs(int(our_xy[0]) - int(o_xy[0])) + abs(int(our_xy[1]) - int(o_xy[1])))

                # maze 切换条件：敌方候选存在 + 更近 + 近到可立即干预。
                use_opponent = bool(opp_id) and (dist_our_to_opp < dist_self) and (dist_our_to_opp <= 2.0)
            except Exception:
                use_opponent = False

        # 3) 最后由策略门控覆盖自动判断结果。
        # - attack_only: 强制 ATTACK
        # - defend_only: 只要有敌方候选就 DEFEND
        # - auto: 维持上面算出的 use_opponent
        if policy == "attack_only":
            use_opponent = False
        elif policy == "defend_only":
            use_opponent = bool(opp_id)
        elif (not opp_enabled):
            use_opponent = False
        elif (not self_enabled):
            use_opponent = bool(opp_id)

        # 4) 根据最终模式选定目标节点，并组装 subgoal 文本/路径。
        picked_id = opp_id if use_opponent else self_id
        picked_desc = opp_desc if use_opponent else self_desc
        picked_tree_text = self._cached_opponent_tree if use_opponent else self._cached_tree
        if picked_desc is None:
            raise RuntimeError(f"picked_desc is None (picked_id={picked_id}, use_opponent={use_opponent})")

        subgoal_text = picked_desc.strip()
        decision_mode = "DEFEND" if use_opponent else "ATTACK"
        if use_opponent:
            hint = (
                "Stop the opponent's objective and disrupt their best route. "
                "Prefer moves that step onto/approach junctions on the opponent's route to block them."
                if self.game_type == "maze"
                else "Stop the opponent's objective and prevent threats to us along this path. "
                "Use our King or another piece to eliminate the threatening piece, or relocate the King to safety."
            )
            subgoal_text = f"{subgoal_text}\n{hint}" if subgoal_text else hint

        subgoal_path = format_path_leaf_to_root(picked_tree_text, picked_id) if picked_id else ""

        # 5) 语义占位符落地：把 (x, n) 等抽象描述替换成当前局面的具体坐标。
        if self.game_type == "chess":
            try:
                subgoal_text = ground_chess_semantic_text(state_str, subgoal_text)
                if subgoal_path:
                    subgoal_path = ground_chess_semantic_text(state_str, subgoal_path)
            except Exception:
                pass
        elif self.game_type == "maze":
            try:
                subgoal_text = ground_maze_semantic_text(state_str, subgoal_text)
                if subgoal_path:
                    subgoal_path = ground_maze_semantic_text(state_str, subgoal_path)
            except Exception:
                pass

        return subgoal_text, subgoal_path, decision_mode

    # ============================================================
    # Public API (placed last for alignment)
    # ============================================================
    def get_action(self, state: Any, player: str, legal_moves: list):
        """单步主入口：选点→可选 direct-execute→异步 slow→fast 出动作。

        这是唯一对外决策接口，一步内完成：
        1) 预处理状态并维护树根
        2) 基于双树选 subgoal
        3) 尝试规则直执行（命中则直接返回）
        4) 后台触发 slow 更新（不阻塞）
        5) 前台调用 fast 生成动作并返回
        """

        # state_str：用于相似度（maze 用 json；chess 用 FEN/str）
        if self.game_type == "maze" and isinstance(state, dict):
            try:
                goal = state.get("goal")
                if isinstance(goal, dict) and ("x" in goal) and ("y" in goal):
                    self._maze_goal_a_xy = (int(goal["x"]), int(goal["y"]))
            except Exception:
                pass
            state_str = json.dumps(state, ensure_ascii=False)
        else:
            state_str = str(state)
        self._sync_root_nodes(player)

        # 1) 模块 4：k/(k-1) 选点（基于当前缓存树）
        subgoal_text, subgoal_path, decision_mode = self._prepare_subgoal(
            state_str=state_str,
            player=player,
            legal_moves=legal_moves,
        )

        # Maze: only simplify the displayed subgoal (NOT the route/path).
        if self.game_type == "maze":
            # Keep prompt/logs concise: '(x1,y1) to (x2,y2)' -> '(x1,y1)'
            try:
                subgoal_text = re.sub(
                    r"(\(\s*\d+\s*,\s*\d+\s*\))\s*to\s*\(\s*\d+\s*,\s*\d+\s*\)",
                    r"\1",
                    subgoal_text or "",
                    flags=re.IGNORECASE,
                )
            except Exception:
                pass

        # Optional bypass: if the chosen subgoal implies an immediate king capture
        # and the corresponding UCI is legal (e.g., c2e1/g2e1), execute directly.
        if self.game_type == "chess":
            forced = pick_forced_uci_from_subgoal(state_str, subgoal_text, legal_moves, player)
            if forced:
                self.last_action_source = "direct_execute"
                self._sync_usage_from_modules()
                return forced

        # Maze direct execution: if the tree path exactly matches current state,
        # execute the next junction step without LLM.
        if self.game_type == "maze":
            if self._maze_disable_direct_execute_once:
                # One-shot bypass; reset immediately.
                self._maze_disable_direct_execute_once = False
            else:
                forced_xy = pick_forced_maze_target_from_subgoal_path(state, player, subgoal_path, legal_moves)
                if forced_xy is not None:
                    self.last_action_source = "direct_execute"
                    self._sync_usage_from_modules()
                    return forced_xy

        # 2) 模块 2/3：仅在未 direct-execute 时，非阻塞触发慢模块更新
        self._trigger_slow_if_done(state, state_str, player, legal_moves)

        # 3) 模块 1：调用快模块出动作
        if self.game_type == "chess":
            action = self.fast_module.get_action(
                state_str,
                player,
                legal_moves,
                decision_mode=decision_mode,
                subgoal_text=subgoal_text,
                subgoal_path=subgoal_path,
            )
            self.last_action_source = getattr(self.fast_module, "last_action_source", "fast")
            self._sync_usage_from_modules()
            return action

        action = self.fast_module.get_action(
            state,
            player,
            legal_moves,
            decision_mode=decision_mode,
            subgoal_text=subgoal_text,
            subgoal_path=subgoal_path,
        )
        self.last_action_source = getattr(self.fast_module, "last_action_source", "fast")
        self._sync_usage_from_modules()
        return action
