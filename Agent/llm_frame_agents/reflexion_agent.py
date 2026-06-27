"""Reflexion 智能体（决策与反思并行）。

机制说明：
1) 前台每步都执行决策模块。
2) 后台线程异步执行反思模块，基于“上一版反思 + 最近轨迹”更新反思摘要。
3) 下一步决策提示词会注入最新反思，从而持续自我修正。

决策流程：
- get_action 先构造带 reflection 的提示词并得到动作
- 同步记录 usage 与本步轨迹
- 唤醒后台反思线程生成下一版 reflection
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from Agent.llm_base_agents.llm_agent import LLMAgent as _BaseLLM
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total


BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROMPTS = {
    "maze": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Maze_reflexion_llm_agent_prompt.txt"),
    "chess": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Chess_reflexion_llm_agent_prompt.txt"),
}

_INCREMENTAL_REFLECTION_PROMPT = (
    "You are updating Reflexion continuously during a game.\n"
    "Keep it concise and actionable (<= 6 lines).\n"
    "Use previous reflection and recent trajectory records.\n\n"
    "\n"
    "Previous Reflection:\n"
    "{previous_reflection}\n\n"
    "Recent 5-step Trajectory (LLM input/output/parsed):\n"
    "{recent_trajectory}\n\n"
    "Output format:\n"
    "Summary:\n"
    "- <rule>\n"
    "- <rule>\n"
    "- <rule>\n"
)


def _derive_decision_log_path(log_file: str, *, game: str, side: str | None = None) -> str:
    return make_log_file(log_file, game=game, side=side, agent="reflexion", module="decision")


def _derive_reflection_log_path(log_file: str, *, game: str, side: str | None = None) -> str:
    return make_log_file(log_file, game=game, side=side, agent="reflexion", module="reflection")


class _ReflexionCore:
    def __init__(
        self,
        decision_inner: _BaseLLM,
        reflection_inner: _BaseLLM,
        act_template: str,
        role_tag: str,
    ):
        self.inner = decision_inner
        self._reflection_inner = reflection_inner
        self._act_template = act_template
        self._role_tag = role_tag

        self._reflection = "(none)"
        self._turn = 0
        self._episode_records: list[dict] = []

        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
        self.last_request_exception = None
        self.last_request_status_code = None

        self._lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._cv = threading.Condition()
        self._stop_event = threading.Event()
        self._pending = False
        self._worker = threading.Thread(target=self._reflection_loop, name=f"ReflexionLoop[{role_tag}]", daemon=True)
        self._worker.start()

    @contextmanager
    def _using_prompt_template(self, inner: _BaseLLM, prompt_template: str):
        old = inner.prompt_template
        inner.prompt_template = prompt_template
        try:
            yield
        finally:
            inner.prompt_template = old

    def _sync_usage_from(self, inner: _BaseLLM, *, update_last: bool) -> None:
        """同步一次 inner 的 usage，并在线程安全条件下更新统计状态。"""
        with self._usage_lock:
            usage = normalize_usage(getattr(inner, "last_usage", {}) or {})
            inner_total = getattr(inner, "total_tokens", None)
            if update_last:
                self.last_usage = usage
                self.last_request_exception = getattr(inner, "last_request_exception", None)
                self.last_request_status_code = getattr(inner, "last_request_status_code", None)
            self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
                self.total_tokens,
                self._last_inner_total,
                inner_total,
                usage,
            )

    def _reflection_snapshot(self) -> str:
        with self._lock:
            return self._reflection

    def _record_step(self, player: str, llm_input: str, llm_output: str, parsed_action) -> None:
        with self._lock:
            self._turn += 1
            rec = {
                "turn": self._turn,
                "player": str(player),
                "llm_input": str(llm_input or ""),
                "llm_output": str(llm_output or ""),
                "parsed_action": "parse_failed" if parsed_action is None else str(parsed_action),
            }
            self._episode_records.append(rec)

    def _recent_trajectory_text(self) -> str:
        with self._lock:
            recent = self._episode_records[-5:]
        if not recent:
            return "(no_records)"
        parts: list[str] = []
        for r in recent:
            llm_input = str(r.get("llm_input") or "")
            llm_output = str(r.get("llm_output") or "")
            parts.append(
                "\n".join(
                    [
                        f"turn={r.get('turn')} player={r.get('player')} parsed_action={r.get('parsed_action')}",
                        "llm_input:",
                        llm_input or "(empty)",
                        "llm_output:",
                        llm_output or "(empty)",
                    ]
                )
            )
        return "\n\n---\n\n".join(parts)

    def _queue_reflection(self) -> None:
        with self._cv:
            self._pending = True
            self._cv.notify()

    def _reflection_loop(self):
        """后台反思循环：消费 pending 信号并增量更新 reflection。"""
        while not self._stop_event.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._stop_event.is_set() or self._pending)
                if self._stop_event.is_set():
                    return
                self._pending = False

            prev = self._reflection_snapshot()
            recent = self._recent_trajectory_text()
            prompt = _INCREMENTAL_REFLECTION_PROMPT.format(
                role=self._role_tag,
                previous_reflection=prev,
                recent_trajectory=recent,
            )
            self._reflection_inner.logger.info("[Reflexion][%s] Incremental reflection prompt:\n%s", self._role_tag, prompt)
            text = self._reflection_inner._call_llm_api(prompt)
            self._sync_usage_from(self._reflection_inner, update_last=False)

            updated = str(text or "").strip()
            if updated:
                with self._lock:
                    self._reflection = updated

    def stop(self):
        self._stop_event.set()
        with self._cv:
            self._cv.notify_all()


class ReflexionDualMaze(_ReflexionCore):
    def __init__(
        self,
        api_setting,
        llm_settings,
        role,
        log_file,
        model_name: str = "",
        decision_model_name: str = "",
        reflection_model_name: str = "",
    ):
        act_template = open(PROMPTS["maze"], "r", encoding="utf-8").read()
        decision_log = _derive_decision_log_path(log_file, game="maze", side=role)
        reflection_log = _derive_reflection_log_path(log_file, game="maze", side=role)
        resolved_shared = str(model_name or "").strip()
        resolved_decision = str(decision_model_name or "").strip()
        resolved_reflection = str(reflection_model_name or "").strip()
        if not resolved_decision:
            resolved_decision = resolved_shared
        if not resolved_reflection:
            resolved_reflection = resolved_shared
        if not resolved_decision or not resolved_reflection:
            raise ValueError("ReflexionDualMaze requires explicit model_name or both decision_model_name/reflection_model_name")

        decision_inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_decision,
            LLM_module_prompt=None,
            log_file=decision_log,
            game="maze",
            side=role,
            log_agent="reflexion",
            log_module="decision",
        )
        reflection_inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_reflection,
            LLM_module_prompt=None,
            log_file=reflection_log,
            game="maze",
            side=role,
            log_agent="reflexion",
            log_module="reflection",
        )

        super().__init__(
            decision_inner=decision_inner,
            reflection_inner=reflection_inner,
            act_template=act_template,
            role_tag=f"maze:{role}",
        )

        self.role = role
        self.slow_module = None
        self.slow_module_opponent = None
        self.fast_module = self

    def get_action(self, frame, player=None, legal_moves=None):
        """Maze 前台决策入口：注入 reflection，执行决策并触发后台反思。"""
        role = player if isinstance(player, str) and player.strip() else self.role
        reflection = self._reflection_snapshot()
        prompt_template = self._act_template.format(
            player="{player}",
            moves="{moves}",
            maze_description="{maze_description}",
            reflection=reflection,
        )

        with self._using_prompt_template(self.inner, prompt_template):
            llm_input = self.inner._build_prompt_maze(frame=frame, legal_moves=legal_moves, role=role)
            role_full = "Red" if role == "red" else "Blue" if role == "blue" else role
            self.inner.logger.info(
                "[LLM_AGENT][%s][model=%s] Prompt sent to LLM:\n%s",
                role_full,
                getattr(self.inner, "model_name", ""),
                llm_input,
            )
            llm_output = self.inner._call_llm_api(llm_input)
            last_sentence = self.inner._maze_parser.get_last_sentence(llm_output or "")
            parsed = self.inner._maze_parser.find_target_junction_in_sentence(last_sentence)
            action = parsed if self.inner._maze_parser.is_valid_target_junction(parsed) else None

        if not llm_output:
            status = getattr(self.inner, "last_request_status_code", None)
            exc = getattr(self.inner, "last_request_exception", None)
            if exc is not None or status is not None:
                self.inner.logger.error("[LLMAgent] No LLM response (status=%s, exc=%r)", status, exc)
            else:
                self.inner.logger.error("[LLMAgent] No LLM response")
        elif action is not None:
            self.inner.logger.info("[LLMAgent] Valid move: %s", action)
        else:
            self.inner.logger.warning("[LLMAgent] Parse failed from response tail: %s", last_sentence)

        self._sync_usage_from(self.inner, update_last=True)
        self._record_step(role, llm_input, llm_output, action)
        self._queue_reflection()
        return action


class ReflexionDualChess(_ReflexionCore):
    def __init__(
        self,
        api_setting,
        llm_settings,
        log_file,
        side: str | None = None,
        model_name: str = "",
        decision_model_name: str = "",
        reflection_model_name: str = "",
    ):
        act_template = open(PROMPTS["chess"], "r", encoding="utf-8").read()
        decision_log = _derive_decision_log_path(log_file, game="chess", side=side)
        reflection_log = _derive_reflection_log_path(log_file, game="chess", side=side)
        resolved_shared = str(model_name or "").strip()
        resolved_decision = str(decision_model_name or "").strip()
        resolved_reflection = str(reflection_model_name or "").strip()
        if not resolved_decision:
            resolved_decision = resolved_shared
        if not resolved_reflection:
            resolved_reflection = resolved_shared
        if not resolved_decision or not resolved_reflection:
            raise ValueError("ReflexionDualChess requires explicit model_name or both decision_model_name/reflection_model_name")

        decision_inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_decision,
            LLM_module_prompt=None,
            log_file=decision_log,
            game="chess",
            side=side,
            log_agent="reflexion",
            log_module="decision",
        )
        reflection_inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_reflection,
            LLM_module_prompt=None,
            log_file=reflection_log,
            game="chess",
            side=side,
            log_agent="reflexion",
            log_module="reflection",
        )

        super().__init__(
            decision_inner=decision_inner,
            reflection_inner=reflection_inner,
            act_template=act_template,
            role_tag="chess",
        )

        self.slow_module = None
        self.slow_module_opponent = None
        self.fast_module = self

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """Chess 前台决策入口：注入 reflection，执行决策并触发后台反思。"""
        reflection = self._reflection_snapshot()
        prompt_template = self._act_template.format(
            my_color="{my_color}",
            opponent_color="{opponent_color}",
            fen_description="{fen_description}",
            moves="{moves}",
            reflection=reflection,
        )

        with self._using_prompt_template(self.inner, prompt_template):
            llm_input = self.inner._build_prompt_chess(
                enhanced_FEN_full=enhanced_FEN_full,
                legal_moves=legal_moves,
                color=color,
            )
            color_full = "White" if color == "w" else "Black" if color == "b" else color
            self.inner.logger.info(
                "[LLM_AGENT][%s][model=%s] Prompt sent to LLM:\n%s",
                color_full,
                getattr(self.inner, "model_name", ""),
                llm_input,
            )
            llm_output = self.inner._call_llm_api(llm_input)
            last_sentence = self.inner._chess_parser.get_last_sentence(llm_output or "")
            parsed = self.inner._chess_parser.find_uci_in_sentence(last_sentence)
            action = parsed if self.inner._chess_parser.is_valid_uci(parsed) else None

        if not llm_output:
            status = getattr(self.inner, "last_request_status_code", None)
            exc = getattr(self.inner, "last_request_exception", None)
            if exc is not None or status is not None:
                self.inner.logger.error("[LLMAgent] No LLM response (status=%s, exc=%r)", status, exc)
            else:
                self.inner.logger.error("[LLMAgent] No LLM response")
        elif action is not None:
            self.inner.logger.info("[LLMAgent] Valid move: %s", action)
        else:
            self.inner.logger.warning("[LLMAgent] Parse failed from response tail: %s", last_sentence)

        self._sync_usage_from(self.inner, update_last=True)
        self._record_step(color, llm_input, llm_output, action)
        self._queue_reflection()
        return action
