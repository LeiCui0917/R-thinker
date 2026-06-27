"""Coding 智能体（双模块并行，薄封装）。

核心机制：
1) 生成模块（后台）
    - 使用 coding 提示词调用 LLM。
    - 解析输出为 state-action rules。
    - 持续写入执行缓存（新规则优先）。
2) 执行模块（前台）
    - 先按缓存规则做 direct execute。
    - 若未命中，立即回退到标准 LLM 决策。

决策流程（每步）：
- 前台先把最新状态投递给后台线程
- 前台立即尝试缓存规则匹配
- 未命中则直接 fallback 到标准 LLM
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import deque
from pathlib import Path

from Agent.R_Thinker import direct_execute_module as _direct_execute
from Agent.R_Thinker.utils.slow_output_parser import parse_chess_slow_output
from Agent.llm_base_agents.llm_agent import LLMAgent as _BaseLLM
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import add_total_tokens, normalize_usage

logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROMPTS = {
    "maze_coding": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Maze_coding.txt"),
    "chess_coding": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Chess_coding.txt"),
    "maze_llm": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Maze_llm_agent_prompt.txt"),
    "chess_llm": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Chess_llm_agent_prompt.txt"),
}


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


CODING_PROMPT_MAZE = _read_text(PROMPTS["maze_coding"])
CODING_PROMPT_CHESS = _read_text(PROMPTS["chess_coding"])
LLM_PROMPT_MAZE = _read_text(PROMPTS["maze_llm"])
LLM_PROMPT_CHESS = _read_text(PROMPTS["chess_llm"])


def _derive_log_path(log_file: str, *, game: str, side: str | None = None, module: str = "decision", k: int | None = None) -> str:
    return make_log_file(log_file, game=game, side=side, agent="coding", module=module, k=k)


_TREE_LINE_RE = re.compile(r"^\s*-\s*0(?:\.\d+)*\s*:\s*(?P<desc>.+?)\s*$")

pick_forced_maze_target_from_rule_path = getattr(
    _direct_execute,
    "pick_forced_maze_target_from_" + "sub" + "goal_path",
)
pick_forced_uci_from_rule = getattr(
    _direct_execute,
    "pick_forced_uci_from_" + "sub" + "goal",
)


def _extract_state_action_rules_with_parser(raw_text: str) -> list[str]:
    """把 LLM 文本规范化为可执行规则列表。

    解析策略：
    1) 先直接调用 slow_output_parser。
    2) 若未解析出规则，则把文本包装成 `Rules:` 列表后重试。
    3) 返回去重后的规则序列（保持优先顺序）。
    """

    def _collect(parsed: dict) -> list[str]:
        out: list[str] = []
        tree = str(parsed.get("tree") or "")
        for ln in tree.splitlines():
            m = _TREE_LINE_RE.match(ln or "")
            if m:
                desc = str(m.group("desc") or "").strip()
                if desc:
                    out.append(desc)
        fix_nodes = parsed.get("fix_nodes") or {}
        if isinstance(fix_nodes, dict):
            for v in fix_nodes.values():
                s = str(v or "").strip()
                if s:
                    out.append(s)
        return out

    text = str(raw_text or "").strip()
    if not text:
        return []

    parsed = parse_chess_slow_output(text)
    state_action_rules = _collect(parsed)
    if state_action_rules:
        return list(dict.fromkeys(state_action_rules))

    bullet_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not bullet_lines:
        return []
    rule_section_text = "Rules:\n" + "\n".join(f"- {ln}" for ln in bullet_lines)
    parsed2 = parse_chess_slow_output(rule_section_text)
    state_action_rules2 = _collect(parsed2)
    return list(dict.fromkeys(state_action_rules2))


class _ExecModuleMaze:
    def __init__(self, role: str, *, decision_log_file: str):
        self.role = role
        self._state_action_rules: deque[str] = deque(maxlen=int(os.getenv("CODING_POLICY_MAX", "5000") or 5000))
        self._lock = threading.Lock()
        self._policy_path: Path | None = None
        try:
            lp = Path(str(decision_log_file))
            self._policy_path = lp.with_name(lp.stem + "_coding_policy.jsonl")
        except Exception:
            self._policy_path = None
            logger.warning("Failed to derive maze coding policy log path", exc_info=True)

    def ingest(self, state_action_rules: list[str], *, model_name: str = "") -> None:
        """写入规则缓存：去重后前插，新规则优先，且可选落盘记录。"""
        if not state_action_rules:
            return
        uniq = [s for s in dict.fromkeys([str(x or "").strip() for x in state_action_rules]) if s]
        if not uniq:
            return
        with self._lock:
            for s in reversed(uniq):
                try:
                    self._state_action_rules.remove(s)
                except ValueError:
                    pass
                self._state_action_rules.appendleft(s)
        if self._policy_path is not None:
            try:
                rec = {"role": self.role, "model": model_name, "state_action_rules": uniq}
                self._policy_path.parent.mkdir(parents=True, exist_ok=True)
                with self._policy_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                logger.warning("Failed to append maze coding policy record", exc_info=True)

    def pick(self, frame, player: str, legal_moves: list) -> tuple[int, int] | None:
        """按缓存顺序逐条匹配，命中第一条可执行规则即返回。"""
        with self._lock:
            items = list(self._state_action_rules)
        for rule_text in items:
            rule_text_norm = rule_text.replace("->", "to")
            forced = pick_forced_maze_target_from_rule_path(frame, player, rule_text_norm, legal_moves)
            if forced is not None and (not legal_moves or forced in set(legal_moves)):
                return forced
        return None


class _ExecModuleChess:
    def __init__(self, *, decision_log_file: str):
        self._state_action_rules: deque[str] = deque(maxlen=int(os.getenv("CODING_POLICY_MAX", "5000") or 5000))
        self._lock = threading.Lock()
        self._policy_path: Path | None = None
        try:
            lp = Path(str(decision_log_file))
            self._policy_path = lp.with_name(lp.stem + "_coding_policy_state_action.jsonl")
        except Exception:
            self._policy_path = None
            logger.warning("Failed to derive chess coding policy log path", exc_info=True)

    def ingest(self, state_action_rules: list[str], *, model_name: str = "") -> None:
        """写入规则缓存：去重后前插，新规则优先，且可选落盘记录。"""
        if not state_action_rules:
            return
        uniq = [s for s in dict.fromkeys([str(x or "").strip() for x in state_action_rules]) if s]
        if not uniq:
            return
        with self._lock:
            for s in reversed(uniq):
                try:
                    self._state_action_rules.remove(s)
                except ValueError:
                    pass
                self._state_action_rules.appendleft(s)
        if self._policy_path is not None:
            try:
                rec = {"model": model_name, "state_action_rules": uniq}
                self._policy_path.parent.mkdir(parents=True, exist_ok=True)
                with self._policy_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                logger.warning("Failed to append chess coding policy record", exc_info=True)

    def pick(self, enhanced_fen_full: str, color: str, legal_moves: list) -> str | None:
        """按缓存顺序逐条匹配，命中第一条合法 UCI 即返回。"""
        with self._lock:
            items = list(self._state_action_rules)
        legal_set = {str(m).strip().lower() for m in (legal_moves or [])}
        for rule_text in items:
            forced = pick_forced_uci_from_rule(enhanced_fen_full, rule_text, legal_moves, color)
            if forced is not None and (not legal_set or str(forced).lower() in legal_set):
                return str(forced).lower()
        return None


class _CodingGeneratorMaze:
    """Maze 规则生成器：仅替换提示词，沿用 LLMAgent 调用链。"""

    def __init__(self, api_setting, llm_settings, role: str, *, decision_log_file: str, model_name: str):
        self.role = role
        self.inner = _BaseLLM(
            api_setting,
            llm_settings,
            model_name,
            LLM_module_prompt=CODING_PROMPT_MAZE,
            log_file=_derive_log_path(decision_log_file, game="maze", side=role, module="decision", k=0),
            game="maze",
            side=role,
            log_agent="coding",
            log_module="decision",
            log_label="CODING_DECISION",
        )

    def generate(self, frame, player: str, legal_moves: list) -> tuple[list[str], dict]:
        """生成并解析规则，同时返回规范化 usage。"""
        prompt = self.inner._build_prompt_maze(frame=frame, legal_moves=legal_moves, role=player)
        role_full = "Red" if player == "red" else "Blue" if player == "blue" else player
        self.inner.logger.info(
            "[%s][%s][model=%s] Prompt sent to LLM:\n%s",
            self.inner.log_label,
            role_full,
            self.inner.model_name,
            prompt,
        )
        text = self.inner._call_llm_api(prompt)
        return _extract_state_action_rules_with_parser(text or ""), normalize_usage(getattr(self.inner, "last_usage", {}) or {})


class _CodingGeneratorChess:
    """Chess 规则生成器：仅替换提示词，沿用 LLMAgent 调用链。"""

    def __init__(self, api_setting, llm_settings, *, decision_log_file: str, model_name: str, side: str | None = None):
        self.inner = _BaseLLM(
            api_setting,
            llm_settings,
            model_name,
            LLM_module_prompt=CODING_PROMPT_CHESS,
            log_file=_derive_log_path(decision_log_file, game="chess", side=side, module="decision", k=0),
            game="chess",
            side=side,
            log_agent="coding",
            log_module="decision",
            log_label="CODING_DECISION",
        )

    def generate(self, enhanced_fen_full: str, color: str, legal_moves: list) -> tuple[list[str], dict]:
        """生成并解析规则，同时返回规范化 usage。"""
        prompt = self.inner._build_prompt_chess(enhanced_FEN_full=enhanced_fen_full, legal_moves=legal_moves, color=color)
        color_full = "White" if color == "w" else "Black" if color == "b" else color
        self.inner.logger.info(
            "[%s][%s][model=%s] Prompt sent to LLM:\n%s",
            self.inner.log_label,
            color_full,
            self.inner.model_name,
            prompt,
        )
        text = self.inner._call_llm_api(prompt)
        return _extract_state_action_rules_with_parser(text or ""), normalize_usage(getattr(self.inner, "last_usage", {}) or {})


class CodingMazePairs:
    def __init__(self, api_setting, llm_settings, role, log_file, model_name: str = ""):
        self.role = role
        resolved_model_name = str(model_name or "").strip()
        if not resolved_model_name:
            raise ValueError("CodingMazePairs requires explicit non-empty model_name")

        decision_log_file = _derive_log_path(log_file, game="maze", side=role, module="decision")
        self.execution_module = _ExecModuleMaze(role=role, decision_log_file=decision_log_file)
        self._generator = _CodingGeneratorMaze(api_setting, llm_settings, role=role, decision_log_file=decision_log_file, model_name=resolved_model_name)
        self._fallback = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_model_name,
            LLM_module_prompt=LLM_PROMPT_MAZE,
            log_file=_derive_log_path(log_file, game="maze", side=role, module="fallback", k=1),
            game="maze",
            side=role,
            log_agent="coding",
            log_module="fallback",
            log_label="CODING_FALLBACK",
        )

        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._usage_lock = threading.Lock()
        self.last_request_exception = None
        self.last_request_status_code = None
        self.last_action_source = "coding_fallback"

        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._latest: tuple[object, str, list] | None = None
        self._rev = 0
        self._worker = threading.Thread(target=self._bg_loop, name=f"CodingGenMaze[{role}]", daemon=True)
        self._worker.start()

    def _add_usage_tokens(self, usage: dict) -> None:
        with self._usage_lock:
            self.total_tokens = add_total_tokens(self.total_tokens, usage)

    def _bg_loop(self):
        """后台循环：消费最新状态，生成规则并更新执行缓存。"""
        seen = -1
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or (self._latest is not None and self._rev != seen))
                if self._stop.is_set():
                    return
                seen = self._rev
                snap = self._latest
            if not snap:
                continue
            frame, role, legal_moves = snap
            state_action_rules, usage = self._generator.generate(frame, role, legal_moves)
            self.execution_module.ingest(state_action_rules, model_name=self._generator.inner.model_name)
            self._add_usage_tokens(usage)

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def get_action(self, frame, player=None, legal_moves=None):
        """前台决策：投递状态→查缓存规则→未命中 fallback LLM。"""
        role = player if isinstance(player, str) and player.strip() else self.role
        with self._cv:
            self._latest = (frame, role, legal_moves)
            self._rev += 1
            self._cv.notify()

        cached = self.execution_module.pick(frame, role, legal_moves or [])
        if cached is not None:
            self.last_usage = {}
            self.last_action_source = "coding_feedback"
            return cached

        action = self._fallback.get_action(frame=frame, role=role, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self._fallback, "last_usage", {}) or {})
        self.last_request_exception = getattr(self._fallback, "last_request_exception", None)
        self.last_request_status_code = getattr(self._fallback, "last_request_status_code", None)
        self.last_action_source = "coding_fallback"
        self._add_usage_tokens(self.last_usage)
        return action


class CodingChessPairs:
    def __init__(self, api_setting, llm_settings, log_file, model_name: str = "", side: str | None = None):
        resolved_model_name = str(model_name or "").strip()
        if not resolved_model_name:
            raise ValueError("CodingChessPairs requires explicit non-empty model_name")
        decision_log_file = _derive_log_path(log_file, game="chess", side=side, module="decision")
        self.execution_module = _ExecModuleChess(decision_log_file=decision_log_file)
        self._generator = _CodingGeneratorChess(
            api_setting,
            llm_settings,
            decision_log_file=decision_log_file,
            model_name=resolved_model_name,
            side=side,
        )
        self._fallback = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_model_name,
            LLM_module_prompt=LLM_PROMPT_CHESS,
            log_file=_derive_log_path(log_file, game="chess", side=side, module="fallback", k=1),
            game="chess",
            side=side,
            log_agent="coding",
            log_module="fallback",
            log_label="CODING_FALLBACK",
        )

        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._usage_lock = threading.Lock()
        self.last_request_exception = None
        self.last_request_status_code = None
        self.last_action_source = "coding_fallback"

        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._latest: tuple[object, str, list] | None = None
        self._rev = 0
        self._worker = threading.Thread(target=self._bg_loop, name="CodingGenChess", daemon=True)
        self._worker.start()

    def _add_usage_tokens(self, usage: dict) -> None:
        with self._usage_lock:
            self.total_tokens = add_total_tokens(self.total_tokens, usage)

    def _bg_loop(self):
        """后台循环：消费最新状态，生成规则并更新执行缓存。"""
        seen = -1
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or (self._latest is not None and self._rev != seen))
                if self._stop.is_set():
                    return
                seen = self._rev
                snap = self._latest
            if not snap:
                continue
            enhanced_fen, color, legal_moves = snap
            state_action_rules, usage = self._generator.generate(enhanced_fen, color, legal_moves)
            self.execution_module.ingest(state_action_rules, model_name=self._generator.inner.model_name)
            self._add_usage_tokens(usage)

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """前台决策：投递状态→查缓存规则→未命中 fallback LLM。"""
        with self._cv:
            self._latest = (enhanced_FEN_full, color, legal_moves)
            self._rev += 1
            self._cv.notify()

        cached = self.execution_module.pick(enhanced_FEN_full, color, legal_moves or [])
        if cached is not None:
            self.last_usage = {}
            self.last_action_source = "coding_feedback"
            return cached

        action = self._fallback.get_action(
            enhanced_FEN_full=enhanced_FEN_full,
            color=color,
            legal_moves=legal_moves,
        )
        self.last_usage = normalize_usage(getattr(self._fallback, "last_usage", {}) or {})
        self.last_request_exception = getattr(self._fallback, "last_request_exception", None)
        self.last_request_status_code = getattr(self._fallback, "last_request_status_code", None)
        self.last_action_source = "coding_fallback"
        self._add_usage_tokens(self.last_usage)
        return action
