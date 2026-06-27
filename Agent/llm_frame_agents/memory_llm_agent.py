"""MemoryLLM 记忆增强智能体。

机制说明：
1) 每步沿用基础 LLM 决策流程。
2) 将最近若干步的成功动作/失败原因写入记忆。
3) 下一步把记忆块拼接进提示词，减少重复错误。

决策流程：
- 组装带 memory_block 的 prompt
- 调用 inner LLM 获取动作
- 同步 usage 统计
- 成功写 chosen_action，失败写 parse_failed 原因
"""

from __future__ import annotations

from contextlib import contextmanager

from Agent.llm_base_agents.llm_agent import LLMAgent as _BaseLLM
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total


def _derive_decision_log_path(log_file: str, *, game: str, side: str | None = None) -> str:
    return make_log_file(log_file, game=game, side=side, agent="memory", module="decision")


class _MemoryLLMShared:
    def __init__(self, inner: _BaseLLM, base_prompt: str, max_memory: int = 10):
        self.inner = inner
        self.base_prompt = base_prompt
        self.max_memory = int(max_memory) if max_memory else 10
        self.memory: list[str] = []
        self._turn = 0
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
        self.last_request_exception = None
        self.last_request_status_code = None

    def _append_memory(self, text: str) -> None:
        """追加一条记忆，并按 max_memory 保留最近窗口。"""
        s = str(text or "").strip()
        if not s:
            return
        self.memory.append(s)
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory :]

    def _memory_block(self, extra_line: str = "") -> str:
        """把历史记忆拼接为可直接注入提示词的文本块。"""
        if not self.memory:
            return ""
        head = (
            "[Memory]\n"
            "- Below are previous responses from this game (recent last).\n"
            "- Use them to avoid repeating failed plans and choose a legal move.\n"
        )
        if extra_line:
            head += f"{extra_line}\n"
        tail = "\n\n".join(self.memory[-self.max_memory:])
        return "\n".join([head, tail, ""]).strip("\n") + "\n\n"

    def _sync_usage(self) -> None:
        """同步本轮调用 usage，并累加到 total_tokens。"""
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        self.last_request_exception = getattr(self.inner, "last_request_exception", None)
        self.last_request_status_code = getattr(self.inner, "last_request_status_code", None)

    @contextmanager
    def _using_prompt_template(self, prompt_template: str):
        old = self.inner.prompt_template
        self.inner.prompt_template = prompt_template
        try:
            yield
        finally:
            self.inner.prompt_template = old

    def _maze_parse_fail_reason(self) -> str:
        if getattr(self.inner, "last_request_exception", None) is not None:
            return "request_exception"
        status = getattr(self.inner, "last_request_status_code", None)
        if isinstance(status, int) and status >= 400:
            return f"http_status_{status}"

        text = str(getattr(self.inner, "last_response_text", "") or "").strip()
        if not text:
            return "empty_response"

        last_sentence = self.inner._maze_parser.get_last_sentence(text)
        if not last_sentence:
            return "empty_last_sentence"

        target = self.inner._maze_parser.find_target_junction_in_sentence(last_sentence)
        if not self.inner._maze_parser.is_valid_target_junction(target):
            return "missing_or_invalid_target"
        return "unknown"

    def _chess_parse_fail_reason(self) -> str:
        if getattr(self.inner, "last_request_exception", None) is not None:
            return "request_exception"
        status = getattr(self.inner, "last_request_status_code", None)
        if isinstance(status, int) and status >= 400:
            return f"http_status_{status}"

        text = str(getattr(self.inner, "last_response_text", "") or "").strip()
        if not text:
            return "empty_response"

        last_sentence = self.inner._chess_parser.get_last_sentence(text)
        if not last_sentence:
            return "empty_last_sentence"

        uci_move = self.inner._chess_parser.find_uci_in_sentence(last_sentence)
        if not self.inner._chess_parser.is_valid_uci(uci_move):
            return "missing_or_invalid_uci"
        return "unknown"


class MemoryLLMMaze(_MemoryLLMShared):
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, role, max_memory=10):
        self.role = role
        inner = _BaseLLM(
            api_setting,
            llm_settings,
            model_name,
            LLM_module_prompt=None,
            log_file=_derive_decision_log_path(log_file, game="maze", side=role),
            game="maze",
            side=role,
            log_agent="memory",
            log_module="decision",
            log_label="MEMORY",
        )
        super().__init__(inner=inner, base_prompt=base_prompt, max_memory=max_memory)
        self.slow_module = None
        self.slow_module_opponent = None
        self.fast_module = self

    def get_action(self, frame, player: str, legal_moves: list):
        """Maze 单步决策：注入记忆→调用模型→记录成功/失败经验。"""
        self._turn += 1

        prompt_template = self.base_prompt.format(
            memory_block=self._memory_block(),
            player="{player}",
            moves="{moves}",
            maze_description="{maze_description}",
        )
        with self._using_prompt_template(prompt_template):
            target = self.inner.get_action(frame=frame, role=player, legal_moves=legal_moves)

        self._sync_usage()
        if target is not None:
            self._append_memory(f"turn={self._turn} player={player} chosen_action={target}")
            return target

        reason = self._maze_parse_fail_reason()
        self._append_memory(f"turn={self._turn} player={player} parse_failed={reason}")
        return None


class MemoryLLMChess(_MemoryLLMShared):
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, side: str | None = None, max_memory=10):
        inner = _BaseLLM(
            api_setting,
            llm_settings,
            model_name,
            LLM_module_prompt=None,
            log_file=_derive_decision_log_path(log_file, game="chess", side=side),
            game="chess",
            side=side,
            log_agent="memory",
            log_module="decision",
            log_label="MEMORY",
        )
        super().__init__(inner=inner, base_prompt=base_prompt, max_memory=max_memory)
        self.slow_module = None
        self.slow_module_opponent = None
        self.fast_module = self

    def get_action(self, state, player: str, legal_moves: list):
        """Chess 单步决策：注入记忆→调用模型→记录成功/失败经验。"""
        self._turn += 1
        prompt_template = self.base_prompt.format(
            memory_block=self._memory_block(),
            my_color="{my_color}",
            opponent_color="{opponent_color}",
            enhanced_FEN_full="{enhanced_FEN_full}",
            moves="{moves}",
            fen_description="{fen_description}",
        )
        with self._using_prompt_template(prompt_template):
            move = self.inner.get_action(
                enhanced_FEN_full=state,
                color=player,
                legal_moves=legal_moves,
            )

        self._sync_usage()

        if move is not None:
            chosen = str(move).strip().lower()
            self._append_memory(f"turn={self._turn} player={player} chosen_action={chosen}")
            return chosen

        reason = self._chess_parse_fail_reason()
        self._append_memory(f"turn={self._turn} player={player} parse_failed={reason}")
        return None
