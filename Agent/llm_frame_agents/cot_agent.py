"""CoT 基线智能体（仅提示词差异的 LLMAgent 包装层）。

设计说明：
- 与统一版 `LLMAgent` 走同一套运行路径
- 唯一差异是提示词模板（CoT 提示词）

决策流程：
1) 读取 CoT 提示词模板。
2) 调用 inner.get_action 完成单步决策。
3) 同步并累加 token 使用统计。

接口：
- Maze: get_action(frame, player, legal_moves)
- Chess: get_action(enhanced_FEN_full, color, legal_moves)
"""

import os

from Agent.llm_base_agents.llm_agent import LLMAgent as _BaseLLM
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROMPTS = {
    "maze": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Maze_cot_llm_agent_prompt.txt"),
    "chess": os.path.join(PROJECT_ROOT, "Agent", "prompt", "Chess_cot_llm_agent_prompt.txt"),
}

def _derive_decision_log_path(log_file: str, *, game: str, side: str | None = None) -> str:
    return make_log_file(log_file, game=game, side=side, agent="cot", module="decision")


class CoTMaze:
    def __init__(self, api_setting, llm_settings, role, log_file, model_name: str = ""):
        prompt = open(PROMPTS["maze"], "r", encoding="utf-8").read()
        self.role = role
        resolved_model_name = str(model_name or "").strip()
        if not resolved_model_name:
            raise ValueError("CoTMaze requires explicit non-empty model_name")

        self.inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_model_name,
            LLM_module_prompt=prompt,
            log_file=_derive_decision_log_path(log_file, game="maze", side=role),
            game="maze",
            side=role,
            log_agent="cot",
            log_module="decision",
            log_label="COT",
        )
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, frame, player=None, legal_moves=None):
        """Maze 决策入口：角色归一化后调用 inner，并同步 usage。"""
        role = player if isinstance(player, str) and player.strip() else self.role
        action = self.inner.get_action(frame=frame, role=role, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        return action


class CoTChess:
    def __init__(self, api_setting, llm_settings, log_file, model_name: str = "", side: str | None = None):
        prompt = open(PROMPTS["chess"], "r", encoding="utf-8").read()
        resolved_model_name = str(model_name or "").strip()
        if not resolved_model_name:
            raise ValueError("CoTChess requires explicit non-empty model_name")

        self.inner = _BaseLLM(
            api_setting,
            llm_settings,
            resolved_model_name,
            LLM_module_prompt=prompt,
            log_file=_derive_decision_log_path(log_file, game="chess", side=side),
            game="chess",
            side=side,
            log_agent="cot",
            log_module="decision",
            log_label="COT",
        )
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """Chess 决策入口：调用 inner 并同步 usage。"""
        action = self.inner.get_action(
            enhanced_FEN_full=enhanced_FEN_full,
            color=color,
            legal_moves=legal_moves,
        )
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        return action
