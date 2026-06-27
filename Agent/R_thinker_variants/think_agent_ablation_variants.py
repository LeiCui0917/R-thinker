"""ThinkAgent 消融变体集合（薄包装）。

包含两类策略：
- without opponent slow：关闭敌方慢模块，偏进攻。
- without self slow：关闭我方慢模块，偏防守。

说明：
- 全部复用 `ThinkAgent` 主流程，仅通过参数开关切换策略。
- 该文件只做实验封装，不引入额外业务逻辑。
"""

from __future__ import annotations

import os

from Agent.R_Thinker.think_agent import ThinkAgent
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

PROMPTS = {
    'maze': {
        'fast': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Maze_fast_think_module_prompt.txt'),
        'slow': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Maze_slow_think_module_prompt.txt'),
        'slow_opponent': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Maze_slow_think_module_prompt.txt'),
    },
    'chess': {
        'fast': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Chess_fast_think_module_prompt.txt'),
        'slow': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Chess_slow_think_module_prompt.txt'),
        'slow_opponent': os.path.join(PROJECT_ROOT, 'Agent', 'prompt', 'Chess_slow_think_module_prompt.txt'),
    }
}


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class _ThinkAgentOnlySelfSlow(ThinkAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('enable_self_slow', True)
        kwargs.setdefault('enable_opp_slow', False)
        kwargs.setdefault('decision_mode_policy', 'attack_only')
        super().__init__(*args, **kwargs)


class _ThinkAgentOnlyOpponentSlow(ThinkAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('enable_self_slow', False)
        kwargs.setdefault('enable_opp_slow', True)
        kwargs.setdefault('decision_mode_policy', 'defend_only')
        super().__init__(*args, **kwargs)


class ThinkAgentWithoutOpponentMaze:
    def __init__(self, api_setting, llm_settings, role, log_file, model_name: str = ""):
        fast_path = PROMPTS['maze']['fast']
        slow_path = PROMPTS['maze']['slow']
        fast = _read_text(fast_path)
        slow = _read_text(slow_path)
        self.inner = _ThinkAgentOnlySelfSlow(
            api_setting,
            llm_settings,
            fast,
            slow,
            log_file,
            model_name=model_name,
            game_type='maze',
            slow_module_opponent_prompt=slow,
            log_agent='two',
            side=role,
        )
        self.role = role
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, frame, player=None, legal_moves=None):
        """Maze 包装入口：调用 inner 并同步 usage 统计。"""
        if legal_moves is None and isinstance(player, list):
            legal_moves = player
            player = None
        role = player if isinstance(player, str) and player.strip() else self.role
        action = self.inner.get_action(state=frame, player=role, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self.inner, 'last_usage', {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, 'total_tokens', None),
            self.last_usage,
        )
        return action


class ThinkAgentWithoutSelfMaze:
    def __init__(self, api_setting, llm_settings, role, log_file, model_name: str = ""):
        fast_path = PROMPTS['maze']['fast']
        slow_opp_path = PROMPTS['maze']['slow_opponent']
        fast = _read_text(fast_path)
        slow_opp = _read_text(slow_opp_path)
        self.inner = _ThinkAgentOnlyOpponentSlow(
            api_setting,
            llm_settings,
            fast,
            slow_opp,
            log_file,
            model_name=model_name,
            game_type='maze',
            slow_module_opponent_prompt=slow_opp,
            log_agent='tws',
            side=role,
        )
        self.role = role
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, frame, player=None, legal_moves=None):
        """Maze 包装入口：调用 inner 并同步 usage 统计。"""
        if legal_moves is None and isinstance(player, list):
            legal_moves = player
            player = None
        role = player if isinstance(player, str) and player.strip() else self.role
        action = self.inner.get_action(state=frame, player=role, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self.inner, 'last_usage', {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, 'total_tokens', None),
            self.last_usage,
        )
        return action


class ThinkAgentWithoutOpponentChess:
    def __init__(self, api_setting, llm_settings, log_file, model_name: str = "", side: str | None = None):
        fast = _read_text(PROMPTS['chess']['fast'])
        slow = _read_text(PROMPTS['chess']['slow'])
        self.inner = _ThinkAgentOnlySelfSlow(
            api_setting,
            llm_settings,
            fast,
            slow,
            log_file,
            model_name=model_name,
            game_type='chess',
            slow_module_opponent_prompt=slow,
            log_agent='two',
            side=side,
        )
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """Chess 包装入口：调用 inner 并同步 usage 统计。"""
        action = self.inner.get_action(state=enhanced_FEN_full, player=color, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self.inner, 'last_usage', {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, 'total_tokens', None),
            self.last_usage,
        )
        return action


class ThinkAgentWithoutSelfChess:
    def __init__(self, api_setting, llm_settings, log_file, model_name: str = "", side: str | None = None):
        fast = _read_text(PROMPTS['chess']['fast'])
        slow_opp = _read_text(PROMPTS['chess']['slow_opponent'])
        self.inner = _ThinkAgentOnlyOpponentSlow(
            api_setting,
            llm_settings,
            fast,
            slow_opp,
            log_file,
            model_name=model_name,
            game_type='chess',
            slow_module_opponent_prompt=slow_opp,
            log_agent='tws',
            side=side,
        )
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """Chess 包装入口：调用 inner 并同步 usage 统计。"""
        action = self.inner.get_action(state=enhanced_FEN_full, player=color, legal_moves=legal_moves)
        self.last_usage = normalize_usage(getattr(self.inner, 'last_usage', {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, 'total_tokens', None),
            self.last_usage,
        )
        return action
