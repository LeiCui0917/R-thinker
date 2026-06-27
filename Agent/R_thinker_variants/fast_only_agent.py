"""FastOnly 变体（仅快模块）。

机制说明：
- 只调用 `FastThinkModule` 进行实时动作决策。
- 不执行任何慢模块扩展与树更新。
- 仍注入根节点语义（节点 0）以对齐 ThinkAgent 的提示词结构。
"""

from Agent.R_Thinker.fast_think_module import FastThinkModule
from Agent.utils.usage_utils import normalize_usage, add_wrapper_tokens_from_inner_total


def _root_subgoal_text_chess(player: str) -> str:
    p = (player or "").strip().lower()
    if p == "w":
        opponent_color, self_color = "Black", "White"
    elif p == "b":
        opponent_color, self_color = "White", "Black"
    else:
        opponent_color, self_color = "Opponent", "Our"
    # Keep alignment with ThinkAgent root-node semantics for chess.
    return f"{opponent_color} King is captured by {self_color} piece at (x, n)"


def _root_subgoal_text_maze(frame: dict, role: str) -> str:
    # Mirror ThinkAgent._root_desc semantics for maze.
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

class FastOnlyMaze:
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, role):
        # Keep original prompt template; inject root-node guidance text (node 0).
        self.inner = FastThinkModule(
            api_setting,
            llm_settings,
            model_name,
            base_prompt,
            log_file,
            game_type="maze",
            log_agent="fo",
            side=role,
        )
        self.role = role
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
    def get_action(self, frame, player=None, legal_moves=None):
        """Maze 单步决策：构造根子目标并调用快模块。"""
        if legal_moves is None and isinstance(player, list):
            legal_moves = player
            player = None
        role = player if isinstance(player, str) and player.strip() else self.role
        root_text = _root_subgoal_text_maze(frame if isinstance(frame, dict) else {}, role)
        action = self.inner.get_action(
            state=frame,
            player=role,
            legal_moves=legal_moves,
            decision_mode="ATTACK",
            subgoal_text=root_text,
            subgoal_path="- 0: " + str(root_text),
        )
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        return action

class FastOnlyChess:
    def __init__(self, api_setting, llm_settings, model_name, base_prompt, log_file, side: str | None = None):
        # Keep original prompt template; inject root-node guidance text (node 0).
        self.inner = FastThinkModule(
            api_setting,
            llm_settings,
            model_name,
            base_prompt,
            log_file,
            game_type="chess",
            log_agent="fo",
            side=side,
        )
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self._last_inner_total: int = 0
    def get_action(self, enhanced_FEN_full, color, legal_moves):
        """Chess 单步决策：构造根子目标并调用快模块。"""
        root_text = _root_subgoal_text_chess(color)
        action = self.inner.get_action(
            state=enhanced_FEN_full,
            player=color,
            legal_moves=legal_moves,
            decision_mode="ATTACK",
            subgoal_text=root_text,
            subgoal_path="- 0: " + str(root_text),
        )
        self.last_usage = normalize_usage(getattr(self.inner, "last_usage", {}) or {})
        self.total_tokens, self._last_inner_total = add_wrapper_tokens_from_inner_total(
            self.total_tokens,
            self._last_inner_total,
            getattr(self.inner, "total_tokens", None),
            self.last_usage,
        )
        return action
