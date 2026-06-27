"""FAST 决策模块（Chess + Maze 通用）。

定位：
- 这是 ThinkAgent 的前台动作执行器，负责“当前步立即出动作”。
- 与基础 `LLMAgent` 的调用链保持一致，仅额外接收调度信号：
    `decision_mode`、`subgoal_text`、`subgoal_path`。

决策流程：
1) `build_prompt` 按游戏类型组装提示词。
2) `_call_llm_api` 请求模型并更新 usage。
3) `get_action` 解析输出并返回可执行动作。
"""

import logging
import os
import re

import requests

from Env.ChessEnv.chess_output_parser import ChessOutputParser
from Env.ChessEnv.fen_translator import FenTranslator
from Env.MazeEnv.maze_output_parser import MazeOutputParser
from Env.MazeEnv.maze_translator import MazeTranslator
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_total_tokens
from ..utils.llm_http_client import post_json_with_retry


def _attach_file_handler(logger: logging.Logger, log_file: str | None):
    """Attach a FileHandler to a per-instance logger (LLMAgent-style)."""
    if not log_file:
        return None
    try:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.debug("[FAST] Logger initialized with file: %s", log_file)
        return fh
    except Exception:
        return None

class FastThinkModule:
    """FAST 模块：每次调用返回一个可执行动作。

    返回约定：
    - chess: 返回 UCI 字符串（如 `e2e4`）
    - maze : 返回目标坐标 `(x, y)`
    """

    def __init__(
        self,
        api_setting: dict,
        llm_settings: dict,
        model_name: str,
        fast_module_prompt: str,
        log_file: str | None,
        game_type: str = "chess",
        game_name: str | None = None,
        log_agent: str = "fast",
        side: str | None = None,
    ):
        # Backward compatibility: some older/debug scripts pass game_name.
        if game_name is not None:
            game_type = game_name
        self.game_type = str(game_type or "chess").lower()
        log_file = make_log_file(log_file, game=self.game_type, side=side, agent=log_agent, module="fast") if log_file else log_file

        self.logger = logging.getLogger(f"FastThinkModule_{self.game_type}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._file_handler = _attach_file_handler(self.logger, log_file)
        self.api_key = api_setting.get("api_key", "")
        self.base_url = api_setting.get("base_url", "")
        self.timeout = api_setting.get("timeout", 120) or 120
        self.max_retries = int(api_setting.get("max_retries", 1) or 1)

        self.temperature = llm_settings.get("temperature", 0.6)
        self.num_lookahead_tokens = llm_settings.get("num_lookahead_tokens", 5000)
        self.max_tokens = llm_settings.get("max_tokens", 100)
        self.default_model = llm_settings.get("default_model", "")

        self.model_name = str(model_name or "").strip()
        if not self.model_name:
            raise ValueError("FastThinkModule requires explicit non-empty model_name")
        self.prompt_template = fast_module_prompt
        self.last_action_source = "fast"
        self.last_usage = {}
        self.total_tokens = 0
        # Filled only when the underlying HTTP request (or response decode) failed.
        # Used by experiment runners to distinguish real API failures from parsing issues.
        self.last_request_exception = None
        self.last_request_status_code = None

        # Use Session to allow toggling env proxy usage (trust_env).
        self._session_env = requests.Session()
        self._session_env.trust_env = True

        if self.game_type == "maze":
            self._parser = MazeOutputParser()
            self._translator = MazeTranslator()
        else:
            self._parser = ChessOutputParser()
            self._translator = FenTranslator()

    # ---------------- Core 1: Construct Prompt ----------------
    def build_prompt(
        self,
        state,
        legal_moves: list,
        player: str,
        decision_mode: str = "ATTACK",
        subgoal_text: str = "",
        subgoal_path: str = "",
    ) -> str:
        """构造 FAST 提示词，并注入调度模块给出的子目标信号。"""
        prompt_template = self.prompt_template

        # Shared extra placeholders (ThinkAgent control signals)
        # Keep naming stable across chess/maze prompts.
        subgoal_title = "Closest subgoal to victory:" if decision_mode != "DEFEND" else "Opponent's closest subgoal to victory:"
        route_title = "Route from this subgoal to victory (root):" if decision_mode != "DEFEND" else "Opponent's route from this subgoal to victory (root):"

        if self.game_type == "maze":
            frame = state if isinstance(state, dict) else {}

            # Align with Agent/llm_base_agents/llm_agent.py: use the same translator entry.
            maze_description = ""
            try:
                maze_description = self._translator.translate(frame, role=player)
            except Exception:
                maze_description = ""

            # Display names: avoid Red/Blue wording in prompts.
            player_display = "Red Red" if player == "red" else "Blue Red" if player == "blue" else player
            opponent_display = "Blue Red" if player == "red" else "Red Red" if player == "blue" else "Opponent"

            post_route_instruction = ""
            if decision_mode == "DEFEND":
                post_route_instruction = (
                    "In Race mode, you cannot move onto the opponent's cell and opponent-used cells are blocked. "
                    "Defend by selecting targets that worsen the opponent's best route while keeping yours feasible."
                )

            mapping = {
                "player": player_display,
                "opponent": opponent_display,
                "moves": legal_moves,
                "maze_description": maze_description,
                "frame": "",
                # think-only extras
                "decision_mode": decision_mode,
                "subgoal_text": subgoal_text,
                "subgoal_path": subgoal_path,
                "subgoal_title": subgoal_title,
                "route_title": route_title,
                "post_route_instruction": post_route_instruction,
            }
            return prompt_template.format(**mapping)

        # chess
        fen_description = ""
        try:
            fen_description = self._translator.translate(state)
        except Exception:
            fen_description = ""

        color_full = "White" if player == "w" else "Black" if player == "b" else player
        opponent_color_full = "Black" if color_full == "White" else "White"

        mapping = {
            # Align with Agent/llm_base_agents/llm_agent.py placeholder naming.
            "my_color": color_full,
            "opponent_color": opponent_color_full,
            "enhanced_FEN_full": state,
            "moves": legal_moves,
            "fen_description": fen_description,
            # extra convenience placeholders sometimes used by Think prompts
            "color": color_full,
            # think-only extras
            "decision_mode": decision_mode,
            "subgoal_text": subgoal_text,
            "subgoal_path": subgoal_path,
            "subgoal_title": subgoal_title,
            "route_title": route_title,
            "post_route_instruction": "",
        }
        return prompt_template.format(**mapping)

    # ---------------- Core 2: Call LLM ----------------
    def _call_llm_api(self, prompt: str) -> str:
        """调用 LLM 接口并同步 usage 状态，返回原始文本。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Clear any previous failure marker at the start of a new request.
        self.last_request_exception = None
        self.last_request_status_code = None
        self.last_usage = {}

        j, status, exc = post_json_with_retry(
            url=self.base_url,
            headers=headers,
            payload=payload,
            timeout=self.timeout,
            max_retries=self.max_retries,
            session_env=self._session_env,
        )
        self.last_request_status_code = status
        if exc is not None:
            self.last_request_exception = exc
            self.logger.exception("[LLMAgent] LLM request failed: %s", exc)
        if not j:
            return ""

        self.logger.info("[LLMAgent] LLM raw response (JSON): %s", j)
        content = j.get("choices", [])[0].get("message", {}).get("content", "").strip()
        try:
            self.last_usage = normalize_usage(j.get("usage", {}))
            self.total_tokens = add_total_tokens(self.total_tokens, self.last_usage)
        except Exception:
            self.last_usage = {}
        return content

    # ---------------- Core 3: Main Decision Logic ----------------
    def get_action(
        self,
        state,
        player: str,
        legal_moves: list,
        decision_mode: str = "ATTACK",
        subgoal_text: str = "",
        subgoal_path: str = "",
    ):
        """主入口：构造提示词→调用模型→按游戏类型解析动作。"""
        self.last_action_source = "fast"
        try:
            prompt = self.build_prompt(
                state,
                legal_moves,
                player,
                decision_mode=decision_mode,
                subgoal_text=subgoal_text,
                subgoal_path=subgoal_path,
            )
        except Exception as e:
            self.logger.exception("[FAST] build_prompt failed: %s", e)
            return None

        # Log prompt sent (LLMAgent style)
        label = player
        if self.game_type != "maze":
            label = "White" if player == "w" else "Black" if player == "b" else player
        self.logger.info("[LLM_AGENT][%s][model=%s] Prompt sent to LLM:\n%s", label, self.model_name, prompt)

        text = self._call_llm_api(prompt)
        if not text:
            self.logger.error("[LLMAgent] No LLM response")
            return None

        if self.game_type == "maze":
            last_sentence = self._parser.get_last_sentence(text)
            target = self._parser.find_target_junction_in_sentence(last_sentence)
            if self._parser.is_valid_target_junction(target):
                self.logger.info("[LLMAgent] Valid move: %s", target)
                return target
            return None

        # chess
        last_sentence = self._parser.get_last_sentence(text)
        uci_move = self._parser.find_uci_in_sentence(last_sentence)
        if self._parser.is_valid_uci(uci_move):
            self.logger.info("[LLMAgent] Valid move: %s", uci_move)
            return uci_move
        return None

