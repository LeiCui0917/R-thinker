"""SLOW 思考模块（Chess + Maze 通用）。

定位：
- 负责“慢思考/语义树更新”，不直接给最终动作。
- 与 `FastThinkModule` 的结构对齐，但输出是 `tree/fix_nodes`。

流程：
1) `build_prompt` 构造慢思考提示词。
2) `_call_llm_api` 请求模型并记录 usage。
3) `get_guidance` 解析并返回树更新结果。
"""

import json
import logging
import re

import requests

from Env.ChessEnv.fen_translator import FenTranslator
from Env.MazeEnv.maze_translator import MazeTranslator
from .utils.slow_output_parser import parse_chess_slow_output
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_total_tokens
from ..utils.llm_http_client import post_json_with_retry

def _attach_file_handler(logger: logging.Logger, log_file: str | None):
    """Attach a FileHandler to a per-instance logger.

    Keep behavior intentionally quiet on failure (no extra logs), matching the
    FastThinkModule / LLMAgent style.
    """
    if not log_file:
        return None
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.debug("[SLOW] Logger initialized with file: %s", log_file)
        return fh
    except Exception:
        return None

class SlowThinkModule:
    """慢思考模块：负责高层策略推演与语义树增量更新。"""
    
    # ============================================================
    # Initialization and Configuration
    # ============================================================
    
    def __init__(
        self,
        api_setting: dict,
        llm_settings: dict,
        model_name: str,
        slow_module_prompt: str,
        log_file: str | None,
        game_type: str = "chess",
        log_agent: str = "slow",
        side: str | None = None,
        log_module: str = "slow_self",
    ):
        """初始化慢思考模块：配置模型、日志与游戏适配器。"""
        self.game_type = str(game_type or "chess").lower()
        log_file = make_log_file(log_file, game=self.game_type, side=side, agent=log_agent, module=log_module) if log_file else log_file

        self._log_tag = "Maze" if self.game_type == "maze" else "Chess"
        self.logger = logging.getLogger(f"SlowThinkModule_{self._log_tag}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._file_handler = _attach_file_handler(self.logger, log_file)

        # API Configuration (Strict Mode)
        self.api_key = api_setting["api_key"]
        self.base_url = api_setting["base_url"]
        self.max_retries = int((api_setting or {}).get("max_retries", 1) or 1)
        # Avoid indefinite hangs if config doesn't specify a timeout.
        self.timeout = (api_setting or {}).get("timeout", 120) or 120

        # LLM Parameter Configuration
        self.temperature = llm_settings["temperature"]
        self.max_tokens = llm_settings["max_tokens"]
        # Model and Template Configuration
        self.model_name = str(model_name or "").strip()
        if not self.model_name:
            raise ValueError("SlowThinkModule requires explicit non-empty model_name")
        self.prompt_template = slow_module_prompt
        self.last_usage = {}
        self.total_tokens = 0
        # Filled only when the underlying HTTP request (or response decode) failed.
        self.last_request_exception = None
        self.last_request_status_code = None

        # Use Session to allow toggling env proxy usage (trust_env).
        self._session_env = requests.Session()
        self._session_env.trust_env = True

        # 适配器（翻译器 + 输出解析器）
        # 这里是运行期唯一与 game_type 相关的部分。
        if self.game_type == "maze":
            self._translator = MazeTranslator()
            self._parser = parse_chess_slow_output
        else:
            self._translator = FenTranslator()
            self._parser = parse_chess_slow_output

    # ============================================================
    # Core 1: Prompt Construction
    # ============================================================

    def build_prompt(
        self,
        state,
        player: str,
        guidance_line_tree_last: str,
        expand_node_id: str = "",
        expand_node_desc: str = "",
        legal_moves: list | None = None,
        extra_instructions: str = "",
    ) -> str:
        """构造慢思考提示词（统一支持 Chess/Maze）。"""

        moves_text = str(legal_moves or [])

        if self.game_type == "maze":
            # Mirror FastThinkModule logic
            frame = state if isinstance(state, dict) else {}
            # Do not include the raw ASCII map in prompts (token-heavy and not needed).
            frame_str = ""
            try:
                # Align with Agent/llm_base_agents/llm_agent.py: role-specific translation (dynamic blocking perspective).
                maze_description = self._translator.translate(frame, role=player)
            except Exception:
                maze_description = ""

            goal = frame.get("goal") if isinstance(frame, dict) else None
            gx = goal.get("x") if isinstance(goal, dict) else None
            gy = goal.get("y") if isinstance(goal, dict) else None
            shared_goal_text = f"({gx}, {gy})" if (gx is not None and gy is not None) else "(unknown)"

            # Race-only: red position is not a terminal condition; no need to anchor trees on capture.

            if player == "red":
                my_role = "Red Red"
                opponent_role = "Blue Red"
                root_node = f"Red Red reaches the shared goal zone at {shared_goal_text}"
            elif player == "blue":
                my_role = "Blue Red"
                opponent_role = "Red Red"
                root_node = f"Blue Red reaches the shared goal zone at {shared_goal_text}"
            else:
                raise ValueError(f"Invalid maze player role: {player!r}. Expected 'red' or 'blue'.")

            # Extract a concrete expansion target from NodeDesc to reduce ambiguity.
            # Rule: take the FIRST (x,y) that appears in expand_node_desc.
            expand_target_xy = "(unknown)"
            try:
                match = re.search(r"\((\d+)\s*,\s*(\d+)\)", str(expand_node_desc))
                if match:
                    expand_target_xy = f"({int(match.group(1))}, {int(match.group(2))})"
            except Exception:
                expand_target_xy = "(unknown)"

            goal_for_player = shared_goal_text

            # Prompt display names: use Red Red / Blue Red (no Blue wording).
            opponent = "Blue Red" if player == "red" else "Red Red"

            prompt = self.prompt_template.format(
                guidance_line_tree_last=guidance_line_tree_last,
                frame=frame_str,
                maze_description=maze_description,
                expand_node_id=expand_node_id,
                expand_node_desc=expand_node_desc,
                expand_target_xy=expand_target_xy,
                player=my_role,
                my_role=my_role,
                opponent_role=opponent_role,
                root_node=root_node,
                goal=goal_for_player,
                opponent=opponent,
                moves=moves_text,
            )
            if extra_instructions:
                prompt += f"\n\n{extra_instructions}"
            return prompt

        # chess (mirror FastThinkModule naming and translator usage)
        try:
            fen_description = self._translator.translate(state)
        except Exception:
            fen_description = ""

        color_full = "White" if player == "w" else "Black" if player == "b" else player
        opponent_color_full = "Black" if color_full == "White" else "White"

        prompt = self.prompt_template.format(
            my_color=color_full,
            opponent_color=opponent_color_full,
            fen_description=fen_description,
            guidance_line_tree_last=guidance_line_tree_last,
            expand_node_id=expand_node_id,
            expand_node_desc=expand_node_desc,
            moves=moves_text,
        )
        if extra_instructions:
            prompt += f"\n\n{extra_instructions}"
        return prompt

    # ============================================================
    # Core 2: Call API
    def _call_llm_api(self, prompt: str) -> str:
        """调用 LLM 接口并更新 usage；异常时返回空字符串。"""
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
        if not j:
            return ""

        content = j.get("choices", [])[0].get("message", {}).get("content", "").strip()
        try:
            self.last_usage = normalize_usage(j.get("usage", {}))
            self.total_tokens = add_total_tokens(self.total_tokens, self.last_usage)
        except Exception:
            self.last_usage = {}

        self.logger.info("[SlowThinkModule] LLM raw response (JSON): %s", j)
        return content

    # ============================================================
    # Core 3/4: Return Output (Main Entry)
    # ============================================================
    
    def get_guidance(
        self,
        state,
        player: str,
        guidance_line_tree_last: str = "",
        expand_node_id: str = "",
        expand_node_desc: str = "",
        legal_moves: list | None = None,
        extra_instructions: str = "",
    ) -> dict:
        """主入口：生成并解析慢思考结果，返回树更新字典。"""
        try:
            # 1) Construct prompt
            prompt = self.build_prompt(
                state=state,
                player=player,
                guidance_line_tree_last=guidance_line_tree_last,
                expand_node_id=expand_node_id,
                expand_node_desc=expand_node_desc,
                legal_moves=legal_moves,
                extra_instructions=extra_instructions,
            )

            if self.game_type == "maze":
                label = "Red Red" if player == "red" else "Blue Red" if player == "blue" else player
            else:
                label = "White" if player == "w" else "Black" if player == "b" else player
            self.logger.info("[LLM_AGENT][%s][model=%s] Prompt sent to LLM:\n%s", label, self.model_name, prompt)

            # 2) Call API
            text = self._call_llm_api(prompt)
            if not text:
                self.logger.error("[LLMAgent] No LLM response")
                return {"tree": "", "fix_nodes": {}}

            # 3) Parse output
            # 4) Return output
            return self._parser(text)
        except Exception as e:
            self.logger.error(f"[SLOW][{self._log_tag}][model={self.model_name}] Failed to get guidance: {e}")
            return {"tree": "", "fix_nodes": {}}



