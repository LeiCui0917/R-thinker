"""基础 LLM 智能体（Maze/Chess 通用）。

职责与流程：
1) 根据环境状态构造提示词（Prompt）。
2) 调用统一 HTTP 接口请求模型。
3) 解析模型输出为可执行动作（Maze 目标点 / Chess UCI）。

说明：
- 该类是多数 frame-agent 的底层执行器。
- 统一维护 last_usage / total_tokens / 请求异常信息。
"""

import logging
import os
import requests

from Agent.utils.llm_http_client import post_json_with_retry
from Agent.utils.log_naming import make_log_file
from Agent.utils.usage_utils import normalize_usage, add_total_tokens
from Env.ChessEnv.chess_output_parser import ChessOutputParser
from Env.ChessEnv.fen_translator import FenTranslator
from Env.MazeEnv.maze_output_parser import MazeOutputParser
from Env.MazeEnv.maze_translator import MazeTranslator


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
        logger.debug("[LLM_AGENT] Logger initialized with file: %s", log_file)
        return fh
    except Exception:
        return None


class LLMAgent:
	def __init__(
		self,
		api_setting,
		llm_settings,
		model_name,
		LLM_module_prompt=None,
		log_file=None,
		game=None,
		side=None,
		log_agent: str = "llm",
		log_module: str | None = None,
		log_label: str | None = None,
	):
		log_file = make_log_file(log_file, game=game, side=side, agent=log_agent, module=log_module) if log_file else log_file
		self.logger = logging.getLogger(f"LLMAgent_{id(self)}")
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
			raise ValueError("LLMAgent requires explicit non-empty model_name")
		self.prompt_template = LLM_module_prompt
		self.log_label = str(log_label or "LLM_AGENT").strip() or "LLM_AGENT"
		self.log_module = str(log_module or "decision").strip() or "decision"
		self.last_action_source = self.log_module

		self._chess_parser = ChessOutputParser()
		self._maze_parser = MazeOutputParser()
		self._chess_translator = FenTranslator()
		self._maze_translator = MazeTranslator()

		self.last_usage = {}
		self.total_tokens = 0
		self.last_request_exception = None
		self.last_request_status_code = None
		self.last_response_text = ""

		self._session_env = requests.Session()
		self._session_env.trust_env = True

	# ---------------- Core 1: Construct Prompt ----------------
	def _build_prompt_chess(self, enhanced_FEN_full: str, legal_moves: list, color: str) -> str:
		"""构造 Chess 提示词：注入颜色、合法走子与 FEN 解释文本。"""
		color_full = "White" if color == "w" else "Black" if color == "b" else color
		opponent_color_full = "Black" if color_full == "White" else "White"
		return (self.prompt_template or "").format(
			my_color=color_full,
			opponent_color=opponent_color_full,
			enhanced_FEN_full=enhanced_FEN_full,
			moves=legal_moves,
			fen_description=self._chess_translator.translate(enhanced_FEN_full),
		)

	def _build_prompt_maze(self, frame: dict, legal_moves: list, role: str) -> str:
		"""构造 Maze 提示词：注入角色、合法目标与迷宫语义描述。"""
		role_full = "Red" if role == "red" else "Blue" if role == "blue" else role
		opponent = "Blue" if role_full == "Red" else "Red" if role_full == "Blue" else "Opponent"
		return (self.prompt_template or "").format(
			player=role_full,
			opponent=opponent,
			frame=frame,
			moves=legal_moves,
			maze_description=self._maze_translator.translate(frame, role=role),
		)

	# ---------------- Core 2: Call LLM ----------------
	def _call_llm_api(self, prompt: str) -> str:
		"""调用模型接口并更新 usage/异常状态，返回文本内容。"""
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

		self.last_request_exception = None
		self.last_request_status_code = None
		self.last_usage = {}
		self.last_response_text = ""

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
			self.logger.exception("[%s] LLM request failed: %s", self.log_label, exc)
		if not j:
			return ""

		self.logger.info("[%s] LLM raw response (JSON): %s", self.log_label, j)
		content = j.get("choices", [])[0].get("message", {}).get("content", "").strip()
		self.last_response_text = content
		try:
			self.last_usage = normalize_usage(j.get("usage", {}))
			self.total_tokens = add_total_tokens(self.total_tokens, self.last_usage)
		except Exception:
			self.last_usage = {}
		return content

	# ---------------- Core 3: Main Decision Logic ----------------
	def _get_action_chess(self, enhanced_FEN_full, color, legal_moves):
		"""Chess 单步决策：构造提示词→调用模型→解析 UCI。"""
		prompt = self._build_prompt_chess(enhanced_FEN_full, legal_moves, color)
		color_full = "White" if color == "w" else "Black" if color == "b" else color
		self.logger.info("[%s][%s][model=%s] Prompt sent to LLM:\n%s", self.log_label, color_full, self.model_name, prompt)

		self.last_action_source = self.log_module
		text = self._call_llm_api(prompt)
		if not text:
			status = getattr(self, "last_request_status_code", None)
			exc = getattr(self, "last_request_exception", None)
			if exc is not None or status is not None:
				self.logger.error("[%s] No LLM response (status=%s, exc=%r)", self.log_label, status, exc)
			else:
				self.logger.error("[%s] No LLM response", self.log_label)
			return None

		last_sentence = self._chess_parser.get_last_sentence(text)
		uci_move = self._chess_parser.find_uci_in_sentence(last_sentence)
		if self._chess_parser.is_valid_uci(uci_move):
			self.logger.info("[%s] Valid move: %s", self.log_label, uci_move)
			return uci_move
		return None

	def _get_action_maze(self, frame: dict, role: str, legal_moves: list):
		"""Maze 单步决策：构造提示词→调用模型→解析目标路口。"""
		prompt = self._build_prompt_maze(frame, legal_moves, role)
		role_full = "Red" if role == "red" else "Blue" if role == "blue" else role
		self.logger.info("[%s][%s][model=%s] Prompt sent to LLM:\n%s", self.log_label, role_full, self.model_name, prompt)

		self.last_action_source = self.log_module
		text = self._call_llm_api(prompt)
		if not text:
			status = getattr(self, "last_request_status_code", None)
			exc = getattr(self, "last_request_exception", None)
			if exc is not None or status is not None:
				self.logger.error("[%s] No LLM response (status=%s, exc=%r)", self.log_label, status, exc)
			else:
				self.logger.error("[%s] No LLM response", self.log_label)
			return None

		last_sentence = self._maze_parser.get_last_sentence(text)
		target = self._maze_parser.find_target_junction_in_sentence(last_sentence)
		if self._maze_parser.is_valid_target_junction(target):
			self.logger.info("[%s] Valid move: %s", self.log_label, target)
			return target
		return None

	def get_action(self, *args, **kwargs):
		"""统一入口：自动识别 Maze/Chess 参数并路由到对应决策函数。"""
		if "enhanced_FEN_full" in kwargs or "color" in kwargs:
			enhanced_FEN_full = kwargs.get("enhanced_FEN_full")
			color = kwargs.get("color")
			legal_moves = kwargs.get("legal_moves")
			return self._get_action_chess(enhanced_FEN_full, color, legal_moves)

		if "frame" in kwargs or "role" in kwargs:
			frame = kwargs.get("frame")
			role = kwargs.get("role")
			legal_moves = kwargs.get("legal_moves")
			return self._get_action_maze(frame, role, legal_moves)

		if len(args) >= 3:
			state = args[0]
			player = args[1]
			legal_moves = args[2]
			if isinstance(state, dict):
				return self._get_action_maze(state, player, legal_moves)
			return self._get_action_chess(state, player, legal_moves)

		return None
