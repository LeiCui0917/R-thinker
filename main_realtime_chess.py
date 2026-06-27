
"""
Concise Real-time Chess Main Controller
Focuses on three core functions:
1. Call specific models based on models and parameters sent from the frontend
2. Get enhanced FEN and available moves in the current environment from the Realtime Chess class
3. Submit agent actions to the environment class to update the state
"""

import os
import sys
import time
import threading
import logging
from Agent.utils.log_naming import make_log_seed
from Agent.available_agents import CHESS_AGENT_NAMES
from main_realtime_common import load_config, load_agent_config, RealtimeControllerBase

# Path Configuration
BASE_DIR = os.path.dirname(__file__)
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "Agent"),
    os.path.join(BASE_DIR, "Env", "ChessEnv")
])

from Env.ChessEnv.RealtimeChessEnv import RealtimeChessEnv

# Global Configuration Instance
_config = load_config(BASE_DIR)
_agent_config = load_agent_config(BASE_DIR)

# ============================================================
#  1. Agent Manager
# ============================================================
class AgentManager:
    """Load agent based on model name"""

    def _agent_instances_cfg(self, llm_settings: dict) -> dict:
        cfg = (_agent_config or {}).get('agent_llm_instances')
        if not isinstance(cfg, dict):
            raise ValueError('agent_config.agent_llm_instances must be a dict')
        return cfg

    def _require_non_empty(self, value: str, *, field: str) -> str:
        out = str(value or '').strip()
        if not out:
            raise ValueError(f'Missing required model instance for {field}')
        return out

    def _agent_model_name(self, model_type: str, parts, llm_settings: dict) -> str:
        if len(parts) > 1 and parts[1]:
            return self._require_non_empty(parts[1], field=f'{model_type} override')

        agent_models = self._agent_instances_cfg(llm_settings)
        entry = agent_models.get(model_type)
        if isinstance(entry, str) and entry.strip():
            return self._require_non_empty(entry, field=f'agent_llm_instances.{model_type}')
        if isinstance(entry, dict):
            model = str(entry.get('model', '') or '').strip()
            return self._require_non_empty(model, field=f'agent_llm_instances.{model_type}.model')
        raise ValueError(f'Missing model instance config for {model_type} in agent_config.agent_llm_instances')

    def _agent_model_split(self, model_type: str, parts, llm_settings: dict) -> tuple[str, str, str]:
        if len(parts) > 1 and parts[1]:
            shared = self._require_non_empty(parts[1], field=f'{model_type} override')
            return shared, shared, shared

        agent_models = self._agent_instances_cfg(llm_settings)
        entry = agent_models.get(model_type)
        if isinstance(entry, dict):
            decision = str(entry.get('decision', '') or '').strip()
            reflection = str(entry.get('reflection', '') or '').strip()
            if not decision or not reflection:
                raise ValueError(
                    f'Missing model instance config for {model_type}; '
                    f'require both .decision and .reflection in agent_config.agent_llm_instances'
                )
            return decision, decision, reflection
        raise ValueError(
            f'Invalid model instance config for {model_type}; '
            f'require dict with .decision and .reflection in agent_config.agent_llm_instances'
        )

    def _delay_seconds(self, parts) -> float:
        if len(parts) > 1 and parts[1]:
            return max(0.0, float(parts[1]))

        delay_cfg = (_agent_config.get('agent_delay_defaults', {}) or {}).get('chess', {}) or {}
        model_type = str(parts[0] if parts else '').strip()
        if model_type == "Random_Agent":
            return max(0.0, float(delay_cfg.get('Random_Agent', 0.0) or 0.0))
        if model_type == "Rule_Agent":
            return max(0.0, float(delay_cfg.get('Rule_Agent', 0.0) or 0.0))
        return 0.0
    
    # ------------------------------
    # Load RuleAgent part in AgentManager class
    # ------------------------------
    def load_agent(
        self,
        model_name: str,
        env=None,
        log_file: str | None = None,
        role: str | None = None,
        move_event=None,
        stop_event=None,
    ):
        parts = model_name.split(':', 1)
        model_type = parts[0]  # First part is always model type
        role_name = (role or 'white').strip().lower()
        api_setting = _config.get('api_settings', {})
        llm_settings = _config.get('llm_settings', {})

        if model_type == "Random_Agent":
            from Agent.rule_agents.random_agent_chess import RandomAgent
            return RandomAgent(delay_s=self._delay_seconds(parts))
        elif model_type == "Human_Agent":
            from Agent.human_agents.human_agent_chess import HumanAgent
            return HumanAgent(move_event=move_event, stop_event=stop_event)
        elif model_type == "Rule_Agent":
            from Agent.rule_agents.rule1_agent_chess import RuleAgent
            return RuleAgent(env=env, delay_s=self._delay_seconds(parts))  # Pass env
        elif model_type == "Think_Agent":
            from Agent.R_Thinker.think_agent import ThinkAgent
            think_model_name = self._agent_model_name(model_type, parts, llm_settings)
            fast_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_fast_think_module_prompt.txt")
            slow_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_slow_think_module_prompt.txt")
            with open(fast_path, 'r', encoding='utf-8') as f:
                fast_prompt = f.read()
            with open(slow_path, 'r', encoding='utf-8') as f:
                slow_prompt = f.read()
            return ThinkAgent(
                api_setting=api_setting,
                llm_settings=llm_settings,
                fast_module_prompt=fast_prompt,
                slow_module_prompt=slow_prompt,
                log_file=log_file,
                model_name=think_model_name,
                game_type='chess',
                slow_module_opponent_prompt=slow_prompt,
                side=role_name,
            )
        elif model_type == "FastOnly_Agent":
            from Agent.R_thinker_variants.fast_only_agent import FastOnlyChess
            fast_only_model_name = self._agent_model_name(model_type, parts, llm_settings)
            fast_prompt_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_fast_think_module_prompt.txt")
            with open(fast_prompt_path, 'r', encoding='utf-8') as f:
                fast_prompt = f.read()
            return FastOnlyChess(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=fast_only_model_name,
                base_prompt=fast_prompt,
                log_file=log_file,
                side=role_name,
            )
        elif model_type == "SlowOnly_Agent":
            from Agent.R_thinker_variants.slow_only_agent import SlowOnlyChess
            slow_only_model_name = self._agent_model_name(model_type, parts, llm_settings)
            slow_prompt_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_slow_think_module_prompt.txt")
            with open(slow_prompt_path, 'r', encoding='utf-8') as f:
                slow_prompt = f.read()
            return SlowOnlyChess(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=slow_only_model_name,
                base_prompt=slow_prompt,
                log_file=log_file,
                side=role_name,
            )
        elif model_type == "Think_WithoutOpponent_Agent":
            from Agent.R_thinker_variants.think_agent_ablation_variants import ThinkAgentWithoutOpponentChess
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return ThinkAgentWithoutOpponentChess(api_setting=api_setting, llm_settings=llm_settings, log_file=log_file, model_name=model_name, side=role_name)
        elif model_type == "Think_WithoutSelf_Agent":
            from Agent.R_thinker_variants.think_agent_ablation_variants import ThinkAgentWithoutSelfChess
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return ThinkAgentWithoutSelfChess(api_setting=api_setting, llm_settings=llm_settings, log_file=log_file, model_name=model_name, side=role_name)
        elif model_type == "LLM_Agent":
            from Agent.llm_base_agents.llm_agent import LLMAgent
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            LLM_module_prompt_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_llm_agent_prompt.txt")
            with open(LLM_module_prompt_path, 'r', encoding='utf-8') as f:
                LLM_module_prompt = f.read()
            return LLMAgent(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=model_name,
                LLM_module_prompt=LLM_module_prompt,
                log_file=log_file,
                game='chess',
                side=role_name,
            )
        elif model_type == "CoT_Agent":
            from Agent.llm_frame_agents.cot_agent import CoTChess
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return CoTChess(
                api_setting=api_setting,
                llm_settings=llm_settings,
                log_file=log_file,
                model_name=model_name,
                side=role_name,
            )
        elif model_type == "Reflexion_Agent":
            from Agent.llm_frame_agents.reflexion_agent import ReflexionDualChess
            shared_model, decision_model, reflection_model = self._agent_model_split(model_type, parts, llm_settings)
            return ReflexionDualChess(
                api_setting=api_setting,
                llm_settings=llm_settings,
                log_file=log_file,
                side=role_name,
                model_name=shared_model,
                decision_model_name=decision_model,
                reflection_model_name=reflection_model,
            )
        elif model_type == "MemoryLLM_Agent":
            from Agent.llm_frame_agents.memory_llm_agent import MemoryLLMChess
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            prompt_path = os.path.join(BASE_DIR, "Agent", "prompt", "Chess_memory_llm_agent_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            return MemoryLLMChess(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=model_name,
                base_prompt=prompt,
                log_file=log_file,
                side=role_name,
            )
        elif model_type == "CodingPairs_Agent":
            from Agent.llm_frame_agents.coding_agent import CodingChessPairs
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return CodingChessPairs(
                api_setting=api_setting,
                llm_settings=llm_settings,
                log_file=log_file,
                model_name=model_name,
                side=role_name,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Available: {', '.join(CHESS_AGENT_NAMES)}")

# ============================================================
#  2. Real-time Chess Controller
# ============================================================
class RealtimeChessController(RealtimeControllerBase):
    """
    Concise Real-time Chess Controller
    Implements three core functions:
    1. Model Management: Load agents based on frontend parameters
    2. Environment Observation: Get enhanced FEN and available moves from Realtime Chess class
    3. Action Submission: Submit agent actions to the environment class
    """

    def __init__(self, white_model: str, black_model: str, push_callback=None):
        super().__init__(push_callback=push_callback)
        self.white_model_name = white_model
        self.black_model_name = black_model
        self.agent_manager = AgentManager()
        self.env = None
        self.white_agent = None
        self.black_agent = None
        self.agent_stats = {
            'w': {
                'valid_decisions': 0,
                'submitted_decisions': 0,
                'failed_submissions': 0,
                'last_action_source': '',
                'last_total_tokens': 0,
                'step_decision_times': [],
                'step_tokens': [],
            },
            'b': {
                'valid_decisions': 0,
                'submitted_decisions': 0,
                'failed_submissions': 0,
                'last_action_source': '',
                'last_total_tokens': 0,
                'step_decision_times': [],
                'step_tokens': [],
            },
        }
        # New: Action event, used to wait briefly when there is no action, and wake up immediately when there is a new action
        self.move_event = threading.Event()
        # Pacing (seconds): align with experiment runners.
        # chess_cooldown: per-piece cooldown duration.
        gs = _config['game_settings']
        self.cooldown = float(gs['chess_cooldown'])

        # Per-game log files (per-agent only; no separate controller log)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_white = self.white_model_name.replace(':', '-').replace('/', '-').replace('\\', '-')
        safe_black = self.black_model_name.replace(':', '-').replace('/', '-').replace('\\', '-')
        logs_dir = os.path.abspath(os.path.join(BASE_DIR, 'Agent', 'logs'))
        # Per-agent log seeds (final normalized filenames are generated by agent modules).
        self.white_log_file = make_log_seed(
            logs_dir,
            run=ts,
            game='chess',
            side='white',
            actor=safe_white,
            opponent=safe_black,
        )
        self.black_log_file = make_log_seed(
            logs_dir,
            run=ts,
            game='chess',
            side='black',
            actor=safe_black,
            opponent=safe_white,
        )
        # Keep a controller logger without a file handler to avoid extra logs
        self.logger = logging.getLogger(f"ChessController_{ts}")
        self.logger.setLevel(logging.INFO)
        # No file handler attached; print concise init message to console
        print(f"Game initialized: {self.white_model_name} (White) vs {self.black_model_name} (Black)")

    def _collect_tokens(self, color: str, agent) -> int:
        stats = self.agent_stats[color]
        try:
            current_total = int(getattr(agent, 'total_tokens', 0) or 0)
        except Exception:
            current_total = 0
        prev_total = int(stats.get('last_total_tokens', 0) or 0)
        if current_total < prev_total:
            delta = current_total
        else:
            delta = current_total - prev_total
        stats['last_total_tokens'] = current_total
        return int(delta)

    def _module_tokens(self, agent) -> dict:
        candidate = getattr(agent, 'inner', None) or agent
        fast = getattr(candidate, 'fast_module', None)
        slow_self = getattr(candidate, 'slow_module', None)
        if slow_self is None:
            slow_self = getattr(candidate, 'slow_module_self', None)
        slow_opp = getattr(candidate, 'slow_module_opponent', None)
        if slow_opp is None:
            slow_opp = getattr(candidate, 'slow_module_opp', None)

        def _tok(obj):
            try:
                return int(getattr(obj, 'total_tokens', 0) or 0)
            except Exception:
                return 0

        return {
            'tokens_fast': _tok(fast),
            'tokens_slow_self': _tok(slow_self),
            'tokens_slow_opp': _tok(slow_opp),
        }

    def _action_source(self, agent) -> str:
        for candidate in (agent, getattr(agent, 'inner', None), getattr(agent, '_fallback', None)):
            src = str(getattr(candidate, 'last_action_source', '') or '').strip()
            if src:
                return src
        return "unknown"

    def _stats_payload(self, color: str, agent) -> dict:
        stats = self.agent_stats[color]
        step_decision_times = list(stats.get('step_decision_times', []) or [])
        step_tokens = list(stats.get('step_tokens', []) or [])
        decisions = len(step_decision_times)
        token_total = int(sum(step_tokens)) if step_tokens else 0
        time_sum = float(sum(step_decision_times)) if step_decision_times else 0.0
        payload = {
            'avg_decision_time': (time_sum / decisions) if decisions > 0 else 0.0,
            'decisions': decisions,
            'valid_decisions': int(stats.get('valid_decisions', 0) or 0),
            'submitted_decisions': int(stats.get('submitted_decisions', 0) or 0),
            'failed_submissions': int(stats.get('failed_submissions', 0) or 0),
            'last_action_source': str(stats.get('last_action_source', '') or ''),
            'tokens': token_total,
            'step_decision_times': step_decision_times,
            'step_tokens': step_tokens,
        }
        payload.update(self._module_tokens(agent))
        return payload

    def _notify_opponent_reflexion(self, actor_color: str, state_before: str, move) -> None:
        target = self.black_agent if actor_color == 'w' else self.white_agent
        if target is None or not hasattr(target, "observe_external_step"):
            return
        try:
            target.observe_external_step(actor_color, state_before, move)
        except Exception:
            pass


    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def initialize(self):
        # Env reads chess_cooldown from config.json (config-only)
        self.env = RealtimeChessEnv()
        # Pass env when loading agents (Fix: Add env=self.env parameter)
        # Use per-agent log files so each agent writes to its own file
        self.white_agent = self.agent_manager.load_agent(
            self.white_model_name,
            env=self.env,
            log_file=self.white_log_file,
            role='white',
            move_event=self.move_event,
            stop_event=self.events["end"],
        )
        self.black_agent = self.agent_manager.load_agent(
            self.black_model_name,
            env=self.env,
            log_file=self.black_log_file,
            role='black',
            move_event=self.move_event,
            stop_event=self.events["end"],
        )
        print(f"Controller initialized successfully: {self.white_agent} (White) vs {self.black_agent} (Black)")
        return True

    # ------------------------------------------------------------
    # Single Agent Loop
    # ------------------------------------------------------------
    def agent_decision_loop(self, color: str, agent):
        """
        Agent continuously gets info from environment, makes decisions and submits actions
        Args:
            color (str): 'w' or 'b'
            agent: Agent instance
        """
        env = self.env
        while self.is_running and not self.events["end"].is_set():
            if self.events["pause"].is_set():
                time.sleep(0.5)
                continue
            
            try:
                # === Original Logic: Get state, decide, submit action ===
                enhanced_FEN_full = env.get_enhancedFENfull()
                legal_moves = env.get_legal_moves_for_color(color)

                # If no legal moves are currently available (often due to cooldown saturation),
                # do not spam agent calls.
                if not legal_moves:
                    time.sleep(0.05)
                    continue
                # === 2. Decide Action ===
                t0 = time.time()
                move = agent.get_action(enhanced_FEN_full, color, legal_moves)
                decision_dt = time.time() - t0
                step_token = self._collect_tokens(color, agent)
                self.agent_stats[color]['step_decision_times'].append(float(decision_dt or 0.0))
                self.agent_stats[color]['step_tokens'].append(int(step_token))
                # If the game ended while the agent was thinking, do not execute/print anything.
                if self.events["end"].is_set():
                    break
                action_source = self._action_source(agent)
                if move is not None:
                    self.agent_stats[color]['last_action_source'] = action_source
                is_valid_decision = move in legal_moves
                if is_valid_decision:
                    self.agent_stats[color]['valid_decisions'] += 1
                decisions = len(self.agent_stats[color]['step_decision_times'])
                print(
                    f"📊 {color} step | dt={decision_dt:.4f}s | decisions={decisions} "
                    f"| valid={self.agent_stats[color]['valid_decisions']} "
                    f"| submitted={self.agent_stats[color]['submitted_decisions']} "
                    f"| source={action_source} "
                    f"| step_tokens={step_token}"
                )
                if not is_valid_decision:
                    print(f"⚠️ {color} returned illegal move: {move}. Ignored")
                    move = None

                else: 
                # === 3. Submit Action ===
                    success = env.process_move_action(move, color)
                    if success:
                        self.agent_stats[color]['submitted_decisions'] += 1
                        self._notify_opponent_reflexion(color, enhanced_FEN_full, move)
                        print(f"✅ {color} successfully submitted action: {move}")
                    else:
                        self.agent_stats[color]['failed_submissions'] += 1
                        print(f"⚠️ {color} failed to submit action: {move}")
                    # Add push logic after getting environment state
                    if success and self.push_callback:
                        game_status = env.check_game_status()
                        game_over = game_status.get("white", {}).get("game_over") or game_status.get("black", {}).get("game_over")
                        fen_full = env.get_enhancedFENfull()

                        self.push_callback({
                            'fen': fen_full,
                            'cooldown_duration': env.cooldown,
                            'game_over': game_over
                        })
                        # Strategy: Stop on forced termination or draw (declarable)
                        if game_over:
                            reason = env.get_termination_reason(game_status)
                            w_steps = game_status.get('white_steps', 0)
                            b_steps = game_status.get('black_steps', 0)
                            
                            self.push_callback({
                                'fen': fen_full,
                                'cooldown_duration': env.cooldown,
                                'game_over': game_over,
                                'termination_reason': reason,
                                'white_steps': w_steps,
                                'black_steps': b_steps,
                                'white_stats': self._stats_payload('w', self.white_agent),
                                'black_stats': self._stats_payload('b', self.black_agent),
                            })
                            
                            print(f"🛑 Game Over or 🤝 Draw: {reason} (White: {w_steps} moves, Black: {b_steps} moves)")
                            self.stop_game()
                            break

                time.sleep(0.01)

            except Exception as e:
                print(f"❌ {color} decision error: {e}")
                time.sleep(0.05)


    # ------------------------------------------------------------
    # Control Flow
    # ------------------------------------------------------------
    def _on_started(self):
        print(f"🚀 Game Started: {self.white_model_name} (White) vs {self.black_model_name} (Black)")

    def _thread_specs(self):
        return {
            "white": (self.agent_decision_loop, ("w", self.white_agent)),
            "black": (self.agent_decision_loop, ("b", self.black_agent)),
        }

    def _on_paused(self):
        self.env and self.env.pause()
        print("⏸️ Game Paused")

    def _on_resumed(self):
        self.env and self.env.resume()
        print("▶️ Game Resumed")

    def _on_stopped(self):
        print("🛑 Game Ended")

    
