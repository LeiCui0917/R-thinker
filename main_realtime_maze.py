
"""
Concise Real-time Maze Main Controller
Focuses on three core functions:
1. Call specific models based on models and parameters sent from the frontend
2. Get frame state and available moves in the current environment from the Realtime Maze class
3. Submit agent actions to the environment class to update the state
"""

import os
import sys
import time
import threading
import logging
from Agent.utils.log_naming import make_log_seed
from Agent.available_agents import MAZE_AGENT_NAMES
from main_realtime_common import load_config, load_agent_config, RealtimeControllerBase

# Path Configuration
BASE_DIR = os.path.dirname(__file__)
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "Agent"),
    os.path.join(BASE_DIR, "Env", "MazeEnv")
])

from Env.MazeEnv.RealtimeMazeEnv import RealtimeMazeEnv

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

        delay_cfg = (_agent_config.get('agent_delay_defaults', {}) or {}).get('maze', {}) or {}
        model_type = str(parts[0] if parts else '').strip()
        if model_type == "Random_Agent":
            return max(0.0, float(delay_cfg.get('Random_Agent', 0.0) or 0.0))
        if model_type == "Rule_Agent":
            return max(0.0, float(delay_cfg.get('Rule_Agent', 0.0) or 0.0))
        return 0.0
    
    def load_agent(
        self,
        model_name: str,
        env=None,
        log_file: str | None = None,
        role: str | None = None,
        move_event=None,
        stop_event=None,
    ):
        """
        Load Agent
        :param model_name: Chess-style string, e.g. "Rule_Agent" or "LLM_Agent:gpt-4"
        :param env: Environment instance
        :param log_file: Log file path
        """
        parts = model_name.split(':', 1)
        model_type = parts[0]
        role_name = (role or 'red').strip().lower()
        api_setting = _config.get('api_settings', {})
        llm_settings = _config.get('llm_settings', {})
        if model_type == "Random_Agent":
            from Agent.rule_agents.random_agent_maze import RandomAgent
            return RandomAgent(delay_s=self._delay_seconds(parts))
        elif model_type == "Rule_Agent":
            from Agent.rule_agents.rule1_agent_maze import RuleAgent
            return RuleAgent(env=env, delay_s=self._delay_seconds(parts))
        elif model_type == "Human_Agent":
            from Agent.human_agents.human_agent_maze import HumanAgent
            return HumanAgent(move_event=move_event, stop_event=stop_event)
        elif model_type == "Think_Agent":
            from Agent.R_Thinker.think_agent import ThinkAgent
            think_model_name = self._agent_model_name(model_type, parts, llm_settings)
            fast_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_fast_think_module_prompt.txt')
            slow_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_slow_think_module_prompt.txt')
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
                game_type='maze',
                slow_module_opponent_prompt=slow_prompt,
                side=role_name,
            )
        elif model_type == "FastOnly_Agent":
            from Agent.R_thinker_variants.fast_only_agent import FastOnlyMaze
            fast_only_model_name = self._agent_model_name(model_type, parts, llm_settings)
            fast_prompt_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_fast_think_module_prompt.txt')
            with open(fast_prompt_path, 'r', encoding='utf-8') as f:
                fast_prompt = f.read()
            return FastOnlyMaze(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=fast_only_model_name,
                base_prompt=fast_prompt,
                log_file=log_file,
                role=role_name,
            )
        elif model_type == "SlowOnly_Agent":
            from Agent.R_thinker_variants.slow_only_agent import SlowOnlyMaze
            slow_only_model_name = self._agent_model_name(model_type, parts, llm_settings)
            slow_prompt_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_slow_think_module_prompt.txt')
            with open(slow_prompt_path, 'r', encoding='utf-8') as f:
                slow_prompt = f.read()
            return SlowOnlyMaze(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=slow_only_model_name,
                base_prompt=slow_prompt,
                log_file=log_file,
                role=role_name,
            )
        elif model_type == "Think_WithoutOpponent_Agent":
            from Agent.R_thinker_variants.think_agent_ablation_variants import ThinkAgentWithoutOpponentMaze
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return ThinkAgentWithoutOpponentMaze(api_setting=api_setting, llm_settings=llm_settings, role=role_name, log_file=log_file, model_name=model_name)
        elif model_type == "Think_WithoutSelf_Agent":
            from Agent.R_thinker_variants.think_agent_ablation_variants import ThinkAgentWithoutSelfMaze
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return ThinkAgentWithoutSelfMaze(api_setting=api_setting, llm_settings=llm_settings, role=role_name, log_file=log_file, model_name=model_name)
        elif model_type == "LLM_Agent":
            from Agent.llm_base_agents.llm_agent import LLMAgent
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            prompt_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_llm_agent_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                llm_prompt = f.read()

            return LLMAgent(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=model_name,
                LLM_module_prompt=llm_prompt,
                log_file=log_file,
                game='maze',
                side=role_name,
            )
        elif model_type == "CoT_Agent":
            from Agent.llm_frame_agents.cot_agent import CoTMaze
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return CoTMaze(api_setting=api_setting, llm_settings=llm_settings, role=role_name, log_file=log_file, model_name=model_name)
        elif model_type == "Reflexion_Agent":
            from Agent.llm_frame_agents.reflexion_agent import ReflexionDualMaze
            shared_model, decision_model, reflection_model = self._agent_model_split(model_type, parts, llm_settings)
            return ReflexionDualMaze(
                api_setting=api_setting,
                llm_settings=llm_settings,
                role=role_name,
                log_file=log_file,
                model_name=shared_model,
                decision_model_name=decision_model,
                reflection_model_name=reflection_model,
            )
        elif model_type == "MemoryLLM_Agent":
            from Agent.llm_frame_agents.memory_llm_agent import MemoryLLMMaze
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            prompt_path = os.path.join(BASE_DIR, 'Agent', 'prompt', 'Maze_memory_llm_agent_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            return MemoryLLMMaze(
                api_setting=api_setting,
                llm_settings=llm_settings,
                model_name=model_name,
                base_prompt=prompt,
                log_file=log_file,
                role=role_name,
            )
        elif model_type == "CodingPairs_Agent":
            from Agent.llm_frame_agents.coding_agent import CodingMazePairs
            model_name = self._agent_model_name(model_type, parts, llm_settings)
            return CodingMazePairs(api_setting=api_setting, llm_settings=llm_settings, role=role_name, log_file=log_file, model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Available: {', '.join(MAZE_AGENT_NAMES)}")

# ============================================================
#  2. Real-time Maze Controller
# ============================================================
class RealtimeMazeController(RealtimeControllerBase):
    """
    Concise Real-time Maze Controller
    Implements three core functions:
    1. Model Management: Load agents based on frontend parameters
    2. Environment Observation: Get frame state and available moves from Realtime Maze class
    3. Action Submission: Submit agent actions to the environment class
    """

    def __init__(self, red_model: str, blue_model: str, push_callback=None):
        """
        :param red_model: Chess-style model_name string, e.g. "LLM_Agent:gpt-4" or "Rule_Agent"
        :param blue_model: Chess-style model_name string, e.g. "LLM_Agent:gpt-4" or "Rule_Agent"
        :param push_callback: Callback function for pushing state
        """
        super().__init__(push_callback=push_callback)
        self.red_model_name = red_model
        self.blue_model_name = blue_model
        self.role_model_names = {
            "red": self.red_model_name,
            "blue": self.blue_model_name,
        }
        
        self.agent_manager = AgentManager()
        self.env = None
        self.red_agent = None
        self.blue_agent = None
        self.agent_stats = {
            'red': {
                'valid_decisions': 0,
                'submitted_decisions': 0,
                'failed_submissions': 0,
                'last_action_source': '',
                'last_total_tokens': 0,
                'step_decision_times': [],
                'step_tokens': [],
            },
            'blue': {
                'valid_decisions': 0,
                'submitted_decisions': 0,
                'failed_submissions': 0,
                'last_action_source': '',
                'last_total_tokens': 0,
                'step_decision_times': [],
                'step_tokens': [],
            },
        }
        self.move_event = threading.Event()

        self._stop_lock = threading.Lock()
        # Shared lock to protect env read/write across role threads
        self.env_lock = threading.Lock()
        
        game_settings = _config['game_settings']

        # Log files (per-agent only)
        ts = time.strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.abspath(os.path.join(BASE_DIR, 'Agent', 'logs'))
        self.role_log_files = {}
        for role, model_name in self.role_model_names.items():
            safe_name = (model_name or 'unknown').replace(':', '-').replace('/', '-').replace('\\', '-')
            opp_role = 'blue' if role == 'red' else 'red'
            opp_name = self.role_model_names.get(opp_role, 'unknown')
            safe_opp_name = (opp_name or 'unknown').replace(':', '-').replace('/', '-').replace('\\', '-')
            self.role_log_files[role] = make_log_seed(
                logs_dir,
                run=ts,
                game='maze',
                side=role,
                actor=safe_name,
                opponent=safe_opp_name,
            )

        # Back-compat attributes
        self.red_log_file = self.role_log_files["red"]
        self.blue_log_file = self.role_log_files["blue"]
        self.logger = logging.getLogger(f"MazeController_{ts}")
        self.logger.setLevel(logging.INFO)

        # Note: We intentionally do not create a controller summary log file.
        # All detailed logs are written into per-agent log files.

    def _collect_tokens(self, role: str, agent) -> int:
        stats = self.agent_stats[role]
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

    def _stats_payload(self, role: str, agent) -> dict:
        stats = self.agent_stats[role]
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

    def _notify_opponent_reflexion(self, actor_role: str, state_before, move) -> None:
        target = self.blue_agent if actor_role == 'red' else self.red_agent
        if target is None or not hasattr(target, "observe_external_step"):
            return
        try:
            target.observe_external_step(actor_role, state_before, move)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def initialize(self):
        # Env reads maze settings from config.json (config-only)
        game_settings = _config['game_settings']
        maze_size = int(game_settings['maze_logical_size'])
        loop_probability = float(game_settings['maze_loop_probability'])
        step_duration = float(game_settings['maze_step_duration'])
        trail_release_seconds = float(game_settings['maze_trail_release_seconds'])
        self.env = RealtimeMazeEnv(
            maze_size=maze_size,
            loop_probability=loop_probability,
            step_duration=step_duration,
            trail_release_seconds=trail_release_seconds,
            seed=None,
        )

        # Pass env when loading agents (Fix: Add env=self.env parameter)
        # Use per-agent log files so each agent writes to its own file
        self.red_agent = self.agent_manager.load_agent(
            self.red_model_name,
            env=self.env,
            log_file=self.red_log_file,
            role='red',
            move_event=self.move_event,
            stop_event=self.events["end"],
        )
        self.blue_agent = self.agent_manager.load_agent(
            self.blue_model_name,
            env=self.env,
            log_file=self.blue_log_file,
            role='blue',
            move_event=self.move_event,
            stop_event=self.events["end"],
        )

        print(f"Maze Controller initialized successfully: Red={self.red_agent}, Blue={self.blue_agent}")
        return True

    # ------------------------------------------------------------
    # Agent Loop (Thread-based)
    # ------------------------------------------------------------
    def agent_decision_loop(self, role: str, agent):
        """
        Agent continuously gets information from environment, makes decisions and submits actions
        Args:
            role (str): 'red' or 'blue'
            agent: Agent instance
        """
        env = self.env

        while self.is_running and not self.events["end"].is_set():
            if self.events["pause"].is_set():
                time.sleep(0.5)
                continue

            try:
                # === Original Logic: Get state, decide, submit action ===
                current_state = env.frame_state()
                legal_moves = env.get_legal_actions(role)

                # If no legal moves are currently available,
                # do not spam agent calls.
                if not legal_moves:
                    time.sleep(0.05)
                    continue

                # === 2. Decide Action ===
                t0 = time.time()
                move = agent.get_action(current_state, role, legal_moves)
                decision_dt = time.time() - t0
                step_token = self._collect_tokens(role, agent)
                self.agent_stats[role]['step_decision_times'].append(float(decision_dt or 0.0))
                self.agent_stats[role]['step_tokens'].append(int(step_token))

                # If the game ended while the agent was thinking, do not execute/print anything.
                if self.events["end"].is_set():
                    break

                action_source = self._action_source(agent)
                if move is not None:
                    self.agent_stats[role]['last_action_source'] = action_source
                is_valid_decision = move in legal_moves
                if is_valid_decision:
                    self.agent_stats[role]['valid_decisions'] += 1
                decisions = len(self.agent_stats[role]['step_decision_times'])
                print(
                    f"📊 {role} step | dt={decision_dt:.4f}s | decisions={decisions} "
                    f"| valid={self.agent_stats[role]['valid_decisions']} "
                    f"| submitted={self.agent_stats[role]['submitted_decisions']} "
                    f"| source={action_source} "
                    f"| step_tokens={step_token}"
                )
                if not is_valid_decision:
                    print(f"⚠️ {role} returned illegal target: {move}. Ignored")
                    move = None

                else:
                    # === 3. Submit Action ===
                    success = env.apply_action(role=role, move=move)
                    if success:
                        self.agent_stats[role]['submitted_decisions'] += 1
                        self._notify_opponent_reflexion(role, current_state, move)
                        print(f"✅ {role} successfully submitted action: {move}")
                    else:
                        self.agent_stats[role]['failed_submissions'] += 1
                        print(f"⚠️ {role} failed to submit action: {move}")
                    # Add push logic after getting environment state
                    if success and self.push_callback:
                        game_status = env.check_game_status()
                        game_over = bool(game_status.get("game_over", False))
                        frame = env.frame_state()
                        ascii_now = env.render_ascii()

                        self.push_callback({
                            'ascii': ascii_now,
                            'frame': frame,
                            'game_over': game_over,
                        })

                        # Strategy: Stop on forced termination.
                        if game_over:
                            reason = game_status.get('reason')
                            red_steps = game_status.get('red_steps', 0)
                            blue_steps = game_status.get('blue_steps', 0)

                            self.push_callback({
                                'ascii': ascii_now,
                                'frame': frame,
                                'game_over': True,
                                'termination_reason': reason,
                                'red_steps': red_steps,
                                'blue_steps': blue_steps,
                                'red_stats': self._stats_payload('red', self.red_agent),
                                'blue_stats': self._stats_payload('blue', self.blue_agent),
                            })
                            print(
                                f"🏁 Maze Game Over: {reason} "
                                f"(Red: {red_steps} steps, Blue: {blue_steps} steps)"
                            )
                            self.stop_game()
                            break

                time.sleep(0.01)

            except Exception as e:
                print(f"❌ {role} decision error: {e}")
                time.sleep(0.05)
                

    # ------------------------------------------------------------
    # Control Flow
    # ------------------------------------------------------------
    def _on_started(self):
        print(f"🚀 Maze Game Started")

        # Push initial frame
        try:
            if self.push_callback:
                self.push_callback({
                    'ascii': self.env.render_ascii(), 
                    'frame': self.env.frame_state(),
                    'game_over': False
                })
        except Exception:
            pass

    def _thread_specs(self):
        return {
            "red": (self.agent_decision_loop, ("red", self.red_agent)),
            "blue": (self.agent_decision_loop, ("blue", self.blue_agent)),
        }

    def _on_paused(self):
        self.env and self.env.pause()
        print("⏸️ Maze Paused")

    def _on_resumed(self):
        self.env and self.env.resume()
        print("▶️ Maze Resumed")

    def _on_stopped(self):
        print("🛑 Maze Ended")
