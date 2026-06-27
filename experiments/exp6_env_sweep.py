import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from Agent.utils.log_naming import make_log_seed
from Env.ChessEnv.RealtimeChessEnv import RealtimeChessEnv
from Env.MazeEnv.RealtimeMazeEnv import RealtimeMazeEnv
from Experiments.experiment_core import episode_row, parse_csv_list, run_with_retries, write_experiment_csvs
from main_realtime_chess import AgentManager as ChessAgentManager
from main_realtime_maze import AgentManager as MazeAgentManager

# ============================================================
# 模块 1) 配置与常量
# ============================================================
# 作用：集中管理实验开关、预算、回合上限与默认对抗列表。

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '3'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
TURN_LIMIT = int(os.getenv('THINK_AGENT_TURN_LIMIT', '2000'))

THINK_AGENT_NAME = 'Think_Agent'
DEFAULT_BASELINES = ['Reflexion_Agent', 'CoT_Agent', 'LLM_Agent', 'MemoryLLM_Agent', 'CodingPairs_Agent', 'Rule_Agent']
DEFAULT_BUDGETS = ['0.2', '0.5', '1.0', '2.0']

_BASE_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, '..'))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, 'Agent', 'logs')


# ============================================================
# 模块 2) 预算执行与计量
# ============================================================
# 作用：提供“单次预算调用”能力，以及局内统计采集。

def _start_timed_call(fn, args: tuple):
    # 异步启动一次 get_action 调用。
    t0 = time.time()
    executor = ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(fn, *args)
    return fut, executor, t0


def _poll_timed_call(fut, started_at: float, budget: float):
    # 在预算窗口内等待结果，区分：正常/超时/异常。
    elapsed = max(0.0, time.time() - started_at)
    remaining = max(0.0, float(budget) - elapsed)
    try:
        res = fut.result(timeout=remaining)
        dt = max(0.0, time.time() - started_at)
        return True, res, dt, False
    except TimeoutError:
        return False, None, float(budget), True
    except Exception:
        dt = max(0.0, time.time() - started_at)
        return True, None, dt, False


def _action_source(agent) -> str:
    for candidate in (agent, getattr(agent, 'inner', None), getattr(agent, '_fallback', None)):
        src = str(getattr(candidate, 'last_action_source', '') or '').strip()
        if src:
            return src
    return "unknown"


class _Meter:
    # 单 agent 局内统计容器：时延、有效决策、token 增量、超时次数。
    def __init__(self, agent):
        self.agent = agent
        self.decisions = 0
        self.valid_decisions = 0
        self.submitted_decisions = 0
        self.last_action_source = ''
        self.decision_times: list[float] = []
        self._last_total_tokens = 0
        self.step_tokens: list[int] = []

    def _current_tokens(self) -> int:
        try:
            return int(getattr(self.agent, 'total_tokens', 0) or 0)
        except Exception:
            return 0

    def collect_tokens_delta(self) -> int:
        current = self._current_tokens()
        prev = int(self._last_total_tokens or 0)
        delta = current if current < prev else (current - prev)
        self._last_total_tokens = current
        return int(delta)

    def on_success(self, dt: float, source: str = ""):
        self.decisions += 1
        self.valid_decisions += 1
        self.submitted_decisions += 1
        source_text = str(source or "").strip()
        if source_text:
            self.last_action_source = source_text
        self.decision_times.append(float(dt or 0.0))
        self.step_tokens.append(self.collect_tokens_delta())

    def on_completed_call_no_move(self):
        self.decisions += 1
        self.step_tokens.append(self.collect_tokens_delta())

    def on_timeout(self):
        self.decisions += 1
        self.step_tokens.append(0)

    def avg_dt(self) -> float:
        if not self.decision_times:
            return 0.0
        return float(sum(self.decision_times) / len(self.decision_times))

    def total_tokens(self) -> int:
        return int(sum(self.step_tokens)) if self.step_tokens else 0


# ============================================================
# 模块 3) 单局对局执行（回合制预算）
# ============================================================
# 作用：分别处理 Chess/Maze 单局流程；每回合预算超时即弃权。

def _run_chess_episode_budget_turn(white_name: str, black_name: str, budget_s: float, run_tag: str, episode: int) -> dict:
    env = RealtimeChessEnv()
    env.reset()
    manager = ChessAgentManager()
    log_white = make_log_seed(_LOGS_DIR, run=run_tag, game='chess', side='white', actor=white_name, opponent=black_name, episode=episode)
    log_black = make_log_seed(_LOGS_DIR, run=run_tag, game='chess', side='black', actor=black_name, opponent=white_name, episode=episode)
    white = manager.load_agent(white_name, env=env, log_file=log_white)
    black = manager.load_agent(black_name, env=env, log_file=log_black)

    meters = {'w': _Meter(white), 'b': _Meter(black)}
    agents = {'w': white, 'b': black}
    colors = ['w', 'b']

    try:
        for turn in range(int(TURN_LIMIT)):
            status = env.check_game_status()
            over = bool(status.get('white', {}).get('game_over') or status.get('black', {}).get('game_over'))
            if over:
                break

            color = colors[turn % 2]
            legal = env.get_legal_moves_for_color(color)
            if not legal:
                continue

            enhanced = env.get_enhancedFENfull()
            fut, executor, started_at = _start_timed_call(agents[color].get_action, (enhanced, color, list(legal)))
            done, move, dt, timed_out = _poll_timed_call(fut, started_at, budget_s)
            executor.shutdown(wait=False, cancel_futures=True)

            if timed_out:
                meters[color].on_timeout()
                continue

            if not done:
                continue

            if not isinstance(move, str) or move not in legal:
                meters[color].on_completed_call_no_move()
                continue

            ok = env.process_move_action(move, color)
            if ok:
                meters[color].on_success(dt, _action_source(agents[color]))
            else:
                meters[color].on_completed_call_no_move()

        status = env.check_game_status()
        reason = ''
        try:
            reason = str(env.get_termination_reason(status) or '')
        except Exception:
            reason = ''
        if 'White wins' in reason:
            result = 'white_win'
        elif 'Black wins' in reason:
            result = 'black_win'
        else:
            result = 'draw'
        return {
            'result': result,
            'white_steps': int(status.get('white_steps', 0) or 0),
            'black_steps': int(status.get('black_steps', 0) or 0),
            'white_avg_dt': meters['w'].avg_dt(),
            'black_avg_dt': meters['b'].avg_dt(),
            'white_decisions': int(meters['w'].decisions),
            'black_decisions': int(meters['b'].decisions),
            'white_valid_decisions': int(meters['w'].valid_decisions),
            'black_valid_decisions': int(meters['b'].valid_decisions),
            'white_submitted_decisions': int(meters['w'].submitted_decisions),
            'black_submitted_decisions': int(meters['b'].submitted_decisions),
            'white_last_action_source': str(meters['w'].last_action_source or ''),
            'black_last_action_source': str(meters['b'].last_action_source or ''),
            'white_tokens': meters['w'].total_tokens(),
            'black_tokens': meters['b'].total_tokens(),
        }
    finally:
        for agent in (white, black):
            try:
                if hasattr(agent, 'stop'):
                    agent.stop()
            except Exception:
                pass


def _parse_maze_target(v) -> tuple[int, int] | None:
    # Maze 合法动作统一解析为 (x, y)。
    if isinstance(v, (tuple, list)) and len(v) == 2:
        try:
            return (int(v[0]), int(v[1]))
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('(') and s.endswith(')') and ',' in s:
            left, right = s[1:-1].split(',', 1)
            try:
                return (int(left.strip()), int(right.strip()))
            except Exception:
                return None
    return None


def _run_maze_episode_budget_turn(red_name: str, blue_name: str, budget_s: float, run_tag: str, episode: int) -> dict:
    env = RealtimeMazeEnv(
        maze_size=int(os.getenv('THINK_AGENT_MAZE_SIZE', '15')),
        loop_probability=float(os.getenv('THINK_AGENT_MAZE_LOOP_PROBABILITY', '0.2')),
        step_duration=float(os.getenv('THINK_AGENT_MAZE_STEP_DURATION', '0.05')),
        trail_release_seconds=float(os.getenv('THINK_AGENT_MAZE_TRAIL_RELEASE_SECONDS', '999.0')),
        seed=None,
    )
    env.reset()
    manager = MazeAgentManager()
    log_red = make_log_seed(_LOGS_DIR, run=run_tag, game='maze', side='red', actor=red_name, opponent=blue_name, episode=episode)
    log_blue = make_log_seed(_LOGS_DIR, run=run_tag, game='maze', side='blue', actor=blue_name, opponent=red_name, episode=episode)
    red = manager.load_agent(red_name, env=env, log_file=log_red, role='red')
    blue = manager.load_agent(blue_name, env=env, log_file=log_blue, role='blue')

    meters = {'red': _Meter(red), 'blue': _Meter(blue)}
    agents = {'red': red, 'blue': blue}
    roles = ['red', 'blue']

    try:
        for turn in range(int(TURN_LIMIT)):
            status = env.check_game_status()
            if bool(status.get('game_over', False)):
                break

            role = roles[turn % 2]
            legal = env.get_legal_actions(role)
            if not legal:
                continue

            frame = env.frame_state()
            fut, executor, started_at = _start_timed_call(agents[role].get_action, (frame, role, list(legal)))
            done, move, dt, timed_out = _poll_timed_call(fut, started_at, budget_s)
            executor.shutdown(wait=False, cancel_futures=True)

            if timed_out:
                meters[role].on_timeout()
                continue

            if not done:
                continue
            target = _parse_maze_target(move)
            if target is None:
                meters[role].on_completed_call_no_move()
                continue

            legal_targets = {_parse_maze_target(x) for x in legal}
            if target not in legal_targets:
                meters[role].on_completed_call_no_move()
                continue

            ok = env.apply_action(role=role, move=target)
            if ok:
                meters[role].on_success(dt, _action_source(agents[role]))
            else:
                meters[role].on_completed_call_no_move()

        status = env.check_game_status()
        result = str(status.get('event') or status.get('reason') or 'draw')
        return {
            'result': result,
            'red_steps': int(status.get('red_steps', 0) or 0),
            'blue_steps': int(status.get('blue_steps', 0) or 0),
            'red_avg_dt': meters['red'].avg_dt(),
            'blue_avg_dt': meters['blue'].avg_dt(),
            'red_decisions': int(meters['red'].decisions),
            'blue_decisions': int(meters['blue'].decisions),
            'red_valid_decisions': int(meters['red'].valid_decisions),
            'blue_valid_decisions': int(meters['blue'].valid_decisions),
            'red_submitted_decisions': int(meters['red'].submitted_decisions),
            'blue_submitted_decisions': int(meters['blue'].submitted_decisions),
            'red_last_action_source': str(meters['red'].last_action_source or ''),
            'blue_last_action_source': str(meters['blue'].last_action_source or ''),
            'red_tokens': meters['red'].total_tokens(),
            'blue_tokens': meters['blue'].total_tokens(),
        }
    finally:
        for agent in (red, blue):
            try:
                if hasattr(agent, 'stop'):
                    agent.stop()
            except Exception:
                pass


# ============================================================
# 模块 4) 实验编排与输出
# ============================================================
# 作用：预算×基线×多局运行，汇总并写入 summary/episode CSV。


def _run_budget_matchups(*, game: str, baselines: list[str], budget_s: float, ts: str) -> tuple[list[dict], list[dict]]:
    game_cfg = {
        'maze': {
            'side_a': 'red',
            'side_b': 'blue',
            'win_event_a': 'red_goal',
            'win_event_b': 'blue_goal',
            'runner': _run_maze_episode_budget_turn,
        },
        'chess': {
            'side_a': 'white',
            'side_b': 'black',
            'win_event_a': 'white_win',
            'win_event_b': 'black_win',
            'runner': _run_chess_episode_budget_turn,
        },
    }
    cfg = game_cfg.get(str(game))
    if cfg is None:
        raise ValueError(f'Unknown game: {game}')

    side_a = str(cfg['side_a'])
    side_b = str(cfg['side_b'])
    win_event_a = str(cfg['win_event_a'])
    win_event_b = str(cfg['win_event_b'])
    runner = cfg['runner']

    summary_rows: list[dict] = []
    episode_rows: list[dict] = []

    for baseline in baselines:
        if PRINT_PROGRESS:
            print(f'[exp5][{game}] budget={budget_s:.3f}s | {THINK_AGENT_NAME} vs {baseline}')

        rows: list[dict] = []
        for episode in range(int(EPISODES)):
            ep = episode + 1
            result = run_with_retries(
                lambda _retry: runner(THINK_AGENT_NAME, baseline, budget_s, f'{ts}_{budget_s:.3f}', ep),
                restart_limit=RESTART_LIMIT,
                label=f'exp5-{game}-{baseline}-b{budget_s}-ep{ep}',
                retry_exceptions=(Exception,),
            )
            result_row = episode_row(game, THINK_AGENT_NAME, baseline, ep, result)
            rows.append(result_row)
            episode_rows.append(result_row)

        wins_a = wins_b = draws = 0
        steps_a = []
        steps_b = []
        dt_a = []
        dt_b = []
        decisions_a = []
        decisions_b = []
        valid_a = []
        valid_b = []
        tokens_a = []
        tokens_b = []

        for row in rows:
            event = str(row.get('result'))
            if event == win_event_a:
                wins_a += 1
            elif event == win_event_b:
                wins_b += 1
            else:
                draws += 1

            sa = int(row.get(f'{side_a}_steps', 0) or 0)
            sb = int(row.get(f'{side_b}_steps', 0) or 0)
            ta = int(row.get(f'{side_a}_total_tokens', 0) or 0)
            tb = int(row.get(f'{side_b}_total_tokens', 0) or 0)
            steps_a.append(sa)
            steps_b.append(sb)
            dt_a.append(float(row.get(f'{side_a}_avg_decision_time', 0.0) or 0.0))
            dt_b.append(float(row.get(f'{side_b}_avg_decision_time', 0.0) or 0.0))
            decisions_a.append(int(row.get(f'{side_a}_decisions', 0) or 0))
            decisions_b.append(int(row.get(f'{side_b}_decisions', 0) or 0))
            valid_a.append(int(row.get(f'{side_a}_valid_decisions', 0) or 0))
            valid_b.append(int(row.get(f'{side_b}_valid_decisions', 0) or 0))
            tokens_a.append(ta)
            tokens_b.append(tb)

        total = wins_a + wins_b + draws
        total = total if total > 0 else 1

        tps_a = [(tokens_a[i] / steps_a[i]) if steps_a[i] > 0 else None for i in range(len(rows))]
        tps_b = [(tokens_b[i] / steps_b[i]) if steps_b[i] > 0 else None for i in range(len(rows))]
        tps_a_clean = [float(v) for v in tps_a if v is not None]
        tps_b_clean = [float(v) for v in tps_b if v is not None]

        summary_rows.append({
            side_a: THINK_AGENT_NAME,
            side_b: baseline,
            f'{side_a}_win_rate': round(wins_a / total, 3),
            f'{side_b}_win_rate': round(wins_b / total, 3),
            'draw_rate': round(draws / total, 3),
            f'{side_a}_avg_steps': round(sum(steps_a) / len(steps_a), 2) if steps_a else None,
            f'{side_b}_avg_steps': round(sum(steps_b) / len(steps_b), 2) if steps_b else None,
            f'{side_a}_avg_decision_time': round(sum(dt_a) / len(dt_a), 4) if dt_a else None,
            f'{side_b}_avg_decision_time': round(sum(dt_b) / len(dt_b), 4) if dt_b else None,
            f'{side_a}_avg_decisions': round(sum(decisions_a) / len(decisions_a), 2) if decisions_a else None,
            f'{side_b}_avg_decisions': round(sum(decisions_b) / len(decisions_b), 2) if decisions_b else None,
            f'{side_a}_avg_valid_decisions': round(sum(valid_a) / len(valid_a), 2) if valid_a else None,
            f'{side_b}_avg_valid_decisions': round(sum(valid_b) / len(valid_b), 2) if valid_b else None,
            f'{side_a}_avg_total_tokens': round(sum(tokens_a) / len(tokens_a), 2) if tokens_a else None,
            f'{side_b}_avg_total_tokens': round(sum(tokens_b) / len(tokens_b), 2) if tokens_b else None,
            f'{side_a}_avg_tokens_per_step': round(sum(tps_a_clean) / len(tps_a_clean), 2) if tps_a_clean else None,
            f'{side_b}_avg_tokens_per_step': round(sum(tps_b_clean) / len(tps_b_clean), 2) if tps_b_clean else None,
        })

    return summary_rows, episode_rows


def run_exp5() -> None:
    ts = time.strftime('%Y%m%d_%H%M%S')
    budgets = [float(x) for x in parse_csv_list('THINK_AGENT_TIME_BUDGETS', DEFAULT_BUDGETS)]
    game_baselines = {
        'maze': parse_csv_list('THINK_AGENT_MAZE_BASELINES', DEFAULT_BASELINES),
        'chess': parse_csv_list('THINK_AGENT_CHESS_BASELINES', DEFAULT_BASELINES),
    }
    game_enabled = {
        'maze': RUN_MAZE,
        'chess': RUN_CHESS,
    }

    for budget_s in budgets:
        rows_summary_by_game: dict[str, list[dict]] = {}
        rows_episode_by_game: dict[str, list[dict]] = {}
        for game in ('maze', 'chess'):
            if not game_enabled[game]:
                rows_summary_by_game[game], rows_episode_by_game[game] = [], []
                continue
            rows_summary_by_game[game], rows_episode_by_game[game] = _run_budget_matchups(
                game=game,
                baselines=game_baselines[game],
                budget_s=budget_s,
                ts=ts,
            )

        budget_tag = str(budget_s).replace('.', 'p')
        for game in ('maze', 'chess'):
            write_experiment_csvs(
                exp_name=f'exp5_b{budget_tag}',
                game=game,
                ts=ts,
                rows_summary=rows_summary_by_game[game],
                rows_episode=rows_episode_by_game[game],
                summary_suffix='summary',
            )


if __name__ == '__main__':
    run_exp5()
