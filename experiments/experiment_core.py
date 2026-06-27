import csv
import os
import time
from typing import Callable

# core 三块：
# 1) 日志与结果输出
# 2) API 失败重启 + 单局执行
# 3) 实验参数统计（多局聚合）

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Experiments', 'records')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1) 日志与结果输出
# ============================================================
def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_list(env_key: str, default: list[str]) -> list[str]:
    raw = os.getenv(env_key, '').strip()
    if not raw:
        return list(default)
    out = [x.strip() for x in raw.split(',') if x.strip()]
    return out if out else list(default)


# ============================================================
# 2) API 失败重启 + 单局执行
# ============================================================


def _maze_result_from_payload(payload: dict) -> str:
    reason = str(payload.get('termination_reason', '') or '').lower()
    if 'red_reached' in reason:
        return 'red_goal'
    if 'blue_reached' in reason:
        return 'blue_goal'
    if 'both_reached' in reason:
        return 'draw'
    if 'timeout' in reason:
        return 'timeout'
    return 'draw'


def _chess_result_from_payload(payload: dict) -> str:
    reason = str(payload.get('termination_reason', '') or '')
    if 'White wins' in reason:
        return 'white_win'
    if 'Black wins' in reason:
        return 'black_win'
    return 'draw'


def _episode_result_from_payload(
    *,
    game: str,
    payload: dict,
    side_a: str,
    side_b: str,
    stats_a: dict,
    stats_b: dict,
) -> dict:
    if game == 'maze':
        result = _maze_result_from_payload(payload)
    elif game == 'chess':
        result = _chess_result_from_payload(payload)
    else:
        raise ValueError(f'Unknown game: {game}')

    out = {
        'result': result,
        f'{side_a}_steps': int(payload.get(f'{side_a}_steps', 0) or 0),
        f'{side_b}_steps': int(payload.get(f'{side_b}_steps', 0) or 0),
        f'{side_a}_avg_dt': float(stats_a.get('avg_decision_time', 0.0) or 0.0),
        f'{side_b}_avg_dt': float(stats_b.get('avg_decision_time', 0.0) or 0.0),
        f'{side_a}_decisions': int(stats_a.get('decisions', 0) or 0),
        f'{side_b}_decisions': int(stats_b.get('decisions', 0) or 0),
        f'{side_a}_valid_decisions': int(stats_a.get('valid_decisions', 0) or 0),
        f'{side_b}_valid_decisions': int(stats_b.get('valid_decisions', 0) or 0),
        f'{side_a}_submitted_decisions': int(stats_a.get('submitted_decisions', 0) or 0),
        f'{side_b}_submitted_decisions': int(stats_b.get('submitted_decisions', 0) or 0),
        f'{side_a}_last_action_source': str(stats_a.get('last_action_source', '') or ''),
        f'{side_b}_last_action_source': str(stats_b.get('last_action_source', '') or ''),
        f'{side_a}_tokens': int(stats_a.get('tokens', 0) or 0),
        f'{side_b}_tokens': int(stats_b.get('tokens', 0) or 0),
        f'{side_a}_tokens_fast': int(stats_a.get('tokens_fast', 0) or 0),
        f'{side_a}_tokens_slow_self': int(stats_a.get('tokens_slow_self', 0) or 0),
        f'{side_a}_tokens_slow_opp': int(stats_a.get('tokens_slow_opp', 0) or 0),
        f'{side_b}_tokens_fast': int(stats_b.get('tokens_fast', 0) or 0),
        f'{side_b}_tokens_slow_self': int(stats_b.get('tokens_slow_self', 0) or 0),
        f'{side_b}_tokens_slow_opp': int(stats_b.get('tokens_slow_opp', 0) or 0),
        f'{side_a}_step_decision_times': list(stats_a.get('step_decision_times', []) or []),
        f'{side_b}_step_decision_times': list(stats_b.get('step_decision_times', []) or []),
        f'{side_a}_step_tokens': list(stats_a.get('step_tokens', []) or []),
        f'{side_b}_step_tokens': list(stats_b.get('step_tokens', []) or []),
        'termination_reason': payload.get('termination_reason'),
    }
    return out


def run_with_retries(
    run_once: Callable[[int], dict],
    *,
    restart_limit: int,
    label: str,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> dict:
    retry = 0
    while True:
        try:
            return run_once(retry)
        except retry_exceptions as exc:
            retry += 1
            print(f'🔁 [{label}] restarting ({retry}/{restart_limit}): {exc}')
            if retry >= int(restart_limit):
                raise


def run_single_episode(
    *,
    game: str,
    agent_a_name: str,
    agent_b_name: str,
    timeout_s: float,
) -> dict:
    final_payload: dict = {}

    def _push_callback(payload: dict):
        nonlocal final_payload
        if isinstance(payload, dict) and payload.get('game_over'):
            final_payload = dict(payload)

    if game == 'maze':
        from main_realtime_maze import RealtimeMazeController
        red_model = str(agent_a_name)
        blue_model = str(agent_b_name)
        controller = RealtimeMazeController(red_model, blue_model, push_callback=_push_callback)
        side_a, side_b = 'red', 'blue'
    elif game == 'chess':
        from main_realtime_chess import RealtimeChessController
        white_model = str(agent_a_name)
        black_model = str(agent_b_name)
        controller = RealtimeChessController(white_model, black_model, push_callback=_push_callback)
        side_a, side_b = 'white', 'black'
    else:
        raise ValueError(f'Unknown game: {game}')

    started = controller.start_game()
    if not started:
        raise RuntimeError(f'{game} controller failed to start')

    t0 = time.time()
    while not controller.events['end'].is_set():
        if (time.time() - t0) > float(timeout_s):
            controller.stop_game()
            raise TimeoutError(f'{game} episode timeout: {timeout_s}s')
        time.sleep(0.05)

    if not final_payload:
        raise RuntimeError(f'{game} episode ended without final payload')

    stats_a = dict(final_payload.get(f'{side_a}_stats') or {})
    stats_b = dict(final_payload.get(f'{side_b}_stats') or {})

    return _episode_result_from_payload(
        game=game,
        payload=final_payload,
        side_a=side_a,
        side_b=side_b,
        stats_a=stats_a,
        stats_b=stats_b,
    )


# ============================================================
# 3) 实验参数统计（多局聚合）
# ============================================================
def evaluate_pair(
    *,
    game: str,
    agent_a_name: str,
    agent_b_name: str,
    episodes: int,
    restart_limit: int,
    print_progress: bool = True,
) -> tuple[dict, list[dict]]:
    game_cfg = {
        'maze': {'side_a': 'red', 'side_b': 'blue', 'win_event_a': 'red_goal', 'win_event_b': 'blue_goal'},
        'chess': {'side_a': 'white', 'side_b': 'black', 'win_event_a': 'white_win', 'win_event_b': 'black_win'},
    }
    cfg = game_cfg.get(str(game))
    if cfg is None:
        raise ValueError(f'Unknown game: {game}')
    side_a = str(cfg['side_a'])
    side_b = str(cfg['side_b'])
    win_event_a = str(cfg['win_event_a'])
    win_event_b = str(cfg['win_event_b'])
    timeout_s = float(os.getenv('THINK_AGENT_EPISODE_TIMEOUT', '300'))

    wins_a = wins_b = draws = 0
    rows: list[dict] = []

    for episode in range(int(episodes)):
        if print_progress:
            print(f'[{game}] {agent_a_name} vs {agent_b_name} | episode {episode + 1}/{episodes}')

        result = run_with_retries(
            lambda _retry: run_single_episode(
                game=game,
                agent_a_name=agent_a_name,
                agent_b_name=agent_b_name,
                timeout_s=timeout_s,
            ),
            restart_limit=restart_limit,
            label=f'{game}:{agent_a_name} vs {agent_b_name} ep{episode + 1}',
            retry_exceptions=(Exception,),
        )

        rows.append(result)
        event = str(result.get('result'))
        if event == win_event_a:
            wins_a += 1
        elif event == win_event_b:
            wins_b += 1
        else:
            draws += 1

    total = wins_a + wins_b + draws
    total = total if total > 0 else 1

    steps_a = [int(r.get(f'{side_a}_steps', 0) or 0) for r in rows]
    steps_b = [int(r.get(f'{side_b}_steps', 0) or 0) for r in rows]
    dt_a = [float(r.get(f'{side_a}_avg_dt', 0.0) or 0.0) for r in rows]
    dt_b = [float(r.get(f'{side_b}_avg_dt', 0.0) or 0.0) for r in rows]
    decisions_a = [int(r.get(f'{side_a}_decisions', 0) or 0) for r in rows]
    decisions_b = [int(r.get(f'{side_b}_decisions', 0) or 0) for r in rows]
    valid_a = [int(r.get(f'{side_a}_valid_decisions', 0) or 0) for r in rows]
    valid_b = [int(r.get(f'{side_b}_valid_decisions', 0) or 0) for r in rows]
    tokens_a = [int(r.get(f'{side_a}_tokens', 0) or 0) for r in rows]
    tokens_b = [int(r.get(f'{side_b}_tokens', 0) or 0) for r in rows]
    tps_a = [(tokens_a[i] / steps_a[i]) if steps_a[i] > 0 else None for i in range(len(rows))]
    tps_b = [(tokens_b[i] / steps_b[i]) if steps_b[i] > 0 else None for i in range(len(rows))]
    tps_a_clean = [float(v) for v in tps_a if v is not None]
    tps_b_clean = [float(v) for v in tps_b if v is not None]

    summary = {
        side_a: agent_a_name,
        side_b: agent_b_name,
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
    }

    return summary, rows


def _game_sides(game: str) -> tuple[str, str]:
    if game == 'maze':
        return 'red', 'blue'
    if game == 'chess':
        return 'white', 'black'
    raise ValueError(f'Unknown game: {game}')


def summary_fields(game: str, extra_fields: list[str] | None = None) -> list[str]:
    side_a, side_b = _game_sides(game)
    fields = [
        side_a,
        side_b,
        f'{side_a}_win_rate',
        f'{side_b}_win_rate',
        'draw_rate',
        f'{side_a}_avg_steps',
        f'{side_b}_avg_steps',
        f'{side_a}_avg_decision_time',
        f'{side_b}_avg_decision_time',
        f'{side_a}_avg_decisions',
        f'{side_b}_avg_decisions',
        f'{side_a}_avg_valid_decisions',
        f'{side_b}_avg_valid_decisions',
        f'{side_a}_avg_total_tokens',
        f'{side_b}_avg_total_tokens',
        f'{side_a}_avg_tokens_per_step',
        f'{side_b}_avg_tokens_per_step',
    ]
    if extra_fields:
        fields.extend(extra_fields)
    return fields


def episode_row(game: str, agent_a: str, agent_b: str, episode: int, result: dict, extra: dict | None = None) -> dict:
    side_a, side_b = _game_sides(game)
    row = {
        side_a: agent_a,
        side_b: agent_b,
        'episode': episode,
        'result': result.get('result'),
        f'{side_a}_steps': int(result.get(f'{side_a}_steps', 0) or 0),
        f'{side_b}_steps': int(result.get(f'{side_b}_steps', 0) or 0),
        f'{side_a}_avg_decision_time': float(result.get(f'{side_a}_avg_dt', 0.0) or 0.0),
        f'{side_b}_avg_decision_time': float(result.get(f'{side_b}_avg_dt', 0.0) or 0.0),
        f'{side_a}_decisions': int(result.get(f'{side_a}_decisions', 0) or 0),
        f'{side_b}_decisions': int(result.get(f'{side_b}_decisions', 0) or 0),
        f'{side_a}_valid_decisions': int(result.get(f'{side_a}_valid_decisions', 0) or 0),
        f'{side_b}_valid_decisions': int(result.get(f'{side_b}_valid_decisions', 0) or 0),
        f'{side_a}_submitted_decisions': int(result.get(f'{side_a}_submitted_decisions', 0) or 0),
        f'{side_b}_submitted_decisions': int(result.get(f'{side_b}_submitted_decisions', 0) or 0),
        f'{side_a}_last_action_source': str(result.get(f'{side_a}_last_action_source', '') or ''),
        f'{side_b}_last_action_source': str(result.get(f'{side_b}_last_action_source', '') or ''),
        f'{side_a}_total_tokens': int(result.get(f'{side_a}_tokens', 0) or 0),
        f'{side_b}_total_tokens': int(result.get(f'{side_b}_tokens', 0) or 0),
        f'{side_a}_tokens_fast': int(result.get(f'{side_a}_tokens_fast', 0) or 0),
        f'{side_a}_tokens_slow_self': int(result.get(f'{side_a}_tokens_slow_self', 0) or 0),
        f'{side_a}_tokens_slow_opp': int(result.get(f'{side_a}_tokens_slow_opp', 0) or 0),
        f'{side_b}_tokens_fast': int(result.get(f'{side_b}_tokens_fast', 0) or 0),
        f'{side_b}_tokens_slow_self': int(result.get(f'{side_b}_tokens_slow_self', 0) or 0),
        f'{side_b}_tokens_slow_opp': int(result.get(f'{side_b}_tokens_slow_opp', 0) or 0),
        f'{side_a}_step_decision_times': repr(result.get(f'{side_a}_step_decision_times', [])),
        f'{side_b}_step_decision_times': repr(result.get(f'{side_b}_step_decision_times', [])),
        f'{side_a}_step_tokens': repr(result.get(f'{side_a}_step_tokens', [])),
        f'{side_b}_step_tokens': repr(result.get(f'{side_b}_step_tokens', [])),
    }
    if extra:
        row.update(extra)
    return row


def write_experiment_csvs(
    *,
    exp_name: str,
    game: str,
    ts: str,
    rows_summary: list[dict],
    rows_episode: list[dict],
    summary_suffix: str = 'summary',
    extra_summary_fields: list[str] | None = None,
) -> None:
    if rows_summary:
        write_csv(
            os.path.join(RESULTS_DIR, f'{exp_name}_{game}_{summary_suffix}_{ts}.csv'),
            summary_fields(game, extra_summary_fields),
            rows_summary,
        )
    if rows_episode:
        write_csv(
            os.path.join(RESULTS_DIR, f'{exp_name}_{game}_episode_{ts}.csv'),
            list(rows_episode[0].keys()),
            rows_episode,
        )


def run_experiment(
    *,
    exp_name: str,
    pair_provider: Callable[[str], list[tuple[str, str]]],
    episodes: int,
    restart_limit: int,
    run_maze: bool,
    run_chess: bool,
    print_progress: bool,
    swap: bool,
    summary_suffix: str = 'summary',
) -> None:
    ts = time.strftime('%Y%m%d_%H%M%S')

    rows_maze_summary: list[dict] = []
    rows_maze_episode: list[dict] = []
    rows_chess_summary: list[dict] = []
    rows_chess_episode: list[dict] = []

    if run_maze:
        for a, b in pair_provider('maze'):
            directions = [(a, b)] if not swap else [(a, b), (b, a)]
            for red_name, blue_name in directions:
                summary, episode_results = evaluate_pair(
                    game='maze',
                    agent_a_name=red_name,
                    agent_b_name=blue_name,
                    episodes=episodes,
                    restart_limit=restart_limit,
                    print_progress=print_progress,
                )
                rows_maze_summary.append(summary)
                for i, result in enumerate(episode_results, start=1):
                    rows_maze_episode.append(episode_row('maze', red_name, blue_name, i, result))

    if run_chess:
        for a, b in pair_provider('chess'):
            directions = [(a, b)] if not swap else [(a, b), (b, a)]
            for white_name, black_name in directions:
                summary, episode_results = evaluate_pair(
                    game='chess',
                    agent_a_name=white_name,
                    agent_b_name=black_name,
                    episodes=episodes,
                    restart_limit=restart_limit,
                    print_progress=print_progress,
                )
                rows_chess_summary.append(summary)
                for i, result in enumerate(episode_results, start=1):
                    rows_chess_episode.append(episode_row('chess', white_name, black_name, i, result))

    write_experiment_csvs(
        exp_name=exp_name,
        game='maze',
        ts=ts,
        rows_summary=rows_maze_summary,
        rows_episode=rows_maze_episode,
        summary_suffix=summary_suffix,
    )
    write_experiment_csvs(
        exp_name=exp_name,
        game='chess',
        ts=ts,
        rows_summary=rows_chess_summary,
        rows_episode=rows_chess_episode,
        summary_suffix=summary_suffix,
    )
