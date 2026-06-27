import json
import os
from pathlib import Path

from Experiments.experiment_core import parse_csv_list, run_experiment

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '20'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
SWAP = os.getenv('THINK_AGENT_SWAP', '0') != '0'

THINK_AGENT_NAME = 'Think_Agent'
DEFAULT_BASELINES = ['Reflexion_Agent', 'CoT_Agent', 'LLM_Agent', 'MemoryLLM_Agent', 'CodingPairs_Agent', 'Rule_Agent']

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent
_CONFIG_PATH = _PROJECT_ROOT / 'config.json'


def _pair_provider(game: str) -> list[tuple[str, str]]:
    key = 'THINK_AGENT_MAZE_BASELINES' if game == 'maze' else 'THINK_AGENT_CHESS_BASELINES'
    baselines = parse_csv_list(key, DEFAULT_BASELINES)
    return [(THINK_AGENT_NAME, baseline) for baseline in baselines]


def _run_one(*, exp_name: str, run_chess: bool, run_maze: bool, game_settings_updates: dict) -> None:
    original_cfg = json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
    cfg = dict(original_cfg)
    gs = dict(cfg.get('game_settings') or {})
    if not isinstance(gs, dict):
        raise RuntimeError("config.json missing 'game_settings' dict")
    gs.update(game_settings_updates)
    cfg['game_settings'] = gs

    try:
        _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        run_experiment(
            exp_name=exp_name,
            pair_provider=_pair_provider,
            episodes=EPISODES,
            restart_limit=RESTART_LIMIT,
            run_maze=run_maze,
            run_chess=run_chess,
            print_progress=PRINT_PROGRESS,
            swap=SWAP,
            summary_suffix='env_sweep',
        )
    finally:
        _CONFIG_PATH.write_text(json.dumps(original_cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_exp6() -> None:
    if RUN_CHESS:
        for cooldown in (5, 10, 15):
            _run_one(
                exp_name=f'exp6_chess_cd{cooldown}s',
                run_chess=True,
                run_maze=False,
                game_settings_updates={
                    'chess_cooldown': float(cooldown),
                    'chess_ai_min_interval': float(cooldown),
                },
            )

    if RUN_MAZE:
        for n, p in ((10, 0.15), (15, 0.10), (20, 0.05)):
            pct = int(round(float(p) * 100))
            _run_one(
                exp_name=f'exp6_maze_n{n}_p{pct}',
                run_chess=False,
                run_maze=True,
                game_settings_updates={
                    'maze_logical_size': int(n),
                    'maze_loop_probability': float(p),
                },
            )


if __name__ == '__main__':
    run_exp6()
