import os
from itertools import combinations

from Experiments.experiment_core import parse_csv_list, run_experiment

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '1'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
SWAP = os.getenv('THINK_AGENT_SWAP', '0') != '0'

DEFAULT_POOL = ['Think_Agent', 'Reflexion_Agent', 'CoT_Agent', 'LLM_Agent', 'MemoryLLM_Agent', 'CodingPairs_Agent', 'Rule_Agent']


def _pair_provider(_game: str) -> list[tuple[str, str]]:
    pool = parse_csv_list('THINK_AGENT_POOL', DEFAULT_POOL)
    return list(combinations(list(pool), 2))


def run_exp3() -> None:
    run_experiment(
        exp_name='exp3',
        pair_provider=_pair_provider,
        episodes=EPISODES,
        restart_limit=RESTART_LIMIT,
        run_maze=RUN_MAZE,
        run_chess=RUN_CHESS,
        print_progress=PRINT_PROGRESS,
        swap=SWAP,
        summary_suffix='round_robin',
    )


if __name__ == '__main__':
    run_exp3()
