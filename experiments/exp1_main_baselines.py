import os
from Experiments.experiment_core import parse_csv_list, run_experiment

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '5'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
SWAP = os.getenv('THINK_AGENT_SWAP', '0') != '0'

THINK_AGENT_NAME = 'Think_Agent'
DEFAULT_BASELINES = ['Reflexion_Agent', 'CoT_Agent', 'LLM_Agent', 'MemoryLLM_Agent', 'CodingPairs_Agent', 'Rule_Agent']


def _pair_provider(game: str) -> list[tuple[str, str]]:
    key = 'THINK_AGENT_MAZE_BASELINES' if game == 'maze' else 'THINK_AGENT_CHESS_BASELINES'
    baselines = parse_csv_list(key, DEFAULT_BASELINES)
    return [(THINK_AGENT_NAME, baseline) for baseline in baselines]


def run_exp1() -> None:
    run_experiment(
        exp_name='exp1',
        pair_provider=_pair_provider,
        episodes=EPISODES,
        restart_limit=RESTART_LIMIT,
        run_maze=RUN_MAZE,
        run_chess=RUN_CHESS,
        print_progress=PRINT_PROGRESS,
        swap=SWAP,
        summary_suffix='baseline',
    )


if __name__ == '__main__':
    run_exp1()
