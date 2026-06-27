import os

from Experiments.experiment_core import parse_csv_list, run_experiment

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '3'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
SWAP = os.getenv('THINK_AGENT_SWAP', '0') != '0'

THINK_AGENT_NAME = 'Think_Agent'
DEFAULT_RULE_DELAYS = ['0.2', '0.5', '1.0', '2.0']


def _pair_provider(_game: str) -> list[tuple[str, str]]:
    delays = parse_csv_list('THINK_AGENT_RULE_DELAYS', DEFAULT_RULE_DELAYS)
    return [(THINK_AGENT_NAME, f'Rule_Agent:{delay}') for delay in delays]


def run_exp4() -> None:
    run_experiment(
        exp_name='exp4',
        pair_provider=_pair_provider,
        episodes=EPISODES,
        restart_limit=RESTART_LIMIT,
        run_maze=RUN_MAZE,
        run_chess=RUN_CHESS,
        print_progress=PRINT_PROGRESS,
        swap=SWAP,
        summary_suffix='rule_delay',
    )


if __name__ == '__main__':
    run_exp4()
