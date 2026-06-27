import os
from Experiments.experiment_core import run_experiment

RUN_MAZE = os.getenv('THINK_AGENT_RUN_MAZE', '0') != '0'
RUN_CHESS = os.getenv('THINK_AGENT_RUN_CHESS', '1') != '0'
PRINT_PROGRESS = os.getenv('THINK_AGENT_PROGRESS', '1') != '0'
EPISODES = int(os.getenv('THINK_AGENT_EPISODES', '3'))
RESTART_LIMIT = int(os.getenv('THINK_AGENT_RESTART_LIMIT', '50'))
SWAP = os.getenv('THINK_AGENT_SWAP', '0') != '0'

PAIRS = [
    ('Think_Agent', 'FastOnly_Agent'),
    ('Think_Agent', 'SlowOnly_Agent'),
    ('Think_Agent', 'Think_WithoutOpponent_Agent'),
    ('Think_Agent', 'Think_WithoutSelf_Agent'),
]


def _pair_provider(_game: str) -> list[tuple[str, str]]:
    return list(PAIRS)


def run_exp2() -> None:
    run_experiment(
        exp_name='exp2',
        pair_provider=_pair_provider,
        episodes=EPISODES,
        restart_limit=RESTART_LIMIT,
        run_maze=RUN_MAZE,
        run_chess=RUN_CHESS,
        print_progress=PRINT_PROGRESS,
        swap=SWAP,
        summary_suffix='ablation',
    )


if __name__ == '__main__':
    run_exp2()
