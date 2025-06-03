import os

from squadro.evaluators.evaluator import Evaluator
from squadro.game import Game
from squadro.tools.constants import DATA_PATH
from squadro.tools.dates import get_now
from squadro.tools.log import benchmark_logger as logger


def benchmark(
    agent_0,
    agent_1,
    n=100,
    save_loss=False,
    min_win_ratio=None,
    n_pawns=5,
) -> float:
    """
    Benchmark evaluation between two agents.
    """
    Evaluator.reload()
    me, vs = agent_0, agent_1
    v = {agent_id: {first: dict(win=0, n=0) for first in (0, 1)} for agent_id in (0, 1)}
    wins = 0
    n_per_section = n // 4
    n = 4 * n_per_section

    if save_loss:
        if not os.path.exists(path := DATA_PATH / 'benchmark'):
            os.makedirs(path)
        if not os.path.exists(path := DATA_PATH / 'benchmark/games'):
            os.makedirs(path)

    for agent_id in (0, 1):
        agent_0, agent_1 = (me, vs) if agent_id == 0 else (vs, me)
        for first in (0, 1):
            for _ in range(n_per_section):
                game = Game(
                    n_pawns=n_pawns,
                    agent_0=agent_0,
                    agent_1=agent_1,
                    first=first,
                )
                game.run()
                # logger.info(game.action_history)
                win = int(game.winner == agent_id)
                v[agent_id][first]['win'] += win
                if save_loss and not win:
                    game.to_file(DATA_PATH / f'benchmark/games/{get_now()}.json')
            wins += v[agent_id][first]['win']
            v[agent_id][first]['n'] = n_per_section
            win_rate = v[agent_id][first]['win'] / n_per_section
            logger.info(
                f"Current model as agent {agent_id}, first {first}: {win_rate * 100 :.0f}% win")

    win_rate = wins / n

    logger.info(v)

    if min_win_ratio:
        assert win_rate >= min_win_ratio / 100, f'{win_rate=} must be higher than {min_win_ratio}%'

    return win_rate
