from squadro.core.game import Game
from squadro.state.evaluators.base import Evaluator
from squadro.tools.constants import DATA_PATH
from squadro.tools.dates import get_now
from squadro.tools.disk import mkdir
from squadro.tools.logs import benchmark_logger as logger


class Benchmark:
    def __init__(
        self,
        agent_0,
        agent_1,
        n_games=100,
        save_loss=False,
        min_win_ratio=None,
        n_pawns=5,
    ):
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.n = int(n_games)
        self.save_loss = save_loss
        if min_win_ratio is not None:
            min_win_ratio = float(min_win_ratio)
            if min_win_ratio > 1:
                min_win_ratio /= 100
        self.min_win_ratio = min_win_ratio
        self.n_pawns = n_pawns

        self.win_rates = {(a_id, f): 0 for a_id in (0, 1) for f in (0, 1)}

    def run(self) -> float:
        Evaluator.reload()
        me, vs = self.agent_0, self.agent_1
        wins = 0
        n_per_section = max(self.n // 4, 1)
        n = 4 * n_per_section

        game_path = DATA_PATH / 'benchmark/games'
        if self.save_loss:
            mkdir(game_path)

        for agent_id in (0, 1):
            agent_0, agent_1 = (me, vs) if agent_id == 0 else (vs, me)
            for first in (0, 1):
                for _ in range(n_per_section):
                    game = Game(
                        n_pawns=self.n_pawns,
                        agent_0=agent_0,
                        agent_1=agent_1,
                        first=first,
                        plot=False,
                    )
                    game.run()
                    # logger.info(game.action_history)
                    win = int(game.winner == agent_id)
                    self.win_rates[agent_id, first] += win
                    if self.save_loss and not win:
                        game.to_file(game_path / f'{get_now()}.json')
                wins += self.win_rates[agent_id, first]
                self.win_rates[agent_id, first] /= n_per_section
                logger.info(
                    f"Current model as agent {agent_id}, first {first}: "
                    f"{self.win_rates[agent_id, first]:.0%} win"
                )

        # logger.info(self.win_rates)

        win_rate = wins / n

        if self.min_win_ratio and win_rate < self.min_win_ratio:
            raise RuntimeError(
                f'Win rate of {win_rate:.0%} failed to beat {self.min_win_ratio:.0%}.')

        return win_rate


def benchmark(*args, **kwargs) -> float:
    """
    Benchmark evaluation between two agents.
    """
    return Benchmark(*args, **kwargs).run()
