from squadro import MonteCarloDeepQLearningAgent
from squadro import logger
from squadro.agents.montecarlo_agent import MonteCarloRolloutAgent, MonteCarloAdvancementAgent
from squadro.core.game import Game
from squadro.state.evaluators.base import Evaluator
from squadro.tools.basic import PrettyDict
from squadro.tools.constants import DATA_PATH, DefaultParams
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


def benchmark_agents(names, n_games=100, n_pawns=5, max_time_per_move=3, results=None):
    if logger.client is None:
        logger.setup(section=['benchmark', 'main'])

    DefaultParams.max_time_per_move = max_time_per_move
    mcts_agent_kwargs = dict(
        is_training=True,
        mcts_kwargs=PrettyDict(tau=.5, p_mix=0, a_dirichlet=0),
    )

    agents = {}
    for name in names:
        agents[name] = name
    if 'mcts_rollout' in names:
        agents['mcts_rollout'] = MonteCarloRolloutAgent(**mcts_agent_kwargs)
    if 'mcts_advancement' in names:
        agents['mcts_advancement'] = MonteCarloAdvancementAgent(**mcts_agent_kwargs)
    if 'mcts_deep_q_learning' in names:
        agents['mcts_deep_q_learning'] = MonteCarloDeepQLearningAgent(**mcts_agent_kwargs)

    names = list(agents.keys())
    logger.info(names)

    results = results or {}
    for i in range(len(agents)):
        name_i = names[i]
        if name_i not in results:
            results[name_i] = {}
        for j in range(i + 1, len(agents)):
            name_j = names[j]
            if results[name_i].get(name_j) is not None:
                continue
            logger.info(f"{name_i} vs {name_j}")
            results[name_i][name_j] = result = benchmark(
                agent_0=agents[name_i],
                agent_1=agents[name_j],
                n_pawns=n_pawns,
                n_games=n_games,
            )
            logger.info(f"{name_i} vs {name_j}: {result}\n")

    return results
