from contextlib import nullcontext
from multiprocessing import Process, Manager
from multiprocessing.managers import ArrayProxy  # noqa
from pathlib import Path

import numpy as np

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.alphabeta_agent import AlphaBetaAdvancementDeepAgent
from squadro.agents.montecarlo_agent import MonteCarloQLearningAgent
from squadro.evaluators.evaluator import QLearningEvaluator
from squadro.state import State
from squadro.tools.constants import DefaultParams
from squadro.tools.log import training_logger as logger


class QLearningTrainer:
    """
    Q-learning trainer.
    Update a Q-table according to a discounted reward function: Q(s) = sum(gamma^t * r_t)
    For 3 pawns, max len(Q) = 1 M. Must train up to len(Q) > 100 k for good results
    For 4 pawns, max len(Q) = 400 M. Must train up to len(Q) > 40 M for good results
    """

    def __init__(
        self,
        n_pawns=None,
        lr=None,
        gamma=None,
        n_steps=None,
        n_cut=None,
        eval_interval=None,
        eval_steps=None,
        model_path=None,
        parallel=None,
    ):
        """
        :param n_pawns: number of pawns in the game.
        :param lr: learning rate.
        :param gamma: discount factor.
        :param n_steps: number of training steps.
        :param eval_interval: interval at which to evaluate the agent.
        :param eval_steps: number of steps to evaluate the agent.
        :param model_path: path to save the model.
        :param parallel: whether to run in parallel on multiple CPUs/GPUs.
        """
        self.n_pawns = n_pawns or DefaultParams.n_pawns
        assert 2 <= self.n_pawns <= 4, "n_pawns must be between 2 and 4"
        self.lr = lr or .2
        self.gamma = gamma or .95
        self.n_steps = n_steps or int(5e4)
        self.eval_interval = eval_interval or int(100 if parallel else 500)
        self.eval_steps = eval_steps or int(100)
        self.parallel = parallel if parallel is not None else False

        self.n_cut = n_cut or 100
        if self.n_steps < self.n_cut or not self.parallel:
            self.n_cut = 1

        self.agent = MonteCarloQLearningAgent(
            evaluator=QLearningEvaluator(model_path=model_path),
            is_training=True,
        )
        self.evaluator_old = QLearningEvaluator(model_path=Path(self.model_path) / "old")

    @property
    def model_path(self):
        return self.evaluator.model_path

    @property
    def evaluator(self) -> QLearningEvaluator:
        return self.agent.evaluator  # noqa

    def run(self) -> None:
        """
        Run the training loop.
        """
        self.evaluator.get_Q(n_pawns=self.n_pawns)
        self.evaluator.dump(self.evaluator_old.model_path)
        self.evaluator_old.clear()
        # Should be close to 50% as the agents are the same
        # logger.info(self.evaluate_agent())

        if self.parallel:
            manager = Manager()
            if isinstance(self.Q, np.ndarray):
                q_shared = manager.Array('f', self.Q)
            else:
                q_shared = manager.dict(self.Q)
            lock = manager.Lock()

            processes = [
                Process(target=self._run, args=(q_shared, lock, i))
                for i in range(self.parallel)
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        else:
            self._run()

    def _run(self, q_shared=None, lock=None, pid=0):
        if self.parallel:
            assert lock is not None
            assert q_shared is not None
            self.evaluator.set_dict(q_shared, n_pawns=self.n_pawns)

        for n in range(self.n_steps // self.n_cut):

            state_histories = []
            for i in range(self.n_cut):
                game = Game(
                    n_pawns=self.n_pawns,
                    agent_0=self.agent,
                    agent_1=self.agent,
                    save_states=True,
                )
                game.run()
                state_histories.append(game.state_history.copy())

                step = n * self.n_cut + i + 1
                if step % self.eval_interval == 0 and pid == 0:
                    ev = self.evaluate_agent(vs='initial')
                    ev_random = self.evaluate_agent(vs='random')
                    ev_other = self.evaluate_agent(vs=AlphaBetaAdvancementDeepAgent(max_depth=5))
                    logger.info(
                        f"{step} steps"
                        f", {ev * 100 :.0f}% vs initial"
                        f", {ev_other * 100 :.0f}% vs ab_deep"
                        f", {ev_random * 100 :.0f}% vs random"
                    )
                    logger.info(f'{pid}, dump start')
                    self.evaluator.dump()
                    logger.info(f'{pid}, dump end')
                    # if ev > .9:
                    #     logger.info(f'{pid}, Overwriting initial agent for better comparison')
                    #     with lock if self.parallel else nullcontext():
                    #         self.evaluator.dump(self.evaluator_old.model_path)
                    #         # Seems to create issue where Q in process 0 gets desynchronized with other
                    #         # processes. Most likely because .reload() is clearing all Q values in
                    #         # the evaluator (should only clear Q in evaluator_old). Try using clear instead of reload
                    #         self.evaluator_old.clear()

            with lock if self.parallel else nullcontext():
                logger.info(f'{pid}, back_propagate start, {len(self.Q)}')
                for state_history in state_histories:
                    self.back_propagate(state_history)
                logger.info(f'{pid}, back_propagate end,   {len(self.Q)}')

    def back_propagate(self, state_history: list[State]) -> None:
        """
        Update the Q-table based on the reward obtained at the end of the game.

        :param state_history: List of states visited during the game.
        :return: None
        """
        state = state_history.pop()
        reward = self.evaluator.get_value(state)
        assert reward in {-1, 1}
        value = self.Q[self.evaluator.get_id(state)] = reward

        while state_history:
            value = - value
            state = state_history.pop()
            q = self.evaluator.get_value(state, check_game_over=False)
            state_id = self.evaluator.get_id(state)
            self.Q[state_id] = value = (1 - self.lr) * q + self.lr * self.gamma * value

    @property
    def Q(self) -> dict:  # noqa
        return self.evaluator.get_Q(n_pawns=self.n_pawns)

    def evaluate_agent(self, vs: str | Agent) -> float:
        """
        Evaluate the success rate of the current agent against another agent.
        """
        if vs == 'initial':
            vs = MonteCarloQLearningAgent(evaluator=self.evaluator_old)

        agent = MonteCarloQLearningAgent(
            evaluator=self.evaluator,
            is_training=False,
        )

        v = 0
        for n in range(self.eval_steps):
            g = Game(
                n_pawns=self.n_pawns,
                agent_0=agent,
                agent_1=vs,
            )
            g.run()
            v += 1 - g.winner

        return v / self.eval_steps
