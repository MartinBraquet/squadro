import json
import random
from contextlib import nullcontext
from multiprocessing import Process, Manager
from multiprocessing.managers import ArrayProxy  # noqa
from pathlib import Path

import torch
from torch import nn, optim

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.alphabeta_agent import AlphaBetaAdvancementDeepAgent
from squadro.agents.montecarlo_agent import MonteCarloDeepQLearningAgent
from squadro.evaluators.evaluator import DeepQLearningEvaluator
from squadro.tools.constants import DefaultParams, inf
from squadro.tools.log import training_logger as logger


class DeepQLearningTrainer:
    """
    Deep Q-learning trainer.
    Update a Q-network according to a discounted reward function: Q(s) = sum(gamma^t * r_t)
    """

    def __init__(
        self,
        n_pawns=None,
        lr=None,
        n_steps=None,
        backprop_interval=None,
        eval_interval=None,
        eval_steps=None,
        model_path=None,
        parallel=None,
        max_mcts_steps=None,
        init_from=None,
    ):
        """
        :param n_pawns: number of pawns in the game.
        :param lr: learning rate.
        :param n_steps: number of training steps.
        :param eval_interval: interval at which to evaluate the agent.
        :param eval_steps: number of steps to evaluate the agent.
        :param model_path: path to save the model.
        :param parallel: whether to run in parallel on multiple CPUs/GPUs.
        """
        self.n_pawns = n_pawns or DefaultParams.n_pawns
        assert 2 <= self.n_pawns <= 4, "n_pawns must be between 2 and 4"
        self.n_steps = n_steps or int(5e4)
        self.eval_interval = eval_interval or int(100 if parallel else 500)
        self.eval_steps = eval_steps or int(100)
        self.parallel = parallel if parallel is not None else False
        self.mcts_kwargs = dict(
            max_steps=max_mcts_steps or 20,
            max_time_per_move=inf,
        )

        self.backprop_interval = backprop_interval or 100
        if self.n_steps < self.backprop_interval:
            self.backprop_interval = 1

        self.agent = MonteCarloDeepQLearningAgent(
            model_path=model_path,
            is_training=True,
            **self.mcts_kwargs,
        )
        self.evaluator_old = DeepQLearningEvaluator(model_path=Path(self.model_path) / "old")

        if init_from == 'scratch':
            self.evaluator.erase(self.n_pawns)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr or 1e-3)

    @property
    def model_path(self):
        return self.evaluator.model_path

    @property
    def evaluator(self) -> DeepQLearningEvaluator:
        return self.agent.evaluator  # noqa

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def run(self) -> None:
        """
        Run the training loop.
        """
        logger.info(
            f"Starting training: {self.n_pawns} pawns\n"
            f"lr: {self.lr}\n"
            f"{self.n_steps} steps\n"
            f"backprop_interval: {self.backprop_interval}\n"
            f"eval_interval: {self.eval_interval}\n"
            f"eval_steps: {self.eval_steps}\n"
            f"path: {self.model_path}"
        )
        self.evaluator.get_model(n_pawns=self.n_pawns)
        self.evaluator.dump(self.evaluator_old.model_path)
        self.evaluator_old.clear()

        if self.parallel:
            manager = Manager()
            q_shared = manager.dict()
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
            self.evaluator.set_model(q_shared, n_pawns=self.n_pawns)

        replay_buffer = []
        results_eval = {}
        for n in range(self.n_steps // self.backprop_interval):

            for i in range(self.backprop_interval):
                replay_buffer += self.get_training_samples()
                step = n * self.backprop_interval + i + 1
                print(step)
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
                    results_eval[step] = dict(
                        ev=ev,
                        ev_random=ev_random,
                        ev_ab_deep=ev_other,
                    )
                    logger.info(f'{pid}, dump start')
                    json.dump(results_eval, open(f'results_eval.json', 'w'))
                    self.evaluator.dump()
                    logger.info(f'{pid}, dump end')

            with lock if self.parallel else nullcontext():
                logger.info(f'{pid}, back_propagate start')
                self.back_propagate(replay_buffer)
                logger.info(f'{pid}, back_propagate end')

        self.evaluator.dump()

    def get_training_samples(self):
        game = Game(
            n_pawns=self.n_pawns,
            agent_0=self.agent,
            agent_1=self.agent,
            save_states=True,
        )
        game.run()

        history = []
        for i, s in enumerate(game.state_history):
            history.append((
                s,
                game.move_info[i]['mcts_move_probs'] if i < len(game.move_info) else None,
                1 if game.winner == s.cur_player else -1,
            ))

        return history

    def back_propagate(self, replay_buffer: list) -> None:
        """
        Update the Q-network based on the reward obtained at the end of the game.

        :param replay_buffer: List of training samples.
        :return: None
        """
        self.model.train()
        v_loss = nn.MSELoss()
        p_loss = nn.CrossEntropyLoss()
        batches = random.sample(replay_buffer, self.backprop_interval * 10)

        for state, probs, reward in batches:
            assert reward in {-1, 1}
            reward = torch.FloatTensor([reward])
            self.optimizer.zero_grad()
            p, v = self.evaluator.evaluate(state, is_torch=True, check_game_over=False)
            check_nan(p)
            check_nan(v)
            check_nan(reward)
            loss = v_loss(reward, v)
            if probs is not None:
                probs = torch.from_numpy(probs)
                check_nan(probs)
                loss += p_loss(p, probs)
            check_nan(loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()

    @property
    def model(self) -> nn.Module:  # noqa
        return self.evaluator.get_model(n_pawns=self.n_pawns)

    def evaluate_agent(self, vs: str | Agent = None) -> float:
        """
        Evaluate the success rate of the current agent against another agent.
        """
        agent = MonteCarloDeepQLearningAgent(evaluator=self.evaluator, **self.mcts_kwargs)

        if vs == 'initial':
            vs = MonteCarloDeepQLearningAgent(evaluator=self.evaluator_old, **self.mcts_kwargs)
        elif vs is None:
            vs = agent

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


def check_nan(tensor):
    """
    >>> check_nan(torch.Tensor([1, 2, 3]))
    >>> check_nan(torch.Tensor([1, float('nan'), 3]))
    Traceback (most recent call last):
       ...
    RuntimeError: NaN detected in network output: tensor([1., nan, 3.])
    """
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in network output: {tensor}")
