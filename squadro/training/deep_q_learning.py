import json
import random
import shutil
from multiprocessing.managers import ArrayProxy  # noqa
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim, Tensor

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.alphabeta_agent import AlphaBetaAdvancementDeepAgent
from squadro.agents.montecarlo_agent import MonteCarloDeepQLearningAgent
from squadro.evaluators.evaluator import DeepQLearningEvaluator
from squadro.tools.constants import DefaultParams, inf, DATA_PATH
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
        backprop_steps=None,
        eval_interval=None,
        eval_steps=None,
        model_path=None,
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
        """
        self.n_pawns = n_pawns or DefaultParams.n_pawns
        assert 2 <= self.n_pawns <= 4, "n_pawns must be between 2 and 4"
        self.n_steps = n_steps or int(5e4)
        self.eval_interval = eval_interval or int(500)
        self.eval_steps = eval_steps or int(100)
        self.mcts_kwargs = dict(
            max_steps=max_mcts_steps or 20,
            max_time_per_move=inf,
        )

        self.backprop_steps = backprop_steps or 50
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
        self.replay_buffer = ReplayBuffer(n_pawns=self.n_pawns)

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
            f"backprop interval: {self.backprop_interval}\n"
            f"backprop steps: {self.backprop_steps}\n"
            f"eval interval: {self.eval_interval}\n"
            f"eval steps: {self.eval_steps}\n"
            f"path: {self.model_path}\n"
            f"replay buffer: {self.replay_buffer}"
        )
        self.evaluator.get_model(n_pawns=self.n_pawns)
        self.evaluator.dump(self.evaluator_old.model_path)
        self.evaluator_old.clear()

        self._run()

    def _run(self):
        results_eval = {}
        for step in range(1, self.n_steps + 1):
            print(step)
            self.get_training_samples()

            if step % self.backprop_interval == 0:
                with logger.context_info('back_propagate'):
                    self.back_propagate()
                self.replay_buffer.save()

            if step % self.eval_interval == 0:
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
                with logger.context_info('dump'):
                    json.dump(results_eval, open(f'results_eval.json', 'w'))
                    self.evaluator.dump()

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
            cur_player = s.cur_player
            s = s.to_list()

            move_probs = game.move_info[i]['mcts_move_probs'] if i < len(game.move_info) else None
            check_nan(move_probs)
            if isinstance(move_probs, np.ndarray):
                move_probs = move_probs.tolist()

            history.append((
                s,
                move_probs,
                1 if game.winner == cur_player else -1,
            ))

        self.replay_buffer += history

        return history

    def back_propagate(self) -> None:
        """
        Update the Q-network based on the reward obtained at the end of the game.

        :return: None
        """
        self.model.train()
        v_loss = nn.MSELoss()
        p_loss = nn.CrossEntropyLoss()
        batches = random.sample(self.replay_buffer.buffer, k=self.backprop_steps)

        for state, probs, reward in batches:
            assert reward in {-1, 1}, f"Got reward {reward} instead of {-1, 1} for state {state}"
            reward = torch.FloatTensor([reward])
            self.optimizer.zero_grad()
            p, v = self.evaluator.evaluate(state, torch_output=True, check_game_over=False)
            check_nan(p)
            check_nan(v)
            check_nan(reward)
            loss = v_loss(reward, v)
            if probs is not None:
                probs = torch.FloatTensor(probs)
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


def check_nan(data):
    """
    >>> check_nan(torch.Tensor([1, 2, 3]))
    >>> check_nan(np.array([1, 2, 3]))
    >>> check_nan(None)
    >>> check_nan(torch.Tensor([1, float('nan'), 3]))
    Traceback (most recent call last):
       ...
    RuntimeError: NaN detected in: tensor([1., nan, 3.])
    >>> check_nan(np.array([1, float('nan'), 3]))
    Traceback (most recent call last):
       ...
    RuntimeError: NaN detected in: [ 1. nan  3.]
    """
    if data is None:
        return
    if isinstance(data, Tensor) and not torch.isnan(data).any():
        return
    if isinstance(data, np.ndarray) and not np.isnan(data).any():
        return
    raise RuntimeError(f"NaN detected in: {data}")


class ReplayBuffer:
    """
    Replay buffer stored in disk
    """

    def __init__(self, path: Path = None, n_pawns: int = None):
        if n_pawns:
            path = DATA_PATH / "deep_q_learning" / f"replay_buffer_{n_pawns}.json"
        if path is None:
            raise ValueError("path must be specified")
        self.path = Path(path)
        self.buffer = []
        self.load()

    def __repr__(self):
        return f"{len(self.buffer)} samples @ {self.path}"

    def load(self):
        if self.path.exists():
            self.buffer = json.load(open(self.path, 'r'))

    def append(self, sample):
        self.buffer.append(sample)

    def __add__(self, other):
        self.buffer += other
        return self

    def save(self):
        shutil.copy(self.path, self.path.with_suffix('.bak'))
        json.dump(self.buffer, open(self.path, 'w'))

    def pretty_save(self):
        text = '\n'.join(map(str, self.buffer))
        with open(self.path.with_suffix('.txt'), 'w') as f:
            f.write(text)
