import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.montecarlo_agent import MonteCarloDeepQLearningAgent
from squadro.benchmarking import benchmark
from squadro.evaluators.evaluator import DeepQLearningEvaluator, Model
from squadro.state import State
from squadro.tools.constants import DefaultParams, inf, DATA_PATH
from squadro.tools.dates import get_now
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
        model_config=None,
        init_from=None,
        mcts_kwargs=None,
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
        self.eval_steps = max(eval_steps or int(100), 4)
        self.mcts_kwargs = mcts_kwargs or {}
        self.mcts_kwargs.setdefault('max_steps', int(1.3 * self.n_pawns ** 3))

        self.backprop_interval = backprop_interval or 100
        self.backprop_steps = backprop_steps or self.backprop_interval * 10

        self.agent = MonteCarloDeepQLearningAgent(
            model_path=model_path,
            model_config=model_config,
            is_training=True,
            **self._agent_kwargs,
        )
        self.evaluator_old = DeepQLearningEvaluator(
            model_path=Path(self.model_path) / "old",
            model_config=model_config,
        )

        lr = lr or 1e-3
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            eta_min=lr / 10,
            T_max=1000 * self.n_pawns ** 2,
        )

        self.replay_buffer = ReplayBuffer(n_pawns=self.n_pawns)

        self.results = dict(backprop_loss={}, eval={})

        self._v_loss = torch.nn.MSELoss()
        self._p_loss = torch.nn.CrossEntropyLoss()

        if init_from == 'scratch':
            self.evaluator.erase(self.n_pawns)
            self.replay_buffer.clear()

    @property
    def _agent_kwargs(self):
        return dict(
            mcts_kwargs=self.mcts_kwargs,
            max_time_per_move=inf,
        )

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
            f"model size: {self.model.byte_size()}\n"
            f"lr: {self.lr}\n"
            f"{self.n_steps} steps\n"
            f"backprop interval: {self.backprop_interval}\n"
            f"backprop steps: {self.backprop_steps}\n"
            f"eval interval: {self.eval_interval}\n"
            f"eval steps: {self.eval_steps}\n"
            f"path: {self.model_path}\n"
            f"replay buffer: {self.replay_buffer}\n"
            f"MCTS steps: {self.agent.max_steps}\n"
        )
        self.evaluator.get_model(n_pawns=self.n_pawns)
        self.evaluator.dump(self.evaluator_old.model_path)
        self.evaluator_old.clear()

        self._run()

    def _run(self):
        filename = get_now()
        if not os.path.exists(path := 'results_eval'):
            os.makedirs(path)
        if not os.path.exists(path_loss := 'results_losses'):
            os.makedirs(path_loss)

        for step in range(1, self.n_steps + 1):
            # print(step)
            self.get_training_samples()

            if step % self.backprop_interval == 0:
                # with logger.context_info('back_propagate'):
                loss_avg = self.back_propagate()
                self.results['backprop_loss'][step] = loss_avg

            if step % self.eval_interval == 0:
                self.evaluate_agents(step)
                if self.results['eval'][step]['checkpoint'] > .6:
                    logger.info(f"Updating best checkpoint")
                    self.model_old.load(self.model)
                with logger.context_info('dump'):
                    json.dump(self.results['eval'], open(f'{path}/{filename}.json', 'w'))
                    json.dump(self.results['backprop_loss'],
                              open(f'{path_loss}/{filename}.json', 'w'))
                    self.evaluator.dump()
                    self.evaluator.dump(Path(self.evaluator_old.model_path) / get_now())
                    self.replay_buffer.save()

        self.evaluator.dump()

    def evaluate_agents(self, step):
        vs_list = [
            'checkpoint',
            'random',
            # AlphaBetaAdvancementDeepAgent(max_depth=5),
        ]
        results = {}
        for vs in vs_list:
            results[str(vs)] = self.evaluate_agent(vs=vs)

        self.results['eval'][step] = results

        msg = f"{step} steps"
        for k, v in results.items():
            msg += f", {v * 100 :.0f}% vs {k}"
        logger.info(msg)

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

    def back_propagate(self):
        """
        Update the Q-network based on the reward obtained at the end of the game.
        """
        self.model.train()
        buffer = self.replay_buffer.data
        batches = random.sample(buffer, k=min(self.backprop_steps, len(buffer)))

        losses = []
        v_losses = []
        p_losses = []
        for state, probs, reward in batches:
            assert reward in {-1, 1}, f"Got reward {reward} instead of {-1, 1} for state {state}"
            reward = torch.FloatTensor([reward]).to(self.model.device)
            self.optimizer.zero_grad()
            p, v = self.evaluator.evaluate(state, torch_output=True, check_game_over=False)
            check_nan(p)
            check_nan(v)
            check_nan(reward)
            loss = self._v_loss(reward, v)
            v_losses += [loss.item()]
            if probs is not None:
                probs = torch.FloatTensor(probs).to(self.model.device)
                p_loss = self._p_loss(p, probs)
                loss += p_loss
                p_losses += [p_loss.item()]
            check_nan(loss)
            losses += [loss.item()]
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        self.model.eval()

        loss_avg = np.mean(losses)
        logger.info(f"Backprop loss: {loss_avg:.2f}"
                    f" (p: {np.mean(p_losses):.2f}, v: {np.mean(v_losses):.2f})")

        return loss_avg

    @property
    def model(self) -> Model:
        return self.evaluator.get_model(n_pawns=self.n_pawns)

    @property
    def model_old(self) -> Model:
        return self.evaluator_old.get_model(n_pawns=self.n_pawns)

    def evaluate_agent(self, vs: str | Agent = None) -> float:
        """
        Evaluate the success rate of the current agent against another agent.

        Note that if the agent is in evaluation mode (`is_training = False`), then it is fully
        deterministic. And two deterministic agents will always have the same result. The only
        randomness would come from who started: 100% if agent 0 wins in either starting position,
        0% if agent 1 wins in either starting position, and close to 50% if agent 0 wins in one
        starting position. This is a very bad measure for benchmark evaluation.

        This is a benchmark evaluation, so we want to compare the two models in many different
        configurations. As the model gets trained, we expect it to have a win rate slowly
        increasing from 50% to 100% against its past checkpoints.
        To do so, we must keep the randomness in each agent's decision by keeping
        `is_training = True`.
        """
        agent = MonteCarloDeepQLearningAgent(
            evaluator=self.evaluator,
            is_training=vs != 'random',
            **self._agent_kwargs,
        )

        if vs == 'checkpoint':
            vs = MonteCarloDeepQLearningAgent(
                evaluator=self.evaluator_old,
                is_training=True,
                **self._agent_kwargs,
            )
        elif vs is None:
            vs = agent

        win_rate = benchmark(
            agent_0=agent,
            agent_1=vs,
            n_pawns=self.n_pawns,
            n=self.eval_steps,
        )

        return win_rate


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
    if isinstance(data, torch.Tensor) and not torch.isnan(data).any():
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
        self.data = []
        self.load()

    def __repr__(self):
        return f"{len(self.data)} samples @ {self.path}"

    def append(self, sample):
        self.data.append(sample)
        self._cap()

    def _cap(self):
        if len(self.data) > 20e3:
            self.data.pop(0)

    def __add__(self, other):
        self.data += other
        self._cap()
        return self

    def load(self):
        if self.path.exists():
            self.data = json.load(open(self.path, 'r'))

    def save(self):
        if self.path.exists():
            shutil.copy(self.path, self.path.with_suffix('.bak'))
        json.dump(self.data, open(self.path, 'w'))

    def pretty_save(self):
        text = '\n'.join(map(str, self.data))
        with open(self.path.with_suffix('.txt'), 'w') as f:
            f.write(text)

    def clear(self):
        self.data = []
        self.save()

    def get_winners(self):
        winners = []
        for s, p, r in self.data:
            if p is None:
                s = State.from_list(s)
                assert s.winner is not None
                winners.append(s.winner)
        return winners
