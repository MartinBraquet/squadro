import contextlib
import json
import random
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
from IPython.core.display_functions import display
from matplotlib import pyplot as plt

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.montecarlo_agent import MonteCarloDeepQLearningAgent
from squadro.core.benchmarking import Benchmark
from squadro.evaluators.evaluator import Model
from squadro.state.evaluators.rl import DeepQLearningEvaluator
from squadro.tools.basic import dict_factory, check_nan
from squadro.tools.constants import DefaultParams, inf, DQL_PATH
from squadro.tools.dates import get_now
from squadro.tools.disk import dump_pickle, mkdir, load_pickle
from squadro.tools.elo import Elo
from squadro.tools.logs import training_logger as logger
from squadro.tools.notebooks import is_notebook
from squadro.tools.probabilities import get_entropy
from squadro.tools.state import get_reward
from squadro.training.buffer import ReplayBuffer

RESULTS_PATH = DQL_PATH / "results"


class Results:
    def __init__(self, data=None):
        self._data = data or dict_factory()

    def __getitem__(self, item):
        return self._data[item]

    @property
    def data(self):
        return self._data

    @classmethod
    def load(cls, path):
        results = load_pickle(path, raise_error=False)
        if isinstance(results, dict):
            results = cls(results)
        if not results:
            results = cls()
        return results

    @property
    def checkpoint_eval(self):
        return self._data['eval']['checkpoint']

    @property
    def self_play_win_rates(self):
        return self._data['self_play_win_rate']

    @property
    def backprop_losses(self):
        return self._data['backprop_loss']

    @property
    def diversity_history(self):
        return self._data['diversity_history']

    @diversity_history.setter
    def diversity_history(self, value):
        self._data['diversity_history'] = value

    @property
    def elo(self) -> Elo:
        if 'elo' not in self._data:
            self._data['elo'] = Elo()
        return self._data['elo']

    @property
    def game_count(self):
        if 'game_step' not in self._data:
            self._data['game_step'] = 1
        return self._data['game_step']

    def set_game_count(self, game_count: int):
        self._data['game_step'] = game_count

    def step_self_play_game(self):
        self._data['game_step'] += 1


class _Base:
    """
    Base class containing results routines
    """

    def __init__(self, results):
        self.results = results

    @property
    def backprop_losses(self):
        return self.results.backprop_losses

    @property
    def game_count(self):
        return self.results.game_count

    def set_game_count(self, game_count: int):
        self.results.set_game_count(game_count)

    @property
    def checkpoint_eval(self):
        return self.results.checkpoint_eval

    @property
    def elo(self) -> Elo:
        return self.results.elo

    def step_self_play_game(self):
        self.results.step_self_play_game()

    @property
    def self_play_win_rates(self):
        return self.results.self_play_win_rates


class Plotter(_Base):
    def __init__(
        self,
        title: str,
        path: str,
        display_plot: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.path = path
        self.display_plot = display_plot

        self._display_handle, self._fig, self._ax = None, None, None

    def init(self):
        self._fig, self._ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        self._ax = self._ax.flatten()
        self._fig.suptitle(self.title)
        if self.display_plot:
            if is_notebook():
                self._display_handle = display(self._fig, display_id=True)
            else:
                plt.ion()
        else:
            matplotlib.use('Agg')

    def close(self):
        self.dump()
        if self.display_plot:
            if is_notebook():
                plt.close(self._fig)
            else:
                plt.ioff()
                plt.show(block=False)

    def dump(self):
        plt.savefig(self.path)

    def _display_plot(self):
        if not self.display_plot:
            return

        if is_notebook():
            self._display_handle.update(self._fig)
        else:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)  # Needed to refresh the figure

    def update(self, name: str):
        if name == 'backprop':
            self._plot_replay_buffer_diversity()
            self._plot_loss()
        elif name == 'all':
            self._plot_replay_buffer_diversity()
            self._plot_loss()
            self._plot_win_rate()
            self._plot_elo()
        elif name == 'eval':
            self._plot_win_rate()
            self._plot_elo()

    def _plot_replay_buffer_diversity(self):
        ax = self._ax[0]
        ax.clear()

        d = self.results.self_play_win_rates
        if d:
            x, y = zip(*d.items())
            ax.plot(x, [.5] * len(x), linestyle='--', color="grey", alpha=.2)
            ax.plot(x, y, 'y', alpha=.6, label="Self-play Win")

        d = self.results.diversity_history
        if d:
            x, y = zip(*d.items())
            ax.plot(x, y, 'k', label="Replay Diversity")

        # ax_win.set_title("")
        # ax.set_xlabel("Epoch")
        # ax_win.set_ylabel("Replay Diversity")
        ax.set_ylim(0, 1.01)
        if ax.get_lines():
            ax.legend(loc='lower left')
        self._display_plot()

    def _plot_loss(self):
        ax = self._ax[1]
        ax.clear()
        labels = {'total': "Total", 'p': "Policy", 'v': "Value"}
        colors = {'total': "k", 'p': "b", 'v': 'g'}
        count = 0
        for name in ('p', 'v', 'total'):
            d = self.results.backprop_losses[name]
            if not d:
                continue
            x, y = zip(*d.items())
            if name == 'total':
                y = np.array(y) / count
            if isinstance(y[0], np.ndarray):
                y = np.array(y).T.tolist()
            else:
                y = [y]
            if len(y) > 1 and name == 'v':
                color = ["y--", "r--"]
            else:
                color = [colors[name]]
            for i in range(len(y)):
                label = labels[name]
                if len(y) > 1:
                    label += f" {i}"
                elif name == 'total':
                    label += f" / {count}"
                ax.plot(
                    x, y[i], color[i],
                    label=label,
                    alpha=1 if name == 'total' else .5,
                )
                count += 1
        # ax_loss.set_title("Loss")
        # ax_loss.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if ax.get_lines():
            ax.legend(loc='lower left')
        self._display_plot()

    def _plot_win_rate(self):
        ax = self._ax[2]
        ax.clear()

        d = self.results.checkpoint_eval

        if not d:
            return

        x, y_all = zip(*d.items())

        ax.plot(x, [.5] * len(x), linestyle='--', color="grey", alpha=.2)

        keys = y_all[0].keys()
        labels = {
            'total': "Total",
            (0, 0): "As P0 (starting)",
            (0, 1): "As P0 (not starting)",
            (1, 0): "As P1 (not starting)",
            (1, 1): "As P1 (starting)",
        }
        colors = {
            'total': "k",
            (0, 0): "y",
            (0, 1): "y--",
            (1, 0): "r",
            (1, 1): "r--",
        }
        for k in list(keys):
            y = [_[k] for _ in y_all]
            ax.plot(
                x, y, colors.get(k),
                label=labels.get(k, k),
                alpha=1 if k == 'total' else .5,
            )

        # ax_win.set_title("Win Rate against Checkpoint")
        # ax_win.set_xlabel("Epoch")
        ax.set_ylabel("Win Rate vs Checkpoint")
        ax.set_ylim(0, 1.01)
        ax.legend(loc='lower left')
        self._display_plot()

    def _plot_elo(self):
        ax = self._ax[3]
        ax.clear()
        x, y = zip(*self.results.elo.history.items())
        ax.plot(x, y, 'k')

        # ax_win.set_title("")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Elo")
        self._display_plot()


class BackPropagation(_Base):
    """
    A class that implements the Backpropagation algorithm for training neural
    networks.

    Backpropagation is a supervised learning algorithm that uses a
    gradient descent approach to minimize the error of a prediction by
    propagating backwards through the layers of a network. It allows weights
    and biases to be adjusted to optimize the learning process and ultimately
    improve the model's predictive performance.
    """

    def __init__(
        self,
        adaptive_lr,
        adaptive_sampling,
        freeze_backprop,
        backprop_games,
        lambda_entropy,
        min_lr_game_count,
        lr,
        min_lr,
        backprop_per_game,
        evaluator,
        backprop_interval,
        self_play_games,
        replay_buffer,
        n_pawns,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adaptive_lr = adaptive_lr
        self.adaptive_sampling = adaptive_sampling
        self.freeze_backprop = freeze_backprop
        self.backprop_games = backprop_games
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer
        self.n_pawns = n_pawns

        self._v_loss = torch.nn.MSELoss(reduction='none')
        self._base_p_loss = torch.nn.CrossEntropyLoss()

        self.min_lr_game_count = min_lr_game_count or 5000
        if self_play_games:
            self.min_lr_game_count = min(self_play_games, self.min_lr_game_count)

        self.lambda_entropy = lambda_entropy if lambda_entropy is not None else .5
        self.entropy_temp = self.min_lr_game_count
        self._max_entropy = np.log(self.n_pawns)

        backprop_per_game = backprop_per_game or self.n_pawns ** 3
        self.backprop_games = backprop_games or backprop_interval * backprop_per_game
        backprop_per_game = int(self.backprop_games / backprop_interval)

        lr = lr or 1e-3
        self.min_lr = min_lr or max(lr / 10., 1e-4)

        self._optimizer = [
            torch.optim.Adam(
                params=self.get_model(player=i).parameters(),
                lr=lr,
                weight_decay=1e-4,
            )
            for i in range(self.n_networks)
        ]

        self._scheduler = None
        if self.min_lr != lr:
            game_count = self.game_count - 1
            self._scheduler = []
            for player in range(self.n_networks):
                if game_count < self.min_lr_game_count:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.get_optimizer(player=player),
                        eta_min=self.min_lr,
                        T_max=int(self.min_lr_game_count * backprop_per_game / self.n_networks),
                    )
                    scheduler.last_epoch = int(game_count * backprop_per_game / self.n_networks)
                    warnings.filterwarnings(
                        "ignore",
                        message="To get the last learning rate computed by the scheduler",
                        category=UserWarning,
                    )
                    self.set_lr(scheduler.get_lr()[0], player=player)
                    self._scheduler.append(scheduler)
                else:
                    self.set_lr(self.min_lr, player=player)

        self._lr_tweak = None
        self._player_losses = None

    def run(self, self_play_win_rate):
        """
        Update the Q-network based on the reward obtained at the end of the game.
        """

        lrs = ', '.join([f"{self.get_lr(i):.1e}" for i in range(self.n_networks)])
        logger.info(f"lr: {lrs}")

        self._train()

        batches = []
        for w, f, data in self.replay_buffer.iter_buckets():
            if self.adaptive_sampling:
                win_rate = self_play_win_rate[f]
                if w == 1:
                    win_rate = 1. - win_rate
            else:
                win_rate = .5
            if np.isnan(win_rate):
                logger.warn(f"win_rate is NaN for {w=}, {f=}. To Fix.")
                win_rate = .5
            k = (1. - win_rate) * (self.backprop_games // 2)
            k = min(max(round(k), 1), len(data))
            batch = random.sample(data, k=k)
            # print(len(batch), '/', len(data))
            if batch:
                batches.append(batch)
                # logger.info(f"{w=}, {f=}, {len(batch)} samples")

        batches = [s for t in batches for s in t]
        random.shuffle(batches)
        # print(len(batches))

        if self.separate_networks:
            # sort by current player
            batches_by_player = {0: [], 1: []}
            for state, probs, winner in batches:
                cur_player = state[1]
                batches_by_player[cur_player] += [(state, probs, winner)]
        else:
            batches_by_player = {0: batches}

        for player, ba in batches_by_player.items():
            logger.info(f"Backprop batch length for player {player}: {len(ba)}")

        self._player_losses = []
        losses, p_losses, v_losses = [], [], []
        for player, ba in batches_by_player.items():
            # model = next(self.get_model(player).parameters())[0][0].clone()
            # other_model = next(self.get_model(1 - player).parameters())[0][0].clone()
            l, p, v = self._run_batch(batches=ba, player=player)
            # model_updated = next(self.get_model(player).parameters())[0][0]
            # other_model_updated = next(self.get_model(1 - player).parameters())[0][0]
            # print('Should update', model, model_updated, sep='\n')
            # print('Should not update', other_model, other_model_updated, sep='\n')
            losses += l
            p_losses += p
            v_losses += v

        if self.adaptive_lr and self.separate_networks:
            self._lr_tweak = self._player_losses[0] / self._player_losses[1]

        loss = float(np.mean(losses))
        p_loss = float(np.mean(p_losses))
        v_loss = np.mean(np.array(v_losses), axis=0)

        self.backprop_losses['total'][self.game_count] = loss
        self.backprop_losses['p'][self.game_count] = p_loss
        self.backprop_losses['v'][self.game_count] = v_loss

        v_txt = ', '.join([f"v{i}: {v:.2f}" for i, v in enumerate(v_loss)])
        logger.info(f"Backprop loss: {loss:.2f} (p: {p_loss:.2f}, {v_txt})")

    def _run_batch(self, batches: list, player: int):
        if self.freeze_backprop == player:
            logger.info(f"Skipping backprop for player {player} (freeze)")
            return [], [], []

        return_all = self.model_config.double_value_head

        losses, p_losses, v_losses, entropies = [], [], [], []
        for state, probs, winner in batches:
            assert winner in {0, 1}, f"Got {winner=} instead of {0, 1} for state {state}"
            cur_player = state[1]
            if self.separate_networks:
                assert cur_player == player, f"Got wrong player for {state}"
            reward = get_reward(winner=winner, return_all=return_all, cur_player=cur_player)
            if not isinstance(reward, np.ndarray):
                reward = [reward]
            reward = torch.FloatTensor(reward).to(self.device)
            self._zero_grad()
            p, v = self.evaluator.evaluate(
                state=state,
                torch_output=True,
                check_game_over=False,
                return_all=return_all,
            )
            check_nan(p)
            check_nan(v)
            v_loss = self._v_loss(reward, v)
            v_losses += [v_loss.cpu().detach().numpy()]
            loss = v_loss.sum()
            if probs is not None:
                p_loss, entropy = self._p_loss(p, probs)
                p_losses += [p_loss.item()]
                loss += p_loss
                entropies.append(entropy)
            check_nan(loss)
            losses += [loss.item()]
            loss.backward()
            with self._tweak_lr(player):
                self.get_optimizer(player).step()
            self._step_lr(player)

        self._train(mode=False)

        entropies = torch.Tensor(entropies).mean()
        logger.info(
            f"Entropy: {entropies:.2f} (theoretical max: {self._max_entropy:.2f})"
            f", lambda: {self._get_lambda_entropy():.2f}"
        )

        if self.separate_networks:
            loss = float(np.mean(losses))
            self._player_losses.append(loss)
            logger.info(f"Backprop loss for player {player}: {loss :.2f}")

        return losses, p_losses, v_losses

    @property
    def device(self):
        return self.get_model().device

    def _get_lambda_entropy(self):
        decay_rate = 0.1
        lambda_t = self.lambda_entropy * (decay_rate ** (self.game_count / self.entropy_temp))
        return lambda_t

    def _step_lr(self, player: int):
        scheduler = self.get_scheduler(player)
        if scheduler and scheduler.last_epoch < scheduler.T_max:
            scheduler.step()
            logger.debug(self.get_lr(player))

    @contextlib.contextmanager
    def _tweak_lr(self, player: int):
        if self.adaptive_lr and self._lr_tweak:
            base_lr = self.get_lr(player)
            tweak = self._lr_tweak if player == 0 else 1. / self._lr_tweak
            tweak = min(1.5, tweak)
            self.set_lr(base_lr * tweak, player=player)
            logger.debug(f"Player {player} lr tweak: {tweak:.2f}, lr: {self.get_lr(player):.1e}")
            yield
            self.set_lr(base_lr, player=player)
            logger.debug(f"Player {player} back to lr: {self.get_lr(player):.1e}")
        else:
            yield

    def _zero_grad(self):
        for i in range(self.n_networks):
            self.get_optimizer(i).zero_grad()

    def _train(self, mode=True):
        for i in range(self.n_networks):
            self.get_model(i).train(mode=mode)

    def _p_loss(self, p, probs):
        probs = torch.FloatTensor(probs).to(self.device)
        lambda_entropy = self._get_lambda_entropy()
        entropy = get_entropy(probs)
        p_loss = self._base_p_loss(p, probs) + lambda_entropy * (self._max_entropy - entropy.sum())
        return p_loss, entropy

    @property
    def lr(self):
        return self.get_lr()

    def get_lr(self, player=0) -> float:
        return self.get_optimizer(player).param_groups[0]["lr"]

    def set_lr(self, lr, player=0) -> None:
        for param_group in self.get_optimizer(player).param_groups:
            if isinstance(param_group["lr"], torch.Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr

    def get_optimizer(self, player=0) -> torch.optim.Optimizer:
        return self._optimizer[player]

    def get_scheduler(self, player=0) -> Optional[torch.optim.lr_scheduler.CosineAnnealingLR]:
        if self._scheduler:
            return self._scheduler[player]
        return None

    @property
    def model_config(self):
        return self.evaluator.model_config

    @property
    def n_networks(self) -> int:
        return 2 if self.separate_networks else 1

    @property
    def separate_networks(self):
        return self.model_config.separate_networks

    def get_model(self, player=0) -> Model:
        return self.evaluator.get_model(n_pawns=self.n_pawns, player=player)


class Benchmarker(_Base):
    """
    Benchmark the trained model against baselines.
    """

    def __init__(
        self,
        eval_games,
        evaluator,
        n_pawns,
        agent_kwargs,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_games = max(eval_games or 100, 4)
        self.n_pawns = n_pawns
        self.agent_kwargs = agent_kwargs
        self.evaluator = evaluator
        self.evaluator_chkpt = DeepQLearningEvaluator(
            model_path=Path(self.evaluator.model_path) / 'checkpoint',
            model_config=self.evaluator.model_config,
        )
        self._opponents = ['checkpoint', 'random']

        if not self.evaluator_chkpt.is_pretrained(n_pawns=self.n_pawns):
            self.copy_weights_to_checkpoint()

    def run(self):
        self.evaluate_agents()
        self.update_checkpoint_model()

    def evaluate_agents(self):
        vs_list = self._opponents
        results = {}
        for vs in vs_list:
            self.results['eval'][str(vs)][self.game_count] = win_rate = self.evaluate_agent(vs=vs)
            win_rate = win_rate['total']
            results[str(vs)] = win_rate

        self.elo.update(win_rate=results['checkpoint'], n=self.eval_games, step=self.game_count)

        msg = f"{self.game_count} self-play games: "
        msg += ', '.join([f"{v:.0%} vs {k}" for k, v in results.items()])
        logger.info(msg)

    def evaluate_agent(self, vs: str | Agent = None) -> dict:
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
        eval_games = self.eval_games
        if vs != 'checkpoint':
            eval_games /= 2

        agent = MonteCarloDeepQLearningAgent(
            evaluator=self.evaluator,
            is_training=vs != 'random',
            **self.agent_kwargs,
        )

        if vs == 'checkpoint':
            vs = MonteCarloDeepQLearningAgent(
                evaluator=self.evaluator_chkpt,
                is_training=True,
                **self.agent_kwargs,
            )
        elif vs is None:
            vs = agent

        logger.info(f"Evaluation against {vs}:")

        benchmark = Benchmark(
            agent_0=agent,
            agent_1=vs,
            n_pawns=self.n_pawns,
            n_games=eval_games,
        )
        win_rate = benchmark.run()
        win_rate_split = benchmark.win_rates.copy()
        win_rate_split['total'] = win_rate

        return win_rate_split

    def update_checkpoint_model(self):
        if self.checkpoint_eval[self.game_count]['total'] > .7:
            logger.info(f"Updating best checkpoint")
            self.copy_weights_to_checkpoint()
            self.evaluator_chkpt.dump()
            self.elo.update_checkpoint()

    def copy_weights_to_checkpoint(self):
        self.evaluator_chkpt.load_weights(self.evaluator)

    def get_model_chkpt(self, player=0) -> Model:
        return self.evaluator_chkpt.get_model(n_pawns=self.n_pawns, player=player)


class SelfPlayer(_Base):
    def __init__(
        self,
        agent,
        n_pawns,
        replay_buffer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent = agent
        self.n_pawns = n_pawns
        self.replay_buffer = replay_buffer

        self.win_rate = ...
        self.clear_win_rate()

    def run(self):
        self.step_self_play_game()

        game = Game(
            n_pawns=self.n_pawns,
            agent_0=self.agent,
            agent_1=self.agent,
            save_states=True,
        )
        game.run()

        history = []
        for i, s in enumerate(game.state_history):
            s = s.to_list()

            move_probs = game.move_info[i]['mcts_move_probs'] if i < len(game.move_info) else None
            check_nan(move_probs)
            if isinstance(move_probs, np.ndarray):
                move_probs = move_probs.tolist()

            history.append((
                s,
                move_probs,
                game.winner,
            ))

        # probs = game.move_info[0]['mcts_move_probs']
        # probs = (100 * np.array(probs)).round().tolist()
        # logger.info(f"Initial state MCTS move probs: {probs}")

        self.replay_buffer.add_game(history=history, winner=game.winner, first=game.first)
        self.win_rate[game.first] += [1 - game.winner]

        return history

    def clear_win_rate(self):
        self.win_rate = {0: [], 1: []}

    def compute_info(self):
        self.update_diversity_ratio()

        win_rate = []
        win_rate_split = {}
        for f, data in self.win_rate.items():
            win_rate_split[f] = np.array(data).mean() if len(data) > 0 else 0.5
            win_rate += data
        self.self_play_win_rates[self.game_count] = win_rate = np.array(win_rate).mean()
        logger.info(f"Self-play win rate: {win_rate:.0%}"
                    f" (first 0: {win_rate_split[0]:.0%}, first 1: {win_rate_split[1]:.0%})")
        self.clear_win_rate()
        return win_rate_split

    def update_diversity_ratio(self):
        self.replay_buffer.compute_diversity_ratio(self.game_count)
        self.results.diversity_history = self.replay_buffer.diversity_history


class DeepQLearningTrainer(_Base):
    """
    Deep Q-learning trainer.
    """

    def __init__(
        self,
        n_pawns=None,
        lr=None,
        min_lr=None,
        min_lr_game_count=None,
        adaptive_lr=True,
        lambda_entropy=None,
        self_play_games=None,
        backprop_interval=None,
        backprop_games=None,
        backprop_per_game=None,
        eval_interval=None,
        eval_games=None,
        model_path=None,
        model_config=None,
        init_from=None,
        mcts_kwargs=None,
        adaptive_sampling=True,
        freeze_backprop=None,
        display_plot=True,
        device=None,
    ):
        """
        Represents a class for initializing and training a deep reinforcement learning agent
        using Monte Carlo and Q-Learning methodologies. The class manages model training,
        evaluation, adaptive sampling, learning rate adjustments, and results logging.

        Attributes:
            n_pawns (int): Number of pawns in the game. Defaults to DefaultParams.n_pawns.
            lr (float): Initial learning rate for the optimizer.
            min_lr (float): Minimum learning rate when using a scheduler.
            min_lr_game_count (int): Final step for learning rate adjustment.
            adaptive_lr (bool): Whether to enable adaptive learning rate.
            lambda_entropy (float): Entropy regularization parameter to balance exploration
                with exploitation.
            self_play_games (int): Total number of training steps to execute.
            backprop_interval (int): Interval in steps for executing backpropagation.
            backprop_games (int): Total number of backpropagation steps.
            backprop_per_game (int): Number of backpropagation operations to be performed per game.
            eval_interval (int): Interval in steps for running evaluations.
            eval_games (int): Number of steps for evaluation during training.
            model_path (Path): Path to save or load the model checkpoint.
            model_config (dict): Configuration dictionary defining the architecture of the model.
            init_from (str): Indicator for initializing models from scratch or checkpoint.
            mcts_kwargs (dict): Keyword arguments for configuring Monte Carlo Tree Search (MCTS).
            adaptive_sampling (bool): Whether to enable adaptive sampling in training.
            freeze_backprop (bool): Whether to disable backpropagation altogether.
            display_plot (bool): Whether to plot results during training.
        """
        self.n_pawns = n_pawns or DefaultParams.n_pawns
        self.self_play_games = self_play_games
        self.backprop_interval = backprop_interval or 100
        self.eval_interval = eval_interval or 500

        self.mcts_kwargs = mcts_kwargs or {}
        self.mcts_kwargs.setdefault('max_steps', int(4 * self.n_pawns ** 2))

        self.agent_kwargs = dict(
            mcts_kwargs=self.mcts_kwargs,
            max_time_per_move=inf,
            device=device,
        )

        self.agent = MonteCarloDeepQLearningAgent(
            model_path=model_path,
            model_config=model_config,
            is_training=True,
            **self.agent_kwargs,
        )

        self.results_path = self.model_path / 'results'
        mkdir(self.results_path)
        logger.info(f'Results will be saved in {self.results_path}')

        from_scratch = init_from == 'scratch'

        results = Results() if from_scratch else Results.load(self.pkl_results_path)
        super().__init__(results=results)

        self.replay_buffer = ReplayBuffer(
            path=self.model_path / 'replay_buffer.pkl',
            max_size=4e3 * (self.n_pawns ** 2),
        )

        if from_scratch:
            self.evaluator.erase(self.n_pawns)
            self.replay_buffer.clear()

        self.self_player = SelfPlayer(
            agent=self.agent,
            results=self.results,
            n_pawns=self.n_pawns,
            replay_buffer=self.replay_buffer,
        )

        self.back_propagation = BackPropagation(
            adaptive_lr=adaptive_lr,
            adaptive_sampling=adaptive_sampling,
            freeze_backprop=freeze_backprop,
            backprop_games=backprop_games,
            lambda_entropy=lambda_entropy,
            min_lr_game_count=min_lr_game_count,
            lr=lr,
            min_lr=min_lr,
            backprop_per_game=backprop_per_game,
            evaluator=self.evaluator,
            backprop_interval=self.backprop_interval,
            self_play_games=self.self_play_games,
            replay_buffer=self.replay_buffer,
            results=self.results,
            n_pawns=self.n_pawns,
        )

        self.benchmarker = Benchmarker(
            results=self.results,
            eval_games=eval_games,
            evaluator=self.evaluator,
            n_pawns=self.n_pawns,
            agent_kwargs=self.agent_kwargs,
        )

        self.plotter = Plotter(
            title=self.title,
            display_plot=display_plot,
            path=self.results_path / 'plots.png',
            results=self.results,
        )

        self.run_ts = None

    def run(self) -> None:
        """
        Run the training loop.
        """
        self.run_ts = get_now()
        logger.info(
            f"Starting training: {self.n_pawns} pawns\n"
            f"model size: {self.get_model().byte_size()}\n"
            f"model config: {self.model_config}\n"
            f"optimizer: {self.back_propagation.get_optimizer()}\n"
            f"scheduler: {self.back_propagation.get_scheduler().__class__.__name__}"
            f" (min lr: {self.back_propagation.min_lr:.0e})\n"
            f"game count: {self.game_count}\n"
            f"max games: {self.self_play_games}\n"
            f"backprop interval: {self.backprop_interval}\n"
            f"backprop steps: {self.back_propagation.backprop_games}\n"
            f"eval interval: {self.eval_interval}\n"
            f"eval steps: {self.benchmarker.eval_games}\n"
            f"MCTS config: {json.dumps(self.agent.mcts_config, indent=4)}\n"
            f"device: {self.device}\n"
            f"path: {self.model_path}\n"
            f"replay buffer: {self.replay_buffer}\n"
            f"run ts: {self.run_ts}\n"
        )

        self.plotter.init()

        try:
            self._run()
        finally:
            self.dump()
            self.plotter.close()

        logger.info("Training finished.")

    def _run(self):
        self.plot('all')

        for game_count in range(self.game_count, (self.self_play_games or int(1e15)) + 1):
            self.generate_training_samples()
            logger.info(f'Self-play game: {game_count}')

            if game_count % self.backprop_interval == 0:
                self.backpropagate()

            if game_count % self.eval_interval == 0:
                self.evaluate()

    def generate_training_samples(self):
        self.self_player.run()

    def backpropagate(self):
        self_play_win_rate = self.self_player.compute_info()
        self.back_propagation.run(self_play_win_rate=self_play_win_rate)
        self.plot('backprop')

    def evaluate(self):
        self.benchmarker.run()
        self.plot('eval')
        self.dump()

    def plot(self, name):
        self.plotter.update(name)

    def dump(self):
        # with logger.context_info('dump'):
        filename = get_now()
        self.dump_results()
        logger.dump_history(self.results_path / f'logs.txt', clear=True)
        self.plotter.dump()
        self.evaluator.dump()
        self.evaluator.dump(Path(self.benchmarker.evaluator_chkpt.model_path) / filename)
        self.replay_buffer.save()
        logger.info(f"Dumped current model to '{filename}'")

    def dump_results(self, fmt='pkl'):
        if fmt == 'json':
            raise ValueError("JSON format is not supported for results.")
            # result = {str(k): v for k, v in self.results.items()}
            # json.dump(result, open(self.pkl_results_path.replace('.pkl', '.json'), 'w'))
        elif fmt == 'pkl':
            dump_pickle(self.results, self.pkl_results_path)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    @property
    def device(self):
        return self.get_model().device

    @property
    def pkl_results_path(self):
        return self.results_path / 'results.pkl'

    def get_model(self, player=0) -> Model:
        return self.evaluator.get_model(n_pawns=self.n_pawns, player=player)

    @property
    def model_config(self):
        return self.evaluator.model_config

    @property
    def model_path(self):
        return self.evaluator.model_path

    @property
    def evaluator(self) -> DeepQLearningEvaluator:
        return self.agent.evaluator  # noqa

    @property
    def title(self):
        ll = []
        l = []
        count = 0
        config = self.agent.mcts_config
        for k, v, in config.items():
            l.append(f"{k}={v}")
            count += 1
            if count % 6 == 0 or count == len(config):
                ll.append(", ".join(l))
                l = []
        mcts_info = '\n'.join(ll)
        return (
            r"$\mathbf{Deep\ Q-Learning\ Training}$"
            f" ({self.n_pawns} pawns)"
            f"\n{mcts_info}"
            f"\n{self.model_config}, lr={self.back_propagation.lr:.0e}"
        )

    def get_lr(self, player=0):
        return self.back_propagation.get_lr(player=player)
