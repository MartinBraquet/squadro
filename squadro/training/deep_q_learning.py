import contextlib
import json
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from IPython.core.display_functions import display
from matplotlib import pyplot as plt

from squadro import Game
from squadro.agents.agent import Agent
from squadro.agents.montecarlo_agent import MonteCarloDeepQLearningAgent
from squadro.benchmarking import Benchmark
from squadro.evaluators.evaluator import Model
from squadro.evaluators.rl import DeepQLearningEvaluator
from squadro.tools.basic import dict_factory, check_nan
from squadro.tools.constants import DefaultParams, inf
from squadro.tools.dates import get_now
from squadro.tools.disk import dump_pickle, mkdir, load_pickle
from squadro.tools.elo import Elo
from squadro.tools.log import training_logger as logger
from squadro.tools.notebooks import is_notebook
from squadro.tools.state import get_reward
from squadro.training.buffer import ReplayBuffer
from squadro.training.constants import DQL_PATH

EPS = 1e-8

RESULTS_PATH = DQL_PATH / "results"


class DeepQLearningTrainer:
    """
    Deep Q-learning trainer.
    """

    def __init__(
        self,
        n_pawns=None,
        lr=None,
        min_lr=None,
        lr_final_step=None,
        adaptive_lr=True,
        lambda_entropy=None,
        n_steps=None,
        backprop_interval=None,
        backprop_steps=None,
        backprop_per_game=None,
        eval_interval=None,
        eval_steps=None,
        model_path=None,
        model_config=None,
        init_from=None,
        mcts_kwargs=None,
        adaptive_sampling=True,
        freeze_backprop=None,
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

        self.adaptive_sampling = adaptive_sampling
        self.adaptive_lr = adaptive_lr
        self.freeze_backprop = freeze_backprop
        self.lambda_entropy = lambda_entropy if lambda_entropy is not None else .5

        from_scratch = init_from == 'scratch'

        self.n_steps = n_steps
        self.lr_final_step = lr_final_step or 5000
        if n_steps:
            self.lr_final_step = min(n_steps, self.lr_final_step)

        self.eval_interval = eval_interval or 500
        self.eval_steps = max(eval_steps or 100, 4)

        self.mcts_kwargs = mcts_kwargs or {}
        self.mcts_kwargs.setdefault('max_steps', int(1.3 * self.n_pawns ** 3))

        self.backprop_interval = backprop_interval or 100
        backprop_per_game = backprop_per_game or self.n_pawns ** 3
        self.backprop_steps = backprop_steps or self.backprop_interval * backprop_per_game
        backprop_per_game = int(self.backprop_steps / self.backprop_interval)

        self.agent = MonteCarloDeepQLearningAgent(
            model_path=model_path,
            model_config=model_config,
            is_training=True,
            **self._agent_kwargs,
        )
        self.evaluator_chkpt = DeepQLearningEvaluator(
            model_path=Path(self.model_path) / "checkpoint",
            model_config=model_config,
        )

        self.results_path = self.model_path / 'results'
        mkdir(self.results_path)
        logger.info(f'Results will be saved in {self.results_path}')

        self.results = load_pickle(self.pkl_results_path, raise_error=False)
        if not self.results or from_scratch:
            self.results = dict_factory()

        lr = lr or 1e-3
        self.min_lr = min_lr or max(lr / 10, 1e-4)

        self.replay_buffer = ReplayBuffer(
            n_pawns=self.n_pawns,
            path=self.model_path / 'replay_buffer.pkl',
            max_size=8e3 * (self.n_pawns ** 2),
        )

        self._v_loss = torch.nn.MSELoss(reduction='none')
        self._p_loss = torch.nn.CrossEntropyLoss()

        self._opponents = [
            'checkpoint',
            'random',
        ]

        if from_scratch:
            self.evaluator.erase(self.n_pawns)
            self.replay_buffer.clear()

        self._optimizer = [torch.optim.Adam(
            params=self.get_model(player=i).parameters(),
            lr=lr,
            weight_decay=1e-4,
        ) for i in range(self.n_networks)]

        self._scheduler = None
        if self.min_lr != lr:
            epoch = self.get_step() - 1
            self._scheduler = []
            for player in range(self.n_networks):
                if epoch < self.lr_final_step:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.get_optimizer(player=player),
                        eta_min=self.min_lr,
                        T_max=int(self.lr_final_step * backprop_per_game / self.n_networks),
                    )
                    scheduler.last_epoch = int(epoch * backprop_per_game / self.n_networks)
                    warnings.filterwarnings(
                        "ignore",
                        message="To get the last learning rate computed by the scheduler",
                        category=UserWarning,
                    )
                    self.set_lr(scheduler.get_lr()[0], player=player)
                    self._scheduler.append(scheduler)
                else:
                    self.set_lr(self.min_lr, player=player)

        self._display_handle, self._fig, self._ax = None, None, None
        self._self_play_win_rate = None
        self.run_ts = None
        self._player_losses = None
        self._lr_tweak = None

    def get_step(self):
        if 'game_step' not in self.results:
            self.results['game_step'] = 1
        return self.results['game_step']

    @property
    def pkl_results_path(self):
        return self.results_path / 'results.pkl'

    @property
    def _agent_kwargs(self):
        return dict(
            mcts_kwargs=self.mcts_kwargs,
            max_time_per_move=inf,
        )

    @property
    def n_networks(self) -> int:
        return 2 if self.separate_networks else 1

    @property
    def separate_networks(self):
        return self.model_config.separate_networks

    def get_optimizer(self, player=0) -> torch.optim.Optimizer:
        return self._optimizer[player]

    def get_scheduler(self, player=0) -> Optional[torch.optim.lr_scheduler.CosineAnnealingLR]:
        if self._scheduler:
            return self._scheduler[player]
        return None

    def get_model(self, player=0) -> Model:
        return self.evaluator.get_model(n_pawns=self.n_pawns, player=player)

    def get_model_chkpt(self, player=0) -> Model:
        return self.evaluator_chkpt.get_model(n_pawns=self.n_pawns, player=player)

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
            f"\n{self.model_config}, lr={self.lr:.0e}"
        )

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

    def run(self) -> None:
        """
        Run the training loop.
        """
        self.run_ts = get_now()
        logger.info(
            f"Starting training: {self.n_pawns} pawns\n"
            f"model size: {self.get_model().byte_size()}\n"
            f"model config: {self.model_config}\n"
            f"optimizer: {self.get_optimizer()}\n"
            f"scheduler: {self.get_scheduler().__class__.__name__} (min lr: {self.min_lr:.0e})\n"
            f"epoch: {self.get_step()}\n"
            f"max games: {self.n_steps}\n"
            f"backprop interval: {self.backprop_interval}\n"
            f"backprop steps: {self.backprop_steps}\n"
            f"eval interval: {self.eval_interval}\n"
            f"eval steps: {self.eval_steps}\n"
            f"MCTS config: {json.dumps(self.agent.mcts_config, indent=4)}\n"
            f"device: {self.device}\n"
            f"path: {self.model_path}\n"
            f"replay buffer: {self.replay_buffer}\n"
            f"run ts: {self.run_ts}\n"
        )
        if not self.evaluator_chkpt.is_pretrained(n_pawns=self.n_pawns):
            self.copy_weights_to_checkpoint()

        self._open_figure()

        try:
            self._run()
        finally:
            self.dump()
            self._close_figure()

        logger.info("Training finished.")

    def copy_weights_to_checkpoint(self):
        self.evaluator_chkpt.load_weights(self.evaluator, n_pawns=self.n_pawns)

    def _run(self):
        self._plot_replay_buffer_diversity()
        self._plot_loss()
        self._plot_win_rate()
        self._plot_elo()

        self._clear_self_play_win_rate()

        n_steps = self.n_steps or int(1e15)
        for step in range(self.get_step(), n_steps + 1):
            self.results['game_step'] += 1
            # print('Step:', step)
            self.get_training_samples()

            if step % self.backprop_interval == 0:
                self._process_self_play_info(step)
                self._back_propagate(step)
                self._plot_loss()
                self.replay_buffer.get_diversity_ratio(step)
                self._plot_replay_buffer_diversity()
                self._clear_self_play_win_rate()

            if step % self.eval_interval == 0:
                self.evaluate_agents(step)
                self.update_checkpoint_model(step)
                self._plot_win_rate()
                self._plot_elo()
                self.dump()

    def _open_figure(self):
        self._fig, self._ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        self._ax = self._ax.flatten()
        self._fig.suptitle(self.title)
        if is_notebook():
            self._display_handle = display(self._fig, display_id=True)
        else:
            plt.ion()

    def _close_figure(self):
        self._save_figure()
        if is_notebook():
            plt.close(self._fig)
        else:
            plt.ioff()
            plt.show(block=False)

    def _save_figure(self):
        plt.savefig(self.results_path / 'plots.png')

    def update_checkpoint_model(self, step):
        if self.results['eval']['checkpoint'][step]['total'] > .7:  # noqa
            logger.info(f"Updating best checkpoint")
            self.copy_weights_to_checkpoint()
            self.evaluator_chkpt.dump()
            self.elo.update_checkpoint()

    def dump(self):
        # with logger.context_info('dump'):
        self.dump_results()
        logger.dump_history(self.results_path / f'logs.txt', clear=True)
        self._save_figure()
        self.evaluator.dump()
        filename = get_now()
        self.evaluator.dump(Path(self.evaluator_chkpt.model_path) / filename)
        self.replay_buffer.save()
        logger.info(f"Dumped current model to '{filename}'")

    def dump_results(self, fmt='pkl'):
        if fmt == 'json':
            result = {str(k): v for k, v in self.results.items()}
            json.dump(result, open(self.pkl_results_path.replace('.pkl', '.json'), 'w'))
        elif fmt == 'pkl':
            dump_pickle(self.results, self.pkl_results_path)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _clear_self_play_win_rate(self):
        self._self_play_win_rate = {0: [], 1: []}

    def _process_self_play_info(self, step):
        win_rate = []
        win_rate_split = {}
        for f, data in self._self_play_win_rate.items():
            win_rate_split[f] = np.array(data).mean() if len(data) > 0 else 0.5
            win_rate += data
        self._self_play_win_rate = win_rate_split
        win_rate = np.array(win_rate).mean()
        self.results['self_play_win_rate'][step] = win_rate
        logger.info(f"Self-play win rate: {win_rate:.0%}"
                    f" (first 0: {win_rate_split[0]:.0%}, first 1: {win_rate_split[1]:.0%})")

    @property
    def backprop_losses(self):
        return self.results['backprop_loss']

    def _display_plot(self):
        if is_notebook():
            self._display_handle.update(self._fig)
        else:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)  # Needed to refresh the figure

    def _plot_replay_buffer_diversity(self):
        ax = self._ax[0]
        ax.clear()

        d = self.results['self_play_win_rate']
        if d:
            x, y = zip(*d.items())
            ax.plot(x, [.5] * len(x), linestyle='--', color="grey", alpha=.2)
            ax.plot(x, y, 'y', alpha=.6, label="Self-play Win")

        d = self.replay_buffer.diversity_history
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
            d = self.backprop_losses[name]
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

        d = self.results['eval']['checkpoint']

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
        x, y = zip(*self.elo.history.items())
        ax.plot(x, y, 'k')

        # ax_win.set_title("")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Elo")
        self._display_plot()

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
        self._self_play_win_rate[game.first] += [1 - game.winner]

        return history

    def _back_propagate(self, step: int = 0):
        """
        Update the Q-network based on the reward obtained at the end of the game.
        """
        lrs = ', '.join([f"{self.get_lr(i):.1e}" for i in range(self.n_networks)])
        logger.info(f"lr: {lrs}")

        self._train()

        batches = []
        for w, f, data in self.replay_buffer.iter_buckets():
            if self.adaptive_sampling:
                win_rate = self._self_play_win_rate[f]
                if w == 1:
                    win_rate = 1. - win_rate
            else:
                win_rate = .5
            if np.isnan(win_rate):
                logger.warn(f"win_rate is NaN for {w=}, {f=}. To Fix.")
                win_rate = .5
            k = (1. - win_rate) * (self.backprop_steps // 2)
            k = min(round(k), len(data))
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
            print(f"Backprop batch length for player {player}: {len(ba)}")

        self._player_losses = []
        losses, p_losses, v_losses = [], [], []
        for player, ba in batches_by_player.items():
            # model = next(self.get_model(player).parameters())[0][0].clone()
            # other_model = next(self.get_model(1 - player).parameters())[0][0].clone()
            l, p, v = self._backprop_batches(batches=ba, player=player, step=step)
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

        self.results['backprop_loss']['total'][step] = loss
        self.results['backprop_loss']['p'][step] = p_loss
        self.results['backprop_loss']['v'][step] = v_loss

        v_txt = ', '.join([f"v{i}: {v:.2f}" for i, v in enumerate(v_loss)])
        logger.info(f"Backprop loss: {loss:.2f} (p: {p_loss:.2f}, {v_txt})")

    def _backprop_batches(self, batches: list, player: int, step: int):
        if self.freeze_backprop == player:
            logger.info(f"Skipping backprop for player {player} (freeze)")
            return [], [], []

        return_all = self.model_config.double_value_head

        max_entropy = np.log(self.n_pawns)
        lambda_entropy = self._get_lambda_entropy(step)

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
                probs = torch.FloatTensor(probs).to(self.device)
                entropy = - torch.sum(p * torch.log(p + EPS), dim=-1)
                p_loss = self._p_loss(p, probs) + lambda_entropy * (max_entropy - entropy.sum())
                p_losses += [p_loss.item()]
                loss += p_loss
                entropies.append(entropy)
            check_nan(loss)
            losses += [loss.item()]
            loss.backward()
            with self._tweak_lr(player):
                # print(self.get_lr(player))
                self.get_optimizer(player).step()
            self._step_lr(player)

        self._train(mode=False)

        entropies = torch.Tensor(entropies).mean()
        logger.info(
            f"Entropy: {entropies:.2f} (theoretical max: {max_entropy:.2f})"
            f", lambda: {lambda_entropy:.2f}"
        )

        if self.separate_networks:
            loss = float(np.mean(losses))
            self._player_losses.append(loss)
            logger.info(f"Backprop loss for player {player}: {loss :.2f}")

        return losses, p_losses, v_losses

    def _get_lambda_entropy(self, step: int):
        decay_rate = 0.1
        lambda_t = self.lambda_entropy * (decay_rate ** (step / self.lr_final_step))
        return lambda_t

    def _step_lr(self, player: int):
        scheduler = self.get_scheduler(player)
        if scheduler and scheduler.last_epoch < scheduler.T_max:
            scheduler.step()
            # print(self.get_lr(player))

    @contextlib.contextmanager
    def _tweak_lr(self, player: int):
        if self.adaptive_lr and self._lr_tweak:
            base_lr = self.get_lr(player)
            tweak = self._lr_tweak if player == 0 else 1. / self._lr_tweak
            tweak = min(1.5, tweak)
            self.set_lr(base_lr * tweak, player=player)
            # logger.info(f"Player {player} lr tweak: {tweak:.2f}, lr: {self.get_lr(player):.1e}")
            yield
            self.set_lr(base_lr, player=player)
            # logger.info(f"Player {player} back to lr: {self.get_lr(player):.1e}")
        else:
            yield


    def _zero_grad(self):
        for i in range(self.n_networks):
            self.get_optimizer(i).zero_grad()

    def _train(self, mode=True):
        for i in range(self.n_networks):
            self.get_model(i).train(mode=mode)

    @property
    def device(self):
        return self.get_model().device

    def evaluate_agents(self, step):
        vs_list = self._opponents
        results = {}
        for vs in vs_list:
            self.results['eval'][str(vs)][step] = win_rate = self.evaluate_agent(vs=vs)
            win_rate = win_rate['total']
            results[str(vs)] = win_rate
            # if vs == 'random' and _['total'] == 100:
            #     self._opponents += AlphaBetaRelativeAdvancementAgent(max_depth=5)

        self.elo.update(win_rate=results['checkpoint'], n=self.eval_steps, step=step)

        msg = f"{step} steps: "
        msg += ', '.join([f"{v:.0%} vs {k}" for k, v in results.items()])
        logger.info(msg)

    @property
    def elo(self) -> Elo:
        if 'elo' not in self.results:
            self.results['elo'] = Elo()
        return self.results['elo']

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
        steps = self.eval_steps
        if vs != 'checkpoint':
            steps /= 2

        agent = MonteCarloDeepQLearningAgent(
            evaluator=self.evaluator,
            is_training=vs != 'random',
            **self._agent_kwargs,
        )

        if vs == 'checkpoint':
            vs = MonteCarloDeepQLearningAgent(
                evaluator=self.evaluator_chkpt,
                is_training=True,
                **self._agent_kwargs,
            )
        elif vs is None:
            vs = agent

        logger.info(f"Evaluation against {vs}:")

        benchmarker = Benchmark(
            agent_0=agent,
            agent_1=vs,
            n_pawns=self.n_pawns,
            n=steps,
        )
        win_rate = benchmarker.run()
        win_rate_split = benchmarker.win_rates.copy()
        win_rate_split['total'] = win_rate

        return win_rate_split
