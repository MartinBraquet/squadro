import json
import random
from collections import defaultdict
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
from squadro.tools.disk import pickle_dump, mkdir
from squadro.tools.elo import get_expected_score, Elo
from squadro.tools.log import training_logger as logger
from squadro.tools.notebooks import is_notebook
from squadro.tools.state import get_reward
from squadro.training.buffer import ReplayBuffer
from squadro.training.constants import DQL_PATH

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

        self.n_steps = n_steps
        self.lr_final_step = lr_final_step or 5000
        if n_steps:
            self.lr_final_step = min(n_steps, self.lr_final_step)

        self.eval_interval = eval_interval or 500
        self.eval_steps = max(eval_steps or 100, 4)

        self.mcts_kwargs = mcts_kwargs or {}
        self.mcts_kwargs.setdefault('max_steps', int(1.3 * self.n_pawns ** 3))

        self.backprop_interval = backprop_interval or 100
        self.backprop_steps = backprop_steps or self.backprop_interval * 10
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

        lr = lr or 1e-3
        self.min_lr = min_lr or max(lr / 10, 1e-4)

        self._optimizer = [torch.optim.Adam(
            params=self.get_model(player=i).parameters(),
            lr=lr,
            weight_decay=1e-4,
        ) for i in range(self.n_networks)]

        self._scheduler = None
        if self.min_lr != lr:
            self._scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(
                self.get_optimizer(player=i),
                eta_min=self.min_lr,
                T_max=self.lr_final_step * backprop_per_game,
            ) for i in range(self.n_networks)]

        self.replay_buffer = ReplayBuffer(
            n_pawns=self.n_pawns,
            path=self.model_path / 'replay_buffer.pkl',
        )

        self.results = defaultdict(dict_factory)
        self.results['elo'] = Elo()

        self._v_loss = torch.nn.MSELoss(reduction='none')
        self._p_loss = torch.nn.CrossEntropyLoss()

        self._opponents = [
            'checkpoint',
            'random',
        ]

        if init_from == 'scratch':
            self.evaluator.erase(self.n_pawns)
            self.replay_buffer.clear()

        self._display_handle, self._fig, self._ax = None, None, None
        self._self_play_win_rate = None
        self._results_path = None

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

    def get_scheduler(self, player=0) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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

    def run(self) -> None:
        """
        Run the training loop.
        """
        logger.info(
            f"Starting training: {self.n_pawns} pawns\n"
            f"model size: {self.get_model().byte_size()}\n"
            f"model config: {self.model_config}\n"
            f"optimizer: {self.get_optimizer()}\n"
            f"scheduler: {self.get_scheduler().__class__.__name__} (min lr: {self.min_lr})\n"
            f"games: {self.n_steps}\n"
            f"backprop interval: {self.backprop_interval}\n"
            f"backprop steps: {self.backprop_steps}\n"
            f"eval interval: {self.eval_interval}\n"
            f"eval steps: {self.eval_steps}\n"
            f"MCTS config: {json.dumps(self.agent.mcts_config, indent=4)}\n"
            f"device: {self.device}\n"
            f"path: {self.model_path}\n"
            f"replay buffer: {self.replay_buffer}\n"
        )
        if not self.evaluator_chkpt.is_pretrained(n_pawns=self.n_pawns):
            self.copy_weights_to_checkpoint()

        self._open_figure()
        self._results_path = RESULTS_PATH / get_now()
        logger.info(f'Results will be saved in {self._results_path}')
        mkdir(self._results_path)

        try:
            self._run()
        finally:
            self.dump()
            self._close_figure()
            logger.dump_history(self._results_path / 'logs.txt')

        logger.info("Training finished.")

    def copy_weights_to_checkpoint(self):
        self.evaluator_chkpt.load_weights(self.evaluator, n_pawns=self.n_pawns)

    def _run(self):
        self._clear_self_play_win_rate()

        n_steps = self.n_steps or int(1e15)
        for step in range(1, n_steps + 1):
            # print(step)
            self.get_training_samples()

            if step % self.backprop_interval == 0:
                self._process_self_play_info()
                self._back_propagate(step)
                self._plot_loss()
                self.replay_buffer.get_state_uniqueness()
                self._clear_self_play_win_rate()

            if step % self.eval_interval == 0:
                self.evaluate_agents(step)
                self.update_checkpoint_model(step)
                self._plot_win_rate()
                self.dump()

    def _open_figure(self):
        self._fig, self._ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        self._ax = self._ax.flatten()
        self._fig.suptitle(self.title)
        if is_notebook():
            self._display_handle = display(self._fig, display_id=True)
        else:
            plt.ion()

    def _close_figure(self):
        plt.savefig(self._results_path / 'plots.png')
        if is_notebook():
            plt.close(self._fig)
        else:
            plt.ioff()
            plt.show(block=False)

    def update_checkpoint_model(self, step):
        if self.results['eval']['checkpoint'][step]['total'] > .7:  # noqa
            logger.info(f"Updating best checkpoint")
            self.copy_weights_to_checkpoint()
            self.evaluator_chkpt.dump()
            self.elo.checkpoint = self.elo.current

    def dump(self):
        # with logger.context_info('dump'):
        self.dump_results()
        self.evaluator.dump()
        filename = get_now()
        self.evaluator.dump(Path(self.evaluator_chkpt.model_path) / filename)
        self.replay_buffer.save()
        logger.info(f"Dumped current model to '{filename}'")

    def dump_results(self, fmt='pkl'):
        for name in ('eval', 'backprop_loss'):
            result = self.results[name]
            if fmt == 'json':
                result = {str(k): v for k, v in result.items()}
                json.dump(result, open(self._results_path / f'{name}.json', 'w'))
            elif fmt == 'pkl':
                pickle_dump(result, self._results_path / f'{name}.pkl')
            else:
                raise ValueError(f"Unknown format: {fmt}")

    def _clear_self_play_win_rate(self):
        self._self_play_win_rate = {0: [], 1: []}

    def _process_self_play_info(self):
        win_rate = []
        win_rate_split = {}
        for f, data in self._self_play_win_rate.items():
            win_rate_split[f] = np.array(data).mean()
            win_rate += data
        self._self_play_win_rate = win_rate_split
        win_rate = np.array(win_rate).mean()
        logger.info(f"Self-play win rate: {win_rate:.0%}"
                    f" (first 0: {win_rate_split[0]:.0%}, first 1: {win_rate_split[1]:.0%})")

    def _display_plot(self):
        if is_notebook():
            self._display_handle.update(self._fig)
        else:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)  # Needed to refresh the figure

    def _plot_loss(self):
        ax_loss = self._ax[0]
        ax_loss.clear()
        labels = {'total': "Total", 'p': "Policy", 'v': "Value"}
        colors = {'total': "k", 'p': "b", 'v': 'g'}
        count = 0
        for name in ('p', 'v', 'total'):
            x, y = zip(*self.results['backprop_loss'][name].items())
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
                ax_loss.plot(
                    x, y[i], color[i],
                    label=label,
                    alpha=1 if name == 'total' else .5,
                )
                count += 1
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        self._display_plot()

    def _plot_win_rate(self):
        ax_win = self._ax[1]
        ax_win.clear()
        x, y_all = zip(*self.results['eval']['checkpoint'].items())

        ax_win.plot(x, [.5] * len(x), linestyle='--', color="grey", alpha=.2)

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
            ax_win.plot(
                x, y, colors.get(k),
                label=labels.get(k, k),
                alpha=1 if k == 'total' else .5,
            )

        ax_win.set_title("Win Rate against Checkpoint")
        ax_win.set_xlabel("Epoch")
        ax_win.set_ylabel("Win Rate")
        ax_win.set_ylim(0, 1.1)
        ax_win.legend()
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

        self.replay_buffer.add_game(history=history, winner=game.winner, first=game.first)
        self._self_play_win_rate[game.first] += [1 - game.winner]

        return history

    def _back_propagate(self, step: int = 0):
        """
        Update the Q-network based on the reward obtained at the end of the game.
        """
        lrs = ', '.join([f"{self.get_lr(i):.1e}" for i in range(self.n_networks)])
        logger.info(f"lr: {lrs}")

        if step > self.lr_final_step:
            self._scheduler = None

        self._train()

        batches = []
        for w, f, data in self.replay_buffer.iter_buckets():
            win_rate = self._self_play_win_rate[f]
            if w == 1:
                win_rate = 1 - win_rate
            k = (1 - win_rate) * (self.backprop_steps // 2)
            k = min(round(k), len(data))
            batch = random.sample(data, k=k)
            if batch:
                batches.append(batch)
                logger.info(f"{w=}, {f=}, {len(batch)} samples")

        batches = [s for t in zip(*batches) for s in t]

        if self.separate_networks:
            # sort by current player
            batches_by_player = {0: [], 1: []}
            for state, probs, winner in batches:
                cur_player = state[1]
                batches_by_player[cur_player] += [(state, probs, winner)]
        else:
            batches_by_player = {0: batches}

        losses, p_losses, v_losses = [], [], []
        for player, ba in batches_by_player.items():
            l, p, v = self._backprop_batches(ba)
            losses += l
            p_losses += p
            v_losses += v

        loss = float(np.mean(losses))
        p_loss = float(np.mean(p_losses))
        v_loss = np.mean(np.array(v_losses), axis=0)

        self.results['backprop_loss']['total'][step] = loss
        self.results['backprop_loss']['p'][step] = p_loss
        self.results['backprop_loss']['v'][step] = v_loss

        v_txt = ', '.join([f"v{i}: {v:.2f}" for i, v in enumerate(v_loss)])
        logger.info(f"Backprop loss: {loss:.2f} (p: {p_loss:.2f}, {v_txt})")

    def _backprop_batches(self, batches: list, player: int = 0):
        return_all = self.model_config.double_value_head

        losses, p_losses, v_losses = [], [], []
        for state, probs, winner in batches:
            assert winner in {0, 1}, f"Got {winner=} instead of {0, 1} for state {state}"
            cur_player = state[1]
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
                p_loss = self._p_loss(p, probs)
                p_losses += [p_loss.item()]
                loss += p_loss
            check_nan(loss)
            losses += [loss.item()]
            loss.backward()
            self.get_optimizer(player).step()
            if scheduler := self.get_scheduler(player):
                scheduler.step()

        self._train(mode=False)

        return losses, p_losses, v_losses

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

        self.compute_elo(win_rate=results['checkpoint'])

        msg = f"{step} steps: "
        msg += ', '.join([f"{v:.0%} vs {k}" for k, v in results.items()])
        logger.info(msg)

    @property
    def elo(self) -> Elo:
        return self.results['elo']

    def compute_elo(self, win_rate: float, k=2):
        expected_score = get_expected_score(self.elo.current, self.elo.checkpoint)
        delta_elo = k * (win_rate - expected_score) * self.eval_steps
        self.elo.update(delta_elo)
        logger.info(f"Elo: {self.elo} (delta: {delta_elo:.0f})")

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
