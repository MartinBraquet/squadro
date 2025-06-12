import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import squadro
from squadro import Game, logger
from squadro.agents.random_agent import RandomAgent
from squadro.evaluators.evaluator import ModelConfig
from squadro.evaluators.rl import DeepQLearningEvaluator
from squadro.state import State
from squadro.tools.ml import assert_models_equal, assert_models_unequal
from squadro.tools.probabilities import set_seed

MEDIUM_PARAMS = dict(
    eval_steps=8,
    eval_interval=16,
    backprop_interval=16,
    backprop_steps=16,
    n_steps=32,
)


class QuickFakeAgent(RandomAgent):
    def get_move_info(self):
        probs = np.random.random(size=3)
        probs /= probs.sum()
        return dict(
            mcts_move_probs=probs,
        )


class RunMock:
    call_count = 0


def run(game: Game):
    RunMock.call_count += 1
    agents = game.agents
    game.agents = [QuickFakeAgent(), QuickFakeAgent()]
    game._run()  # noqa
    game.agents = agents


class TestDeepQLearningTrainer(TestCase):
    def setUp(self):
        set_seed()
        self.model_config = ModelConfig(
            num_blocks=2,
            cnn_hidden_dim=2,
            value_hidden_dim=2,
            policy_hidden_dim=2,
            double_value_head=False,
            board_flipping=False,
            separate_networks=False,
        )
        self.state = State(cur_player=0, advancement=[[1, 2, 3], [4, 5, 6]])

    def get_trainer(self, **kwargs):
        model_path = Path(TemporaryDirectory().name)
        kwargs = dict(
            n_pawns=3,
            eval_steps=4,
            eval_interval=4,
            backprop_interval=2,
            backprop_steps=2,
            n_steps=4,
            model_path=model_path,
            model_config=self.model_config,
            mcts_kwargs=dict(
                max_steps=4,
            ),
            plot=False,
        ) | kwargs
        trainer = squadro.DeepQLearningTrainer(**kwargs)
        return trainer

    # @pytest.mark.slow
    def test_from_scratch(self):
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        p, v = evaluator.evaluate(self.state)
        self.assertEqual(0.30241554975509644, v)
        np.testing.assert_almost_equal(p, [0.3195003, 0.2665146, 0.4139851])

    def test_board_flipping(self):
        self.model_config.board_flipping = True
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        p, v = evaluator.evaluate(self.state)
        self.assertEqual(-0.08411766588687897, v)
        np.testing.assert_almost_equal(p, [0.403206, 0.2753415, 0.3214525])

    def test_separate_networks(self):
        self.model_config.separate_networks = True
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        p, v = evaluator.evaluate(self.state)
        self.assertEqual(-0.085907943546772, v)
        np.testing.assert_almost_equal(p, [0.4568438, 0.22258231, 0.3205739])

    @patch.object(Game, 'run', run)
    def test_longer_training_fake_game(self):
        """
        In the other tests, there are so few values in different attributes of the trainer
        that they might miss quiet bugs (which made very small differences).
        This test should make sure the attributes, like win rates, stay correct.
        """
        trainer = self.get_trainer(**MEDIUM_PARAMS)
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        p, v = evaluator.evaluate(self.state)
        self.assertEqual(0.2970362603664398, v)
        np.testing.assert_almost_equal(p, [0.3038661, 0.3249999, 0.371134])

    @patch.object(Game, 'run', run)
    def test_from_file(self):
        logger.setup(section=['training', 'benchmark'])

        trainer = self.get_trainer()

        self.assertEqual(1, trainer.get_step())
        self.assertEqual(trainer.get_lr(0), 1e-3)

        trainer.run()

        self.assertEqual(5, trainer.get_step())
        self.assertLess(trainer.get_lr(0), 1e-3)

        model_path = trainer.model_path

        path = model_path / 'model_3.pt'
        self.assertTrue(path.exists())

        path = model_path / 'replay_buffer.pkl'
        self.assertTrue(path.exists())

        path = model_path / 'results/results.pkl'
        self.assertTrue(path.exists())

        path = model_path / 'checkpoint'
        files = os.listdir(path)
        self.assertGreaterEqual(len(files), 1)
        # files.remove('checkpoint/model_3.pt')
        # print(files)
        path = path / files[0] / 'model_3.pt'
        self.assertTrue(path.exists())

        path = model_path / 'results/logs.txt'
        self.assertTrue(path.exists())

        trainer = self.get_trainer(n_steps=10, model_path=model_path)

        self.assertEqual(5, trainer.get_step())
        self.assertLess(trainer.get_lr(0), 1e-3)

        buffer = trainer.replay_buffer
        self.assertGreater(len(buffer), 0)
        self.assertGreater(len(buffer.diversity_history), 0)

        self.assertGreater(len(trainer.results), 0)
        self.assertGreater(len(trainer.self_play_win_rates), 0)
        self.assertGreater(len(trainer.backprop_losses), 0)
        self.assertGreater(len(trainer.checkpoint_eval), 0)
        self.assertGreater(len(trainer.elo.history), 1)

        trainer.run()

        self.assertEqual(11, trainer.get_step())

    def test_update_checkpoint(self):
        trainer = self.get_trainer()

        checkpoint, model = trainer.get_model_chkpt(0), trainer.get_model(0)

        assert_models_unequal(checkpoint, model)

        trainer.checkpoint_eval[trainer.get_step()]['total'] = .1
        trainer.update_checkpoint_model()

        assert_models_unequal(checkpoint, model)

        trainer.checkpoint_eval[trainer.get_step()]['total'] = .9
        trainer.update_checkpoint_model()

        assert_models_equal(checkpoint, model)

        self.assertEqual(trainer.elo.current, trainer.elo.checkpoint)

    def test_self_play_info(self):
        trainer = self.get_trainer()

        trainer._self_play_win_rate = {0: [1, 0, 1, 0, 0], 1: [1, 1, 0, 0]}
        trainer._process_self_play_info()
        self.assertEqual(trainer.self_play_win_rates[trainer.get_step()], 4 / 9)
        self.assertEqual(trainer._self_play_win_rate, {0: 0.4, 1: 0.5})

        trainer._self_play_win_rate = {0: [1], 1: []}
        trainer._process_self_play_info()
        self.assertEqual(trainer.self_play_win_rates[trainer.get_step()], 1)
        self.assertEqual(trainer._self_play_win_rate, {0: 1, 1: .5})

    def test_plot(self):
        trainer = self.get_trainer(plot=True)
        trainer.run()

    def test_step_lr(self):
        trainer = self.get_trainer()
        lr = trainer.get_lr(0)
        trainer._step_lr(0)
        self.assertLess(trainer.get_lr(0), lr)

    def test_tweak_lr(self):
        trainer = self.get_trainer()
        trainer._lr_tweak = a = .42
        lr = trainer.get_lr(0)
        with trainer._tweak_lr(0):
            self.assertEqual(trainer.get_lr(0), a * lr)
        self.assertEqual(trainer.get_lr(0), lr)

    def test_policy_entropy(self):
        trainer = self.get_trainer()

        trainer.set_step(0)
        self.assertEqual(trainer.lambda_entropy, trainer._get_lambda_entropy())

        trainer.set_step(trainer.entropy_final_step)
        self.assertEqual(.1 * trainer.lambda_entropy, trainer._get_lambda_entropy())

    @patch.object(Game, 'run', run)
    def test_self_play(self):
        trainer = self.get_trainer()
        self.assertEqual(len(trainer.replay_buffer), 0)

        trainer._clear_self_play_win_rate()

        self.assertEqual(trainer._self_play_win_rate, {0: [], 1: []})

        trainer.get_training_samples()

        self.assertGreater(len(trainer.replay_buffer), 0)
        first = 1
        self.assertGreater(len(trainer._self_play_win_rate[first]), 0)

    @patch.object(Game, 'run', run)
    def test_evaluation(self):
        RunMock.call_count = 0

        eval_steps = 8
        trainer = self.get_trainer(eval_steps=eval_steps)

        win_rate_split = trainer.evaluate_agent(vs='random')
        self.assertEqual({
            'total': 0.5,
            (0, 0): 1.0,
            (0, 1): 0.0,
            (1, 0): 0.0,
            (1, 1): 1.0
        }, win_rate_split)
        self.assertEqual(eval_steps / 2, RunMock.call_count)

        RunMock.call_count = 0
        trainer.evaluate_agent(vs='checkpoint')
        self.assertEqual(eval_steps, RunMock.call_count)

    @patch.object(Game, 'run', run)
    def test_evaluation(self):
        trainer = self.get_trainer(eval_steps=8)
        self.assertEqual(trainer.elo.current, trainer.elo.checkpoint)

        trainer.evaluate_agents()

        s = trainer.get_step()
        n_data = 5
        self.assertEqual(len(trainer.results['eval']['checkpoint'][s]), n_data)
        self.assertEqual(len(trainer.results['eval']['random'][s]), n_data)

        # Might be equal in some cases (if the win rate is 50%), but not with this seed
        self.assertNotEqual(trainer.elo.current, trainer.elo.checkpoint)
