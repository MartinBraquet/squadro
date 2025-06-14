import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from matplotlib import image as mpimg

import squadro
from squadro import Game, logger
from squadro.agents.random_agent import RandomAgent
from squadro.core.state import State
from squadro.evaluators.evaluator import ModelConfig
from squadro.evaluators.rl import DeepQLearningEvaluator
from squadro.ml.ml import assert_models_equal, assert_models_unequal
from squadro.tests.base import Base
from squadro.tools.disk import load_pickle, dump_pickle, load_txt
from squadro.tools.probabilities import set_seed

DIR = Path(__file__).parent

MEDIUM_PARAMS = dict(
    eval_games=8,
    eval_interval=16,
    backprop_interval=16,
    backprop_games=16,
    self_play_games=32,
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


class _Base(Base):
    def setUp(self):
        super().setUp()
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

    def get_trainer(self, **kwargs) -> squadro.DeepQLearningTrainer:
        model_path = Path(TemporaryDirectory().name)
        kwargs = dict(
            n_pawns=3,
            eval_games=4,
            eval_interval=4,
            backprop_interval=2,
            backprop_games=2,
            self_play_games=4,
            model_path=model_path,
            model_config=self.model_config,
            mcts_kwargs=dict(
                max_steps=4,
            ),
            display_plot=False,
        ) | kwargs
        trainer = squadro.DeepQLearningTrainer(**kwargs)
        return trainer


class Data:
    data = {}
    _data_path = DIR / 'data/dql_data.pkl'

    @classmethod
    def load(cls):
        data = load_pickle(cls._data_path, raise_error=False)
        # print(data)
        if data:
            cls.data = data
        return data

    @classmethod
    def dump(cls):
        dump_pickle(cls.data, cls._data_path)

    @classmethod
    def get(cls, key):
        return cls.data[key]

    @classmethod
    def set(cls, key, value):
        cls.data[key] = value


class Results(Data):
    data = {}
    _data_path = DIR / 'data/results.pkl'


# @pytest.mark.slow
class TestDeepQLearningTrainerTight(_Base):
    """
    Those are very tight integration tests.
    They will frequently break as soon as the implementation changes. When they change, we want
    to be able to quickly update the expected results here.
    If there is a bug in a low-level method, the bug should be caught somewhere else. The tests
    here are not intended to catch low-level bugs.
    The primary goal of these tests is to be able to refactor the code or any other code changes
    that are not expected to affect the results.
    In other words, if we change the results, those tests should be ignored.
    If we refactor code without result change, we should pay close attention to those tests and
    ensure that they pass before and after the refactoring.
    """
    UPDATE_DATA = False

    def setUp(self):
        super().setUp()
        Data.load()

    def check(self, evaluator, key):
        evaluation = evaluator.evaluate(self.state)
        if self.UPDATE_DATA:
            Data.set(key, evaluation)
            Data.dump()
        expected = Data.get(key)
        # print(expected)
        # print(evaluation)
        for a, b in zip(expected, evaluation):
            self.assertEqualGeneral(a, b)

    def test_from_scratch(self):
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        self.check(evaluator, key='from_scratch')

    def test_board_flipping(self):
        self.model_config.board_flipping = True
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        self.check(evaluator, key='board_flipping')

    def test_separate_networks(self):
        self.model_config.separate_networks = True
        trainer = self.get_trainer()
        trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        self.check(evaluator, key='separate_networks')

    @patch.object(Game, 'run', run)
    def test_longer_training_fake_game(self):
        """
        In the other tests, there are so few values in different attributes of the trainer
        that they might miss quiet bugs (which made very small differences).
        This test should make sure the attributes, like win rates, stay correct.
        """
        trainer = self.get_trainer(**MEDIUM_PARAMS)
        with logger.setup_in_context(section=['training', 'benchmark']):
            trainer.run()
        evaluator = DeepQLearningEvaluator(
            model_path=trainer.model_path,
            model_config=self.model_config,
        )
        self.check(evaluator, key='longer_training')

        logs = load_txt(trainer.model_path / 'results/logs.txt')
        self.assertGreater(len(logs), 0)

        results = load_pickle(trainer.model_path / 'results/results.pkl')
        if self.UPDATE_DATA:
            Results.data = results
            Results.dump()
        expected = Results.load()
        self.assertEqual(str(expected.data), str(results.data))

        mpimg.imread(trainer.model_path / 'results/plots.png')


class TestDeepQLearningTrainer(_Base):
    # Adam optimizer is very slow to initialize (1 sec), even with very light model.
    # A fix for it would greatly speed up the tests.

    @patch.object(Game, 'run', run)
    def test_from_file(self):
        logger.setup(section=['training', 'benchmark'])

        trainer = self.get_trainer()

        self.assertEqual(1, trainer.game_count)
        self.assertEqual(trainer.get_lr(0), 1e-3)

        trainer.run()

        self.assertEqual(5, trainer.game_count)
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
        # print(files)
        self.assertGreaterEqual(len(files), 1)
        pt = 'checkpoint/model_3.pt'
        if pt in files:
            files.remove(pt)
        path = path / files[0] / 'model_3.pt'
        # print(path)
        self.assertTrue(path.exists())

        path = model_path / 'results/logs.txt'
        self.assertTrue(path.exists())

        trainer = self.get_trainer(self_play_games=10, model_path=model_path)

        self.assertEqual(5, trainer.game_count)
        self.assertLess(trainer.get_lr(0), 1e-3)

        buffer = trainer.replay_buffer
        self.assertGreater(len(buffer), 0)
        self.assertGreater(len(buffer.diversity_history), 0)

        self.assertGreater(len(trainer.self_play_win_rates), 0)
        self.assertGreater(len(trainer.backprop_losses), 0)
        self.assertGreater(len(trainer.checkpoint_eval), 0)
        self.assertGreater(len(trainer.elo.history), 1)

        trainer.run()

        self.assertEqual(11, trainer.game_count)

    def test_update_checkpoint(self):
        trainer = self.get_trainer()

        checkpoint, model = trainer.get_model_chkpt(0), trainer.get_model(0)

        assert_models_unequal(checkpoint, model)

        trainer.checkpoint_eval[trainer.game_count]['total'] = .1
        trainer.update_checkpoint_model()

        assert_models_unequal(checkpoint, model)

        trainer.checkpoint_eval[trainer.game_count]['total'] = .9
        trainer.update_checkpoint_model()

        assert_models_equal(checkpoint, model)

        self.assertEqual(trainer.elo.current, trainer.elo.checkpoint)

    def test_self_play_info(self):
        trainer = self.get_trainer()

        trainer._self_play_win_rate = {0: [1, 0, 1, 0, 0], 1: [1, 1, 0, 0]}
        trainer._process_self_play_info()
        self.assertEqual(trainer.self_play_win_rates[trainer.game_count], 4 / 9)
        self.assertEqual(trainer._self_play_win_rate, {0: 0.4, 1: 0.5})

        trainer._self_play_win_rate = {0: [1], 1: []}
        trainer._process_self_play_info()
        self.assertEqual(trainer.self_play_win_rates[trainer.game_count], 1)
        self.assertEqual(trainer._self_play_win_rate, {0: 1, 1: .5})

    def test_plot(self):
        trainer = self.get_trainer(display_plot=True)
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

        trainer.set_game_count(0)
        self.assertEqual(trainer.lambda_entropy, trainer._get_lambda_entropy())

        trainer.set_game_count(trainer.entropy_temp)
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

        eval_games = 8
        trainer = self.get_trainer(eval_games=eval_games)

        win_rate_split = trainer.evaluate_agent(vs='random')
        self.assertEqual({
            'total': 0.5,
            (0, 0): 1.0,
            (0, 1): 1.0,
            (1, 0): 0.0,
            (1, 1): 0.0
        }, win_rate_split)
        self.assertEqual(eval_games / 2, RunMock.call_count)

        RunMock.call_count = 0
        trainer.evaluate_agent(vs='checkpoint')
        self.assertEqual(eval_games, RunMock.call_count)

    @patch.object(Game, 'run', run)
    def test_elo_evaluation(self):
        trainer = self.get_trainer(eval_games=8)
        self.assertEqual(trainer.elo.current, trainer.elo.checkpoint)

        trainer.evaluate_agents()

        s = trainer.game_count
        n_data = 5
        self.assertEqual(len(trainer.results['eval']['checkpoint'][s]), n_data)
        self.assertEqual(len(trainer.results['eval']['random'][s]), n_data)

        # Might be equal in some cases (if the win rate is 50%), but not with this seed
        self.assertNotEqual(trainer.elo.current, trainer.elo.checkpoint)
