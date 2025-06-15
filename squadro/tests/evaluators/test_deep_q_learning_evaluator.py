from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import tensor

from squadro.ml.channels import get_channels
from squadro.ml.ml import assert_models_equal, assert_models_unequal
from squadro.state.evaluators.ml import ModelConfig
from squadro.state.evaluators.rl import DeepQLearningEvaluator
from squadro.state.state import State
from squadro.tests.evaluators.tools import ML


class TestDeepQLearningEvaluator(ML):
    evaluator: DeepQLearningEvaluator

    def get_evaluator(self):
        model_path = TemporaryDirectory().name
        model_config = ModelConfig(
            num_blocks=2,
            cnn_hidden_dim=2,
            value_hidden_dim=2,
            policy_hidden_dim=2,
            double_value_head=False,
            board_flipping=True,
            separate_networks=False,
        )
        return DeepQLearningEvaluator(model_path=model_path, model_config=model_config)

    def test_model(self):
        model_path = Path(TemporaryDirectory().name)
        n_pawns = 2
        model_config = ModelConfig(
            num_blocks=1,
            cnn_hidden_dim=2,
            value_hidden_dim=2,
            policy_hidden_dim=2,
            double_value_head=False,
            board_flipping=True,
            separate_networks=False,
        )
        evaluator = DeepQLearningEvaluator(model_path=model_path, model_config=model_config)
        self.assertEqual('unknown', evaluator.get_weight_update_timestamp(n_pawns))
        self.assertTrue(n_pawns not in evaluator.models)

        model = evaluator.get_model(n_pawns)
        self.assertTrue(n_pawns in evaluator.models)

        self.assertNotEqual('unknown', evaluator.get_weight_update_timestamp(n_pawns))
        self.assertFalse(evaluator.is_pretrained(n_pawns))
        file_path = Path(evaluator.get_filepath(n_pawns))
        self.assertEqual(model_path / 'model_2.pt', file_path)
        self.assertFalse(file_path.exists())
        evaluator.dump(model_path)
        self.assertTrue(file_path.exists())
        self.assertTrue(evaluator.is_pretrained(n_pawns))

        weights = list(model.parameters())

        evaluator = DeepQLearningEvaluator(model_path=model_path)
        weights2 = list(evaluator.get_model(n_pawns).parameters())

        evaluator_r = DeepQLearningEvaluator(
            model_path=model_path / 'new',
            model_config=model_config
        )
        evaluator_r.get_model(n_pawns).load(evaluator.get_model(n_pawns))
        weights3 = list(evaluator.get_model(n_pawns).parameters())

        assert_models_equal(weights, weights2)
        assert_models_equal(weights, weights3)

        evaluator.erase(n_pawns=n_pawns)
        self.assertFalse(file_path.exists())
        self.assertFalse(evaluator.is_pretrained(n_pawns))

        self.assertNotEqual({}, evaluator._models)
        evaluator.reload()
        self.assertEqual({}, evaluator._models)

    def test_to_key(self):
        n_pawns = 3
        key = self.evaluator.get_key(n_pawns=n_pawns, player=0)
        self.assertEqual(n_pawns, key)

        self.evaluator._separate_networks = True
        for player in range(2):
            key = self.evaluator.get_key(n_pawns=n_pawns, player=player)
            self.assertEqual(f"{n_pawns}_{player}", key)

    def test_from_key(self):
        e = self.evaluator
        n_pawns = 3
        d = dict(n_pawns=n_pawns)
        result = e.extract_from_key(e.get_key(**d, player=0))
        self.assertEqual(d, result)

        e._separate_networks = True
        for player in range(2):
            d = dict(n_pawns=n_pawns, player=player)
            result = e.extract_from_key(e.get_key(**d))
            self.assertEqual(d, result)

    def test_eval(self):
        """
        Test that the policy probabilities sum to one and that they are zero for finished pieces.
        """
        p, value = self.evaluator.evaluate(self.state)
        # print(p, value)
        self.assertTrue((p >= 0.).all())
        self.assertAlmostEqual(1., p.sum(), places=6)
        self.assertEqual(0., p[1])

        expected_value = -0.085907943546772
        self.assertAlmostEqual(expected_value, value)

    def test_eval_from_list(self):
        """
        Test that the evaluation from a list representation of a state is the
        same as from the state itself.
        """
        p, value = self.evaluator.evaluate(self.state.to_list())
        p2, value2 = self.evaluator.evaluate(self.state)
        self.assertEqual(value, value2)
        np.testing.assert_array_equal(p, p2)

    def test_eval_torch_output(self):
        p, value = self.evaluator.evaluate(self.state)
        p2, value2 = self.evaluator.evaluate(self.state, torch_output=True)

        self.assertIsInstance(p2, torch.Tensor)
        self.assertIsInstance(value2, torch.Tensor)

        self.assertEqual(value, value2)
        np.testing.assert_array_equal(p, p2.cpu().detach().numpy())

        p, value = self.evaluator.evaluate(
            self.end_state,
            torch_output=True,
            check_game_over=True
        )
        self.assertIsInstance(p, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(1, value)

        p, value = self.evaluator.evaluate(
            self.end_state,
            torch_output=True,
            check_game_over=False,
        )
        self.assertIsInstance(p, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue((p != 1 / self.state.n_pawns).any())
        self.assertTrue(-1 < value < 1)

    def test_eval_return_all(self):
        model_path = self.evaluator.model_path
        model_config = self.evaluator.model_config.copy()
        model_config.double_value_head = True
        evaluator = DeepQLearningEvaluator(model_path=model_path, model_config=model_config)

        p, value = evaluator.evaluate(self.state, return_all=True)

        self.assertIsInstance(p, np.ndarray)
        self.assertIsInstance(value, np.ndarray)

        self.assertEqual(2, len(value))
        self.assertEqual(self.state.n_pawns, len(p))

        p, value = evaluator.evaluate(self.end_state, return_all=True)
        self.assertIsInstance(p, np.ndarray)
        self.assertIsInstance(value, np.ndarray)

        self.assertTrue(np.all(p == 1 / self.state.n_pawns))
        np.testing.assert_array_equal(value, [1, -1])

    def test_load_weights(self):
        other = DeepQLearningEvaluator(
            model_path=self.evaluator.model_path / 'other',
            model_config=self.evaluator.model_config,
        )
        n_pawns = 2

        assert_models_unequal(self.evaluator.get_model(n_pawns), other.get_model(n_pawns))

        self.evaluator.load_weights(other)

        assert_models_equal(self.evaluator.get_model(n_pawns), other.get_model(n_pawns))

    def test_cnn_channels(self):
        turn_count = 13
        state = State(advancement=[[1, 8, 3], [1, 5, 4]], cur_player=0,
                      turn_count=turn_count)
        channels = get_channels(state, board_flipping=True)
        expected = [tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.3333, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.6667, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.6667, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., -1., 0., 0., 0.],
                            [-1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.3750, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.1250, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.1250, 0.0000],
                            [0.0000, 0.6250, 0.0000, 0.0000, 0.0000],
                            [0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.]]) * turn_count / state.max_moves]
        self.assertEqual(len(expected), len(channels))
        for a, b in zip(expected, channels):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)

    def test_cnn_channels_board_flipping(self):
        state = State(advancement=[[1, 8, 3], [1, 5, 4]], cur_player=1)
        channels = get_channels(state, board_flipping=True)
        expected = [tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.6667, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.3333, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.6667, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., -1., 0.],
                            [0., 0., -1., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
                            [0.0000, 0.0000, 0.6250, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.1250, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.1250, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                            [0.0000, 0.3750, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]])]
        self.assertEqual(len(expected), len(channels))
        for a, b in zip(expected, channels):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)

    def test_cnn_channels_separate_networks(self):
        state = State(advancement=[[1, 8, 3], [1, 5, 4]], cur_player=1)
        channels = get_channels(state, separate_networks=True, board_flipping=False)
        expected = [tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.3333, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.6667, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.6667, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., -1., 0., 0., 0.],
                            [-1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.3750, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.1250, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000]]),
                    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.1250, 0.0000],
                            [0.0000, 0.6250, 0.0000, 0.0000, 0.0000],
                            [0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]])]
        print(channels)
        self.assertEqual(len(expected), len(channels))
        for a, b in zip(expected, channels):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
