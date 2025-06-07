from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import tensor

from squadro.evaluators.channels import get_channels
from squadro.evaluators.evaluator import ModelConfig
from squadro.evaluators.rl import DeepQLearningEvaluator
from squadro.state import State
from squadro.tests.tools import ML


class TestDeepQLearningEvaluator(ML):
    def get_evaluator(self):
        model_path = TemporaryDirectory().name
        return DeepQLearningEvaluator(model_path=model_path)

    def test_model(self):
        model_path = Path(TemporaryDirectory().name)
        n_pawns = 2
        model_config = ModelConfig(
            num_blocks=1,
            cnn_hidden_dim=2,
            value_hidden_dim=2,
            policy_hidden_dim=2,
            double_value_head=False,
            board_flipping=False,
            separate_networks=False,
        )
        evaluator = DeepQLearningEvaluator(model_path=model_path, model_config=model_config)
        weights = list(evaluator.get_model(n_pawns).parameters())
        self.assertFalse(model_path.exists())
        evaluator.dump(model_path)
        self.assertTrue(model_path.exists())

        evaluator = DeepQLearningEvaluator(model_path=model_path)
        weights2 = list(evaluator.get_model(n_pawns).parameters())

        evaluator_r = DeepQLearningEvaluator(model_path=model_path, model_config=model_config)
        evaluator_r.get_model(n_pawns).load(evaluator.get_model(n_pawns))
        weights3 = list(evaluator.get_model(n_pawns).parameters())

        for w1, w2, w3 in zip(weights, weights2, weights3):
            torch.testing.assert_close(w1, w2)
            torch.testing.assert_close(w1, w3)



    def test_eval(self):
        """
        Test that the policy probabilities sum to one and that they are zero for finished pieces.
        """
        state = State(advancement=[[1, 8, 3], [1, 2, 4]], cur_player=0)
        p, value = self.evaluator.evaluate(state)
        # self.assertEqual(expected_value, value)
        # print(p, value)
        self.assertAlmostEqual(1., p.sum())
        self.assertEqual(0., p[1])

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
