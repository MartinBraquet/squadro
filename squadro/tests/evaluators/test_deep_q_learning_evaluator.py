from unittest import TestCase

import torch
from torch import tensor

from squadro.evaluators.evaluator import DeepQLearningEvaluator, get_channels
from squadro.state import State
from squadro.tools.probabilities import set_seed


class TestDeepQLearningEvaluator(TestCase):
    def setUp(self):
        set_seed()
        self.evaluator = DeepQLearningEvaluator(model_path='/tmp/test')

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

    def test_game_over(self):
        state = State(advancement=[[8, 8, 3], [1, 2, 4]], cur_player=0)
        value = self.evaluator.get_value(state)
        self.assertEqual(1, value)

        state.cur_player = 1
        value = self.evaluator.get_value(state)
        self.assertEqual(-1, value)

    def test_cnn_channels(self):
        state = State(advancement=[[1, 8, 3], [1, 5, 4]], cur_player=0)
        channels = get_channels(state)
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
                            [0., 0., 0., 2., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 3., 0.],
                            [0., 3., 0., 0., 0.],
                            [2., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]),
                    tensor([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]])]
        self.assertEqual(len(expected), len(channels))
        for a, b in zip(expected, channels):
            torch.testing.assert_close(a, b)
