from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from squadro.evaluators.evaluator import ModelConfig, Model
from squadro.tools.probabilities import set_seed


class TestModel(TestCase):
    def setUp(self):
        set_seed()

    def test(self):
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
        with TemporaryDirectory() as model_path:
            path = Path(model_path) / 'model.pt'
            model = Model(
                n_pawns=n_pawns,
                config=model_config,
            )
            weights = list(model.parameters())
            self.assertFalse(path.exists())
            model.save(path)
            self.assertTrue(path.exists())
            model = Model(
                n_pawns=n_pawns,
                path=path,
                config=model_config,
            )
            weights2 = list(model.parameters())
            for w1, w2 in zip(weights, weights2):
                torch.testing.assert_close(w1, w2)
