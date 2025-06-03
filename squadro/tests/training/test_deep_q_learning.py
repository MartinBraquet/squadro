from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest

import squadro
from squadro.evaluators.evaluator import ModelConfig
from squadro.tools.probabilities import set_seed


# def run(game: Game):
#     state = State(game.n_pawns)
#     game.state = state.copy()
#     game.state.set_from_advancement([[0, 0, 0], [8, 8, 0]])
#     game.state_history = [state, state.get_next_state(0), game.state]


class TestDeepQLearningTrainer(TestCase):
    def setUp(self):
        set_seed()

    # @patch.object(Game, 'run', run)
    @pytest.mark.slow
    def test_from_scratch(self):
        config = ModelConfig()
        config.num_blocks = 2
        config.cnn_hidden_dim = 2
        config.value_hidden_dim = 2

        with TemporaryDirectory() as model_path:
            trainer = squadro.DeepQLearningTrainer(
                n_pawns=3,
                eval_steps=4,
                eval_interval=3,
                backprop_interval=3,
                backprop_steps=2,
                n_steps=4,
                model_path=model_path,
                model_config=config,
                mcts_kwargs=dict(
                    max_steps=4,
                )
            )
            trainer.run()
            # q = json.load(open(f"{model_path}/model_3.pt"))
    #
    # @patch.object(Game, 'run', run)
    # def test_from_file(self):
    #     with TemporaryDirectory() as model_path:
    #         key = '[[1, 2, 3], [4, 5, 6]], 0'
    #         value = 0.42
    #         json.dump({key: value}, open(f"{model_path}/model_3.json", mode='w'))
    #         trainer = QLearningTrainer(
    #             n_pawns=3,
    #             eval_steps=2,
    #             eval_interval=3,
    #             n_steps=4,
    #             lr=.2,
    #             gamma=.95,
    #             model_path=model_path,
    #         )
    #         trainer.run()
    #         q = json.load(open(f"{model_path}/model_3.json"))
    #         self.assertEqual(value, q[key])
    #         self.assertGreater(len(q), 1)
