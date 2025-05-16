import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from squadro.state import State, get_moves_from_advancement
from squadro.tools.constants import DATA_PATH
from squadro.tools.evaluation import evaluate_advancement
from squadro.tools.log import logger


class Evaluator(ABC):
    """
    Base class for state evaluation.
    """

    @abstractmethod
    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        """
        Evaluate the current state (Q value and policy), according to the current player.

        Args:
            state: The current game state to evaluate

        Returns:
            A tuple containing:
                - NDArray[np.float64]: Probability distribution over possible actions (policy)
                - float: Value estimation for the current state from the perspective of the player
                playing the next move at that state.
        """
        ...

    @staticmethod
    def get_policy(state: State) -> np.ndarray:
        """
        Get the policy for the given state.
        """
        return np.ones(state.n_pawns) / state.n_pawns

    def get_value(self, state: State) -> float:
        """
        Get the value for the given state.
        """
        return self.evaluate(state)[1]

    @classmethod
    def reload(cls):
        ...


class AdvancementEvaluator(Evaluator):
    """
    Evaluate a state according to the advancement heuristic.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State, **kwargs) -> float:
        return evaluate_advancement(state=state)


class ConstantEvaluator(Evaluator):
    """
    Evaluate a state as a constant value.
    """

    def __init__(self, constant: float = 0):
        self.constant = constant

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        return p, self.constant


class RolloutEvaluator(Evaluator):
    """
    Evaluate a state using random playouts until the end of the game.
    """

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    @staticmethod
    def get_value(state: State, **kwargs) -> float:
        cur_player = state.cur_player
        while not state.game_over():
            action = state.get_random_action()
            state = state.get_next_state(action)
        return 1 if state.winner == cur_player else -1


def get_grid_shape(n_pawns: int):
    return [2 * (n_pawns + 1) + 1] * n_pawns * 2 + [2]


def state_to_index(state: State):
    dims = get_grid_shape(state.n_pawns)
    x, y = state.get_advancement()
    return np.ravel_multi_index(x + y + [state.cur_player], dims=dims)


def index_to_state(index: int, n_pawns: int):
    shape = get_grid_shape(n_pawns)
    return np.unravel_index(index, shape=shape)


class _RLEvaluator(Evaluator, ABC):
    _default_dir = 'default'

    def __init__(self, model_path: str | Path = None, dtype='json'):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        self.model_path = Path(model_path or DATA_PATH / self._default_dir)
        self.dtype = dtype

    def erase(self, n_pawns: int):
        filepath = self.get_filepath(n_pawns)
        if os.path.isfile(filepath):
            os.remove(filepath)

    @classmethod
    def reload(cls):
        cls._Q = {}

    def get_filepath(self, n_pawns: int, model_path=None) -> str:
        model_path = Path(model_path or self.model_path)
        return str(model_path / f"model_{n_pawns}.{self.dtype}")

    def clear(self):
        self._Q[self.dir_key] = {}

    @property
    def dir_key(self):
        return str(self.model_path)

    @property
    def models(self):
        """
        :return: A dictionary mapping pawn numbers to Q tables.
        """
        if self.dir_key not in self._Q:
            self._Q[self.dir_key] = {}
        return self._Q[self.dir_key]

    def set_model(
        self,
        d: DictProxy | dict | np.ndarray | ArrayProxy | torch.nn.Module,
        n_pawns: int,
    ) -> None:
        self.models[n_pawns] = d

    def dump(self, model_path: str | Path = None):
        model_path = str(model_path or self.model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for n_pawns, model in self.models.items():
            filepath = self.get_filepath(n_pawns, model_path=model_path)
            self._dump(model, filepath)

    @abstractmethod
    def _dump(self, model, filepath: str):
        ...


class QLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a Q-lookup table.

     _Q is a cache for all Q-tables.
     _Q['/path/dir'][3] is the Q-table in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """
    _Q = {}
    _default_dir = 'q_learning'

    @property
    def is_json(self):
        return self.dtype == 'json'

    def get_Q(self, n_pawns: int) -> dict:  # noqa
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                if self.is_json:
                    self.models[n_pawns] = json.load(open(filepath, 'r'))
                else:
                    self.models[n_pawns] = np.load(filepath, allow_pickle=True)
                logger.info(f"Using Q table at {filepath}")
            else:
                if self.is_json:
                    self.models[n_pawns] = {}
                else:
                    shape = get_grid_shape(n_pawns)
                    length = np.ravel_multi_index([s - 1 for s in shape], dims=shape)
                    self.models[n_pawns] = np.zeros(length, dtype=np.float32)
                logger.warn(f"No file at {filepath}, creating new Q table")

        return self.models[n_pawns]

    def _dump(self, model, filepath: str):
        if isinstance(model, (DictProxy, ArrayProxy)):
            model = model._getvalue()
        if isinstance(model, dict):
            json.dump(model, open(filepath, 'w'), indent=4)
        elif isinstance(model, np.ndarray):
            np.save(filepath, model, allow_pickle=True)

    def evaluate(self, state: State) -> tuple[NDArray[np.float64], float]:
        p = self.get_policy(state)
        value = self.get_value(state)
        return p, value

    def get_value(self, state: State, check_game_over: bool = True) -> float:
        if check_game_over and state.game_over():
            return 1 if state.winner == state.cur_player else -1

        q = self.get_Q(state.n_pawns)
        state_id = self.get_id(state)
        if self.is_json:
            return q.get(state_id, 0)
        else:
            return q[state_id]

    def get_id(self, state: State):
        if self.is_json:
            return f'{state.get_advancement()}, {state.cur_player}'
        else:
            return state_to_index(state)


@dataclass
class ModelConfig:
    n_attn_layers: int = 4
    n_attn_head: int = 8
    d_emb: int = 256
    dropout: float = 0.0


class Model(nn.Module):
    """
    Embedding is advised (instead of CNN) as the state can be represented as a vector of integers.
    Self-attention: When the relationships between different pieces of information in the input are
     non-local and need to be discovered via self-attention mechanisms (like relationships between
     distant game pieces or different turns). Important, because the value of one action must
     strongly depend on the position of all the opponent's pieces.

    TODO: check that they are all using device
    """

    def __init__(
        self,
        n_pawns: int,
        path=None,
        config: ModelConfig = None,
    ):
        super().__init__()

        self.d_s = 2 * n_pawns + 1  # state dim
        config = config or ModelConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.piece_emb = nn.Embedding(2 * (n_pawns + 1) + 1, config.d_emb)
        self.movement_emb = nn.Embedding(3, config.d_emb)
        self.player_emb = nn.Embedding(2, config.d_emb)
        self.position_emb = nn.Embedding(2 * n_pawns, config.d_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_emb,
            nhead=config.n_attn_head,
            dim_feedforward=4 * config.d_emb,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_attn_layers)
        self.ln = nn.LayerNorm(config.d_emb)

        self.policy_mlp = nn.Sequential(
            nn.Linear(config.d_emb, config.d_emb),
            nn.LayerNorm(config.d_emb),
            nn.ReLU(),
            nn.Linear(config.d_emb, 1)
        )

        self.value_attention = nn.Sequential(
            nn.Linear(config.d_emb, 1)
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(config.d_emb, config.d_emb),
            nn.LayerNorm(config.d_emb),
            nn.ReLU(),
            nn.Linear(config.d_emb, 1),
            nn.Tanh(),
        )

        self.load(path)

        self.eval()

    def load(self, path):
        if path is None or not os.path.exists(path):
            return
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
       Evaluate the neural network
        """
        b, d_s = 1, x.size()[0]
        assert d_s == self.d_s, f"Input size {d_s} does not match expected size {self.d_s}"
        assert b == 1, f"Batch size {b} is not supported"

        advancement, player_id = x[:-1], x[-1]
        n_pieces = len(advancement)
        n_pawns = n_pieces // 2

        move_ids = torch.IntTensor(get_moves_from_advancement(advancement)) - 1
        position_ids = torch.arange(0, n_pieces, device=x.device)  # (n_pieces,)

        piece_emb = self.piece_emb(advancement)
        move_emb = self.movement_emb(move_ids)
        player_emb = self.player_emb(player_id)
        pos_emb = self.position_emb(position_ids)

        piece_features = piece_emb + move_emb + player_emb + pos_emb  # (n_pieces, d_emb)

        piece_features = self.ln(piece_features)
        piece_features = self.transformer(piece_features)

        p_logits = self.policy_mlp(piece_features).squeeze(-1)  # (n_pieces,)
        mask = range(n_pawns) if player_id.item() == 1 else range(n_pawns, 2 * n_pawns)
        p_logits = p_logits[mask]
        p = torch.softmax(p_logits, dim=0)  # (n_pawns,)

        attn_scores = self.value_attention(piece_features)  # (n_pieces,)
        attn_weights = torch.softmax(attn_scores, dim=0)
        weighted_sum = (piece_features * attn_weights).sum(dim=0)  # (d_emb,)
        value = self.value_mlp(weighted_sum)  # ()

        return p, value


class DeepQLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a deep neural network.

     _Q is a cache for all neural networks.
     _Q['/path/dir'][3] is the neural network in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """

    _Q = {}
    _default_dir = 'deep_q_learning'

    def __init__(self, **kwargs):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        kwargs.setdefault('dtype', 'pt')
        super().__init__(**kwargs)

    def evaluate(
        self,
        state: State,
        is_torch: bool = False,
        check_game_over: bool = True,
    ) -> tuple[NDArray[np.float64], float]:
        if check_game_over and state.game_over():
            p = self.get_policy(state)
            v = 1 if state.winner == state.cur_player else -1
            if is_torch:
                p = torch.from_numpy(p)
                v = torch.FloatTensor([v])
            return p, v

        l0, l1 = state.get_advancement()
        x = torch.IntTensor(l0 + l1 + [state.cur_player])
        model = self.get_model(state.n_pawns)
        p, v = model(x)
        if not is_torch:
            p = p.detach().numpy()
            v = v.item()
        return p, v

    def get_model(self, n_pawns: int) -> Model:
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                self.models[n_pawns] = torch.load(filepath, weights_only=False)
                logger.info(f"Using model at {filepath}")
            else:
                self.models[n_pawns] = Model(path=filepath, n_pawns=n_pawns)
                logger.warn(f"No file at {filepath}, creating new model")

        return self.models[n_pawns]

    # def set_model_path(self, path, n_pawns: int):
    #     model = self.get_model(n_pawns)
    #     model.load(path)

    def _dump(self, model, filepath: str):
        torch.save(obj=model, f=filepath)
