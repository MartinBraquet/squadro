import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor

from squadro.state import State
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
    channels: int = 64
    value_hidden_dim: int = 64
    policy_hidden_channels: int = 2
    num_blocks: int = 5


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.functional.relu(out + residual)


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

        config = config or ModelConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_channels = 5 + 2 * n_pawns
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(config.channels) for _ in range(config.num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, config.policy_hidden_channels, kernel_size=1),
            nn.BatchNorm2d(config.policy_hidden_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.policy_hidden_channels * n_pawns ** 2, n_pawns),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(config.channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_pawns ** 2, config.value_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.value_hidden_dim, 1),
            nn.Tanh()
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
        x = self.input_conv(x)
        x = self.res_blocks(x)
        p = self.policy_head(x)
        value = self.value_head(x)
        # print(p)
        # print(value)
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
        state: State | list,
        torch_output: bool = False,
        check_game_over: bool = True,
    ) -> tuple[NDArray[np.float64], float]:
        if isinstance(state, list):
            state = State.from_list(state)

        if check_game_over and state.game_over():
            p = self.get_policy(state)
            v = 1 if state.winner == state.cur_player else -1
            if torch_output:
                p = torch.from_numpy(p)
                v = torch.FloatTensor([v])
            return p, v

        d = state.grid_dim

        channels = get_channels(state)

        x = torch.stack(channels, dim=0)

        model = self.get_model(n_pawns=state.n_pawns)
        p, v = model(x)

        if not torch_output:
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


def get_channels(state: State) -> list[Tensor]:
    """
    Get a list of grid where each grid is a binary tensor of shape (d, d)
    where the value is 1 if the pawn is on that grid and 0 otherwise.
    """
    d = state.grid_dim
    channels = []
    for p_id, player_pos in enumerate(state.pos):
        for i, p in enumerate(player_pos):
            grid = torch.zeros((d, d))
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            grid[idx] = 1
            channels.append(grid)

    for p_id, player_pos in enumerate(state.pos):
        grid = torch.zeros((d, d))
        for i, p in enumerate(player_pos):
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            if state.finished[p_id][i]:
                direction = 0
            else:
                direction = -1 if state.returning[p_id][i] else 1
            grid[idx] = direction
        channels.append(grid)

    max_advancement = 2 * state.max_pos
    for p_id, player_pos in enumerate(state.pos):
        grid = torch.zeros((d, d))
        for i, p in enumerate(player_pos):
            idx = (i + 1, p) if p_id == 1 else (p, i + 1)
            grid[idx] = state.get_pawn_advancement(p_id, i) / max_advancement
        channels.append(grid)

    grid = torch.ones((d, d)) * state.cur_player
    channels.append(grid)

    return channels
