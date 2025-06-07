import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy._typing import NDArray
from torch import nn

from squadro import logger
from squadro.evaluators.channels import get_num_channels, get_channels
from squadro.evaluators.evaluator import Evaluator
from squadro.state import State
from squadro.tools.constants import DATA_PATH, inf
from squadro.tools.dates import get_file_modified_time, get_now, READABLE_DATE_FMT
from squadro.tools.ml import get_model_size
from squadro.tools.state import get_grid_shape, state_to_index, get_reward

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _RLEvaluator(Evaluator, ABC):
    _default_dir = 'default'
    _weight_update_timestamp = defaultdict(lambda: 'unknown')

    def __init__(self, model_path: str | Path = None, dtype='json'):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        self.model_path = Path(model_path or DATA_PATH / self._default_dir)
        self.dtype = dtype

    def get_weight_update_timestamp(self, n_pawns: int):
        return self._weight_update_timestamp[self.get_filepath(n_pawns)]

    def erase(self, n_pawns: int):
        filepath = self.get_filepath(n_pawns)
        if os.path.isfile(filepath):
            os.remove(filepath)

    @classmethod
    def reload(cls):
        cls._models = {}

    def get_filepath(self, n_pawns: int, model_path=None) -> str:
        model_path = Path(model_path or self.model_path)
        return str(model_path / f"model_{n_pawns}.{self.dtype}")

    def clear(self):
        self._models[self.dir_key] = {}

    @property
    def dir_key(self):
        return str(self.model_path)

    @property
    def models(self):
        """
        :return: A dictionary mapping pawn numbers to Q tables.
        """
        if self.dir_key not in self._models:
            self._models[self.dir_key] = {}
        return self._models[self.dir_key]

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
    _models = {}
    _default_dir = 'q_learning'

    @property
    def is_json(self):
        return self.dtype == 'json'

    def get_model(self, n_pawns: int) -> dict:  # noqa
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                if self.is_json:
                    self.models[n_pawns] = json.load(open(filepath, 'r'))
                else:
                    self.models[n_pawns] = np.load(filepath, allow_pickle=True)
                logger.info(f"Using Q table at {filepath}")
                self._weight_update_timestamp[filepath] = get_file_modified_time(filepath)
            else:
                if self.is_json:
                    self.models[n_pawns] = {}
                else:
                    shape = get_grid_shape(n_pawns)
                    length = np.ravel_multi_index([s - 1 for s in shape], dims=shape)
                    self.models[n_pawns] = np.zeros(length, dtype=np.float32)
                logger.warn(f"No file at {filepath}, creating new Q table")
                self._weight_update_timestamp[filepath] = get_now(fmt=READABLE_DATE_FMT)

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

        q = self.get_model(state.n_pawns)
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
    cnn_hidden_dim: int = 64
    value_hidden_dim: int = 64
    policy_hidden_dim: int = 4
    num_blocks: int = 5
    double_value_head: bool = False
    board_flipping: bool = True
    separate_networks: bool = False

    def __repr__(self):
        text = (
            f"cnn_d={self.cnn_hidden_dim}"
            f", v_d={self.value_hidden_dim}"
            f", p_d={self.policy_hidden_dim}"
            f", blocks={self.num_blocks}"
        )
        if self.double_value_head:
            text += f", double_value"
        if self.board_flipping:
            text += f", board_flip"
        return text


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
    CNN model
    """

    def __init__(
        self,
        n_pawns: int,
        path=None,
        config: ModelConfig = None,
        device=None,
    ):
        super().__init__()

        config = config or ModelConfig()
        self.config = config

        self.device = device or default_device

        in_channels = get_num_channels(n_pawns, board_flipping=config.board_flipping)
        grid_dim = n_pawns + 2
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, config.cnn_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(config.cnn_hidden_dim) for _ in range(config.num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.cnn_hidden_dim, config.policy_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(config.policy_hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.policy_hidden_dim * grid_dim ** 2, n_pawns),
        )
        n_value_heads = 2 if config.double_value_head else 1
        self.value_head = nn.Sequential(
            nn.Conv2d(config.cnn_hidden_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(grid_dim ** 2, config.value_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.value_hidden_dim, n_value_heads),
            nn.Tanh()
        )

        self.load(path)

        self.eval()

        self.to(device=self.device)

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

    def load(self, obj: Union['Model', str | Path]):
        if isinstance(obj, Model):
            state_dict = obj.state_dict()
        elif obj is not None and os.path.exists(obj):
            state_dict = torch.load(obj, map_location=self.device)
        else:
            return
        self.load_state_dict(state_dict)
        self.to(self.device)

    def byte_size(self, human_readable=True) -> int | str:
        return get_model_size(self, human_readable)


class DeepQLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a deep neural network.

     _Q is a cache for all neural networks.
     _Q['/path/dir'][3] is the neural network in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """

    _models = {}
    _default_dir = 'deep_q_learning'

    def __init__(
        self,
        device=None,
        model_config: ModelConfig = None,
        **kwargs,
    ):
        """
        :param model_path: Path to the directory where the model is stored.
        """
        kwargs.setdefault('dtype', 'pt')
        super().__init__(**kwargs)
        self.model_config = model_config
        self.device = device or default_device

    def evaluate(
        self,
        state: State | list,
        torch_output: bool = False,
        check_game_over: bool = True,
        return_all: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | float]:
        if isinstance(state, list):
            state = State.from_list(state)

        if check_game_over and state.game_over():
            p = self.get_policy(state)
            v = get_reward(
                winner=state.winner,
                cur_player=state.cur_player,
                return_all=return_all,
            )
            if torch_output:
                p = torch.from_numpy(p).to(self.device)
                v = torch.from_numpy(v).to(self.device)
            return p, v

        model = self.get_model(n_pawns=state.n_pawns)

        channels = get_channels(state, board_flipping=model.config.board_flipping)
        x = torch.stack(channels, dim=0).unsqueeze(0).to(self.device)

        p, v = model(x)

        # Only 1 batch for now
        p = p[0]
        v = v[0]

        finished_mask = torch.asarray(state.finished[state.cur_player]).to(self.device)
        p[finished_mask] = -inf
        p = torch.softmax(p, dim=-1)

        if not torch_output:
            p = p.cpu().detach().numpy()
            v = v.cpu().detach().numpy()

        if not return_all:
            if len(v) > 1:
                v = v[state.cur_player]
            elif not torch_output:
                v = v.item()

        return p, v

    def is_pretrained(self, n_pawns: int) -> bool:
        return os.path.exists(self.get_filepath(n_pawns))

    def get_model(self, n_pawns: int) -> Model:
        if self.models.get(n_pawns) is None:
            filepath = self.get_filepath(n_pawns)
            if os.path.exists(filepath):
                self.models[n_pawns] = torch.load(filepath, weights_only=False).to(self.device)
                logger.info(f"Using model at {filepath}")
                self._weight_update_timestamp[filepath] = get_file_modified_time(filepath)
            else:
                self.models[n_pawns] = Model(
                    path=filepath,
                    n_pawns=n_pawns,
                    config=self.model_config,
                    device=self.device,
                )
                logger.warn(f"No file at {filepath}, creating new model")
                self._weight_update_timestamp[filepath] = get_now(fmt=READABLE_DATE_FMT)

        return self.models[n_pawns]

    # def set_model_path(self, path, n_pawns: int):
    #     model = self.get_model(n_pawns)
    #     model.load(path)

    def _dump(self, model, filepath: str):
        torch.save(obj=model, f=filepath)
