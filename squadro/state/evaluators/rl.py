import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from numpy.typing import NDArray

from squadro.evaluators.evaluator import default_device, Model
from squadro.ml.channels import get_channels
from squadro.state.evaluators.base import Evaluator
from squadro.state.evaluators.ml import ModelConfig
from squadro.state.state import State
from squadro.tools.constants import inf
from squadro.tools.dates import get_file_modified_time, get_now, READABLE_DATE_FMT
from squadro.tools.logs import logger
from squadro.tools.state import get_grid_shape, state_to_index, get_reward

HF_REPO_ID = "martin-shark/squadro"


class _RLEvaluator(Evaluator, ABC):
    _default_dir = 'default'
    _weight_update_timestamp = defaultdict(lambda: 'unknown')
    _online_models = ()

    def __init__(self, model_path: str | Path = None, dtype='json'):
        """
        :param model_path: Path to the directory where the model(s) is stored.
        """
        self._model_path = model_path
        if self._model_path:
            self.model_path = Path(self._model_path)
        else:
            logger.info(f"No model path specified, using online pre-trained models.")
            dir_path = self._default_dir
            hf_files = [f for f in list_repo_files(HF_REPO_ID) if f.startswith(dir_path)]
            path = None
            for f in hf_files:
                path = hf_hub_download(repo_id=HF_REPO_ID, filename=f)
            assert path is not None, f"No files found in repo {HF_REPO_ID} for dir {dir_path}"
            self.model_path = Path(path).parent
        self.dtype = dtype

    def get_weight_update_timestamp(self, n_pawns: int):
        return self._weight_update_timestamp[self.get_filepath(n_pawns)]

    def erase(self, n_pawns: int, filepath=None):
        filepath = filepath or self.get_filepath(n_pawns)
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
    def models(self) -> dict:
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

    def check_online_model(self, n_pawns: int):
        if not self._model_path:
            raise ValueError(
                "When `model_path` is not specified, a pre-trained model must be loaded."
                f" But there is no pre-trained model available for {self._default_dir}"
                f" with {n_pawns} pawns. Available number of pawns: {self._online_models}."
            )


class QLearningEvaluator(_RLEvaluator):
    """
    Evaluate a state using a Q-lookup table.

     _Q is a cache for all Q-tables.
     _Q['/path/dir'][3] is the Q-table in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')
    """
    _models = {}
    _default_dir = 'q_learning'
    _online_models = (2, 3)

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
                self.check_online_model(n_pawns)
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


class DeepQLearningEvaluatorMultipleGrids(_RLEvaluator):
    """
    Evaluate a state using a deep neural network.

     _models is a cache for all neural networks.
     _models['/path/dir'][3] is the neural network in '/path/dir' for 3 pawns (stared in file '/path/dir/model_3.json')

     Remember that an Evaluator does not depend on the number of pawns. Multiple models can be
     attached to the same evaluator.

     `model_path` is the dir where the models for all pawns are stored.
    """

    _models = {}
    _default_dir = 'deep_q_learning'
    _online_models = (3, 4, 5)

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

        # This can only be used to initialize a model
        self.model_config: ModelConfig = model_config or ModelConfig()

        self.device = device or default_device

    def evaluate(
        self,
        state: State | list,
        torch_output: bool = False,
        check_game_over: bool = True,
        return_all: bool = False,
    ) -> tuple[NDArray[np.float64] | torch.Tensor, NDArray[np.float64] | float | torch.Tensor]:
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
                v = torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                v = v.to(self.device)
            return p, v

        model = self.get_model(n_pawns=state.n_pawns, player=state.cur_player)

        channels = get_channels(
            state,
            board_flipping=model.config.board_flipping,
            separate_networks=model.config.separate_networks,
        )
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
        key = self.get_key(n_pawns, player=0)
        return os.path.exists(self.get_filepath(key))

    def get_weight_update_timestamp(self, n_pawns: int):
        return super().get_weight_update_timestamp(self.get_key(n_pawns, player=0))

    def is_separate_networks(self, n_pawns: int) -> bool:
        return self.model_config.separate_networks

    def get_key(self, n_pawns, player):
        if self.is_separate_networks(n_pawns):
            assert player is not None, "Player must be specified for separate networks"
            key = f"{n_pawns}_{player}"
        else:
            key = n_pawns
        return key

    @staticmethod
    def extract_from_key(key):
        if isinstance(key, int):
            return dict(n_pawns=key)
        elif isinstance(key, str):
            n_pawns, player = key.split("_")
            return dict(n_pawns=int(n_pawns), player=int(player))
        raise ValueError(f"Invalid key: {key}")

    def get_model(self, n_pawns: int, player: int = None) -> Model:
        key = self.get_key(n_pawns, player=player)
        if self.models.get(key) is None:
            filepath = self.get_filepath(key)
            if os.path.exists(filepath):
                logger.info(f"Using pre-trained model at {filepath}")
                model = torch.load(filepath, weights_only=False, map_location=self.device)
                model.device = self.device
                self.model_config = model.config
                self.models[key] = model
                self._weight_update_timestamp[filepath] = get_file_modified_time(filepath)
            else:
                self.check_online_model(n_pawns)
                logger.warn(f"No file at {filepath}, creating new model")
                self.models[key] = Model(
                    n_pawns=n_pawns,
                    config=self.model_config,
                    device=self.device,
                )
                self._weight_update_timestamp[filepath] = get_now(fmt=READABLE_DATE_FMT)

        return self.models[key]

    def _dump(self, model: Model, filepath: str):
        model.save(filepath)

    def erase(self, n_pawns: int, filepath=None):
        if self.is_separate_networks(n_pawns) and not filepath:
            for player in range(2):
                key = self.get_key(n_pawns, player=player)
                filepath = self.get_filepath(key)
                super().erase(n_pawns=n_pawns, filepath=filepath)
            return

        super().erase(n_pawns=n_pawns, filepath=filepath)

    def load_weights(self, other: 'DeepQLearningEvaluator'):
        """
        Loads the weights from the specified DeepQLearningEvaluator instance and applies
        them to the current model. This function does not return any value but modifies
        the internal state of the model to reflect the newly loaded weights.

        Arguments:
            other: DeepQLearningEvaluator
                An instance of DeepQLearningEvaluator containing the weights to be
                loaded and applied to the model.

        Returns:
            None
        """
        for k, model in other.models.items():
            key_info = self.extract_from_key(k)
            self.get_model(**key_info).load(model)


class DeepQLearningEvaluator(DeepQLearningEvaluatorMultipleGrids):
    """
    Evaluator for a single grid size (specific `n_pawns`), and having a single model.
    Since it infers the (unique) model, it allows for loading
    such a pre-trained model without having to pass its model config again.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._separate_networks = {}

    def is_separate_networks(self, n_pawns: int) -> bool:
        if isinstance(self._separate_networks, bool):
            return self._separate_networks
        n_pawns = int(str(n_pawns)[0])
        if self._separate_networks.get(n_pawns) is None:
            model_path = self.model_path
            files = os.listdir(model_path) if os.path.exists(model_path) else []
            if {f'model_{n_pawns}_0.pt', f'model_{n_pawns}_1.pt'}.issubset(files):
                self._separate_networks[n_pawns] = True
            elif {f'model_{n_pawns}.pt'}.issubset(files):
                self._separate_networks[n_pawns] = False
            else:
                self._separate_networks[n_pawns] = super().is_separate_networks(n_pawns)
        return self._separate_networks[n_pawns]

    def get_model(self, n_pawns: int, player: int = None) -> Model:
        return super().get_model(n_pawns=n_pawns, player=player or 0)
