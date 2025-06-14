# Ideally, this would go into state.evaluators.ml, but some previous models stored in disk
# point to this file, so we need to keep it here for backwards compatibility reasons.

import os
from multiprocessing.managers import DictProxy, ArrayProxy  # noqa
from pathlib import Path
from typing import Union

import torch
from torch import nn

from squadro.ml.channels import get_num_channels
from squadro.ml.ml import get_model_size
from squadro.state.evaluators.ml import ModelConfig

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        in_channels = get_num_channels(
            n_pawns,
            board_flipping=config.board_flipping,
            separate_networks=config.separate_networks,
        )
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

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
       Evaluate the neural network
        """
        x = self.input_conv(x)
        x = self.res_blocks(x)
        p = self.policy_head(x)
        value = self.value_head(x)
        return p, value

    def load(self, obj: Union['Model', str | Path]):
        state_dict = None
        if isinstance(obj, Model):
            state_dict = obj.state_dict()
        elif obj is not None and os.path.exists(obj):
            state_dict = torch.load(obj, weights_only=False, map_location=self.device)

        if state_dict:
            self.load_state_dict(state_dict)

        self.to(self.device)

    def save(self, filepath: str | Path, weights_only=False):
        obj = self.state_dict() if weights_only else self
        # mkdir(filepath)
        torch.save(obj=obj, f=filepath)

    def byte_size(self, human_readable=True) -> int | str:
        return get_model_size(self, human_readable)
