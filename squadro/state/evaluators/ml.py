from dataclasses import dataclass

from squadro.tools.logs import logger


@dataclass
class ModelConfig:
    cnn_hidden_dim: int = 64
    value_hidden_dim: int = 64
    policy_hidden_dim: int = 4
    num_blocks: int = 5
    double_value_head: bool = False
    board_flipping: bool = False
    separate_networks: bool = True

    def __post_init__(self):
        if self.double_value_head and self.separate_networks:
            raise ValueError("Cannot use separate networks with double value head")
        if self.separate_networks and self.board_flipping:
            logger.warn("Using separate networks with board flipping, not recommended")

    def __repr__(self):
        text = ''
        if self.double_value_head:
            text += f"double_value"
        if self.board_flipping:
            text += f"board_flip"
        if self.separate_networks:
            text += f"separate_networks"
        text += (
            f", cnn_d={self.cnn_hidden_dim}"
            f", v_d={self.value_hidden_dim}"
            f", p_d={self.policy_hidden_dim}"
            f", blocks={self.num_blocks}"
        )
        return text

    def copy(self):
        return ModelConfig(**self.__dict__)
