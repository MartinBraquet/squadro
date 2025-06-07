import io

import torch
from torch import nn

from squadro.tools.disk import human_readable_size


def get_model_size(model: nn.Module, human_readable=True) -> int | str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_in_bytes = buffer.tell()
    if human_readable:
        size_in_bytes = human_readable_size(size_in_bytes)
    return size_in_bytes
