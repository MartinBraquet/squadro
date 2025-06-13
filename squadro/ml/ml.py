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


def assert_models_unequal(model1, model2):
    try:
        assert_models_equal(model1, model2)
    except AssertionError:
        return None
    raise AssertionError("Models are equal")


def assert_models_equal(model1, model2):
    params1 = model1.parameters() if isinstance(model1, nn.Module) else model1
    params2 = model2.parameters() if isinstance(model2, nn.Module) else model2

    for p1, p2 in zip(params1, params2):
        torch.testing.assert_close(p1, p2)
