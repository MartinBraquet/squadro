from unittest import TestCase

import numpy as np
import torch


class _GradFunction:

    def backward(self):
        ...


class _SavingBackward(_GradFunction):
    def __init__(self, *tensors):
        super().__init__()
        self.saved_tensors = tensors


class _MulBackward(_SavingBackward):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def backward(self, grad=None):
        if grad is None:
            grad = 1.0
        if isinstance(self.saved_tensors[0], _Tensor):
            self.saved_tensors[0].backward(_Tensor(grad * self.saved_tensors[1]))
        if isinstance(self.saved_tensors[1], _Tensor):
            self.saved_tensors[1].backward(_Tensor(grad * self.saved_tensors[0]))


class _AddBackward(_SavingBackward):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def backward(self, grad=None):
        for t in self.saved_tensors:
            if isinstance(t, _Tensor):
                t.backward(grad)


class _Tensor:
    """
    Custom minimal implementation of a tensor object.
    """

    data: np.array

    def __init__(self, data):
        # This is the actual numerical content (e.g., a matrix of floats).
        if isinstance(data, _Tensor):
            data = data.data
        self.data = np.array(data)

        # Stores the gradient (∂output/∂this tensor).
        self.grad = None

        # This links to the computation graph node (e.g., AddBackward0, MulBackward0, etc.).
        # It's None for leaf tensors (like inputs).
        self.grad_fn = None

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @staticmethod
    def _to_data(obj):
        if isinstance(obj, _Tensor):
            return obj.data
        return obj

    def __add__(self, other):
        tensor = _Tensor(self.data + self._to_data(other.data))
        tensor.grad_fn = _AddBackward(self, other)
        return tensor

    def __mul__(self, other):
        tensor = _Tensor(self.data * self._to_data(other))
        tensor.grad_fn = _MulBackward(self, other)
        return tensor

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Tensor({self.data})"

    def backward(self, grad=None):
        if self.is_leaf:
            self.grad = grad
        if self.grad_fn is not None:
            self.grad_fn.backward(grad)


class Optimizer:
    def __init__(self, params, lr=.1):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad.data


class TestCustomTensor(TestCase):
    def test(self):
        values = 2.0, 3.0, 4.0, 5.0

        custom_values = [_Tensor(v) for v in values]
        torch_values = [torch.tensor(v, requires_grad=True) for v in values]

        def forward(a, b, w, x):
            y = w * x
            c = a * b
            out = 2 * (c + y)
            out.backward()
            self.assertIsNone(y.grad)  # None, as non-leaf tensors
            self.assertIsNone(c.grad)
            return out

        z = forward(*custom_values)
        tz = forward(*torch_values)

        np.testing.assert_allclose(z.data, tz.detach().data)

        for v, tv in zip(custom_values, torch_values):
            np.testing.assert_allclose(v.grad.data, tv.grad.detach().data)

        optimizer = Optimizer(params=custom_values[::2], lr=0.1)
        optimizer.step()

        to = torch.optim.SGD(params=torch_values[::2], lr=0.1)
        to.step()

        for v, tv in zip(custom_values, torch_values):
            np.testing.assert_allclose(v.data, tv.detach().data)
