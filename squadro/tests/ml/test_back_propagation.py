from unittest import TestCase

import numpy as np
import torch
from torch import optim, nn

from squadro.tools.probabilities import set_seed


class _Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        return x


class _Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        return x


class TestBackPropagation(TestCase):
    def setUp(self):
        set_seed()

    def test_grad(self):
        model = _Model1()
        # model.train() is the default
        self.assertTrue(model.training)
        for p in list(model.parameters()):
            self.assertTrue(p.requires_grad)

        x = torch.randn(2)
        y = torch.randn(1)
        z = model(x)

        for p in list(model.parameters()):
            self.assertIsNone(p.grad)
            self.assertIsNone(p.grad_fn)

        loss = (z - y) ** 2
        loss.backward()

        for p in list(model.parameters()):
            self.assertIsNotNone(p.grad)

        optim.SGD(model.parameters()).zero_grad()

        model.eval()
        self.assertFalse(model.training)
        model(torch.randn(2))
        for p in list(model.parameters()):
            self.assertIsNone(p.grad)

    def test_MLP_1_layer(self):
        model = _Model1()

        param_w, param_b = list(model.parameters())
        b = param_b.data.clone().numpy()
        w = param_w.data.clone().numpy()

        lr = 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr)
        l_mse = nn.MSELoss()

        x = torch.randn(2)
        y = torch.randn(1)
        z = model(x)

        optimizer.zero_grad()
        loss = l_mse(z, y)
        loss.backward()
        optimizer.step()

        x = x.detach().numpy()
        y = y.detach().numpy()
        z = z.detach().numpy()

        grad_z = 2 * (z - y)  # dL/dz
        grad_b = grad_z * 1  # dL/db
        grad_w = grad_z * x  # dL/dw

        b -= lr * grad_b
        w -= lr * grad_w

        param_w, param_b = list(model.parameters())

        np.testing.assert_allclose(grad_b, param_b.grad.clone().numpy())
        np.testing.assert_allclose(grad_w, param_w.grad.clone().numpy().squeeze(0))

        np.testing.assert_allclose(b, param_b.data.detach().numpy())
        np.testing.assert_allclose(w, param_w.data.detach().numpy(), rtol=1e-6)

    def test_MLP_2_layer(self):
        model = _Model2()

        b1 = model.f[0].bias.data.clone().numpy()
        w1 = model.f[0].weight.data.clone().numpy()
        b = model.f[2].bias.data.clone().numpy()
        w = model.f[2].weight.data.clone().numpy()

        lr = 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr)
        l_mse = nn.MSELoss()

        x = torch.randn(2)
        y = torch.randn(1)
        z = model(x)

        optimizer.zero_grad()
        loss = l_mse(z, y)
        loss.backward()
        optimizer.step()

        z1 = model.f[0](x).detach()
        a1 = model.f[1](z1).detach().numpy()

        x = x.detach().numpy()
        y = y.detach().numpy()
        z = z.detach().numpy()

        dz_da1 = w
        da1_dz1 = torch.where(z1 > 0, 1, 0).detach().numpy()

        grad_z = 2 * (z - y)
        grad_b = grad_z * 1
        grad_w = grad_z * a1
        grad_z1 = grad_z * dz_da1 * da1_dz1
        grad_b1 = grad_z1.squeeze(0)
        grad_w1 = grad_z1.T @ x[np.newaxis, :]

        b1 -= lr * grad_b1
        w1 -= lr * grad_w1
        b -= lr * grad_b
        w -= lr * grad_w

        np.testing.assert_allclose(b1, model.f[0].bias.data.detach().numpy())
        np.testing.assert_allclose(w1, model.f[0].weight.data.detach().numpy(), rtol=1e-5)
        np.testing.assert_allclose(b, model.f[2].bias.data.detach().numpy())
        np.testing.assert_allclose(w, model.f[2].weight.data.detach().numpy(), rtol=1e-5)
