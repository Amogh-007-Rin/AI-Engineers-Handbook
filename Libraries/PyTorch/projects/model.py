"""Small deterministic PyTorch training and checkpoint fixture."""

import io
import torch
from torch import nn


def data():
    x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    return x, 2 * x + 1


def train(seed: int = 7, steps: int = 200):
    torch.manual_seed(seed)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    x, y = data()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward(); optimizer.step()
    return model


def round_trip(model):
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = nn.Linear(1, 1); restored.load_state_dict(torch.load(buffer, weights_only=True)); restored.eval()
    return restored
