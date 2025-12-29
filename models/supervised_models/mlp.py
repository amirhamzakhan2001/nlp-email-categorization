# models/supervised_models/mlp.py

import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list[int],
        dropout: float,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)