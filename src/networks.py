import torch
import torch.nn as nn
from fastkan import FastKAN as KAN
from utils import apply_leverage_constraints
from config import CONFIG as config


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class PRICE_NETWORK(nn.Module):
    def __init__(self, insize, outsize, HLsizes=[]):
        super().__init__()
        self.model = KAN(layers_hidden=[insize] + HLsizes + [outsize],
                         grid_min=-1.5,
                         grid_max=1.5,
                         num_grids=10
                         )
        self.outact = torch.nn.Softplus(beta=1)

    def forward(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] / 10.0
        x[:, 1:] = x[:, 1:] * 10.0
        x = self.model(x)
        x = self.outact(x)
        return x


class ActionNetwork(nn.Module):
    def __init__(self, input_size_net, hidden_sizes, output_size_net, dropout_prob):
        super().__init__()
        self.activation = Swish()
        self.dropout_prob = dropout_prob
        self.input_layer = nn.Linear(input_size_net, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]) for i in range(1, len(hidden_sizes))
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size_net)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_prob) for _ in range(len(hidden_sizes))
        ])

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropouts[0](x)

        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = self.activation(layer(x))
            x = self.dropouts[i + 1](x)
            if residual.shape == x.shape:
                x = x + residual

        raw_output = self.output_layer(x)

        return raw_output
