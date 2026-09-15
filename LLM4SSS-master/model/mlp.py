import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, num_layers, activation, dropout=0.0):
        super(Mlp, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        layers = [nn.Linear(self.dim_in, self.hidden_dim), nn.Dropout(self.dropout)]
        for _ in range(self.num_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), self.activation, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, self.dim_out)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)