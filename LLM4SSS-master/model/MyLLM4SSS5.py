import torch
from torch import nn
from utils.conf import Configuration


class MyLLM4SSS5(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS5, self).__init__()

        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.selected_device = self.conf.getEntry("GPUs")
        self.batch_size = self.conf.getEntry('batch_size')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')

        self.mlp_layers = conf.getEntry('mlp_layers')
        self.mlp_dim = conf.getEntry('mlp_dim')
        self.dropout = conf.getEntry('dropout')

        # Activation function
        if self.conf.getEntry('activation') == 'relu':
            self.activation = nn.ReLU()
        elif self.conf.getEntry('activation') == 'tanh':
            self.activation = nn.Tanh()
        elif self.conf.getEntry('activation') == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.conf.getEntry('activation') == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        # Define the layers
        self.initial_layer = nn.Linear(self.len_series, self.mlp_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Define the residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.mlp_layers - 3):
            self.blocks.append(self._residual_block(self.mlp_dim, self.activation, self.dropout))

        self.final_layer = nn.Linear(self.mlp_dim, self.len_series)
        self.output_layer = nn.Linear(self.len_series, self.len_reduce)

    def _residual_block(self, in_dim, activation, dropout):
        """Residual block with a skip connection."""
        block = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        return block

    def forward(self, x):
        # Initial layer
        out = self.activation(self.initial_layer(x))
        out = self.dropout_layer(out)

        # Residual blocks with skip connections
        for block in self.blocks:
            residual = out
            out = block(out)
            out += residual  # Adding the residual connection

        # Final layers
        out = self.activation(self.final_layer(out))
        out = self.dropout_layer(out)
        out = self.output_layer(out)

        return out
