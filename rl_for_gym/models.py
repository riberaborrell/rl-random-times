import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, d_in, hidden_sizes, d_out, activation=torch.tanh, seed=None):
        super(FeedForwardNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # hidden_sizes
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1
        self.d_out = d_out

        # activation function
        self.activation = activation

    def forward(self, x):
        layer_d_in = self.d_in
        for i, size in enumerate(self.hidden_sizes):

            # linear layer
            layer_d_out = size
            x = nn.Linear(layer_d_in, layer_d_out, bias=True)(x)

            # activation
            x = self.activation(x)

            # update layer input size
            layer_d_in = size

        # last layer without activation
        layer_d_out = self.d_out
        x = nn.Linear(layer_d_in, layer_d_out, bias=True)(x)
        return x

class TwoLayerNN(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh):
        super(TwoLayerNN, self).__init__()

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out

        # two linear layers
        self.linear1 = nn.Linear(d_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x
