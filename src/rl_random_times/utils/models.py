import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity()):

    # preallocate layers list
    layers = []

    for j in range(len(sizes)-1):

        # actiavtion function
        act = activation if j < len(sizes)-2 else output_activation

        # linear layer
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]

    # Sequential model with given layers
    return nn.Sequential(*layers)
