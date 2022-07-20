import torch
import torch.nn as nn

class State_Generator(nn.Module):
    def __init__(self, state_dim, action):
        super(State_Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, state_dim)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.main(x)
        return x