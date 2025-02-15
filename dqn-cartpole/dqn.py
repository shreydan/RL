import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.state_space, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_space),
        )

    def forward(self, x):
        return self.model(x)