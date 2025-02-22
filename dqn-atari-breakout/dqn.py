import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4,32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,4)
        )
    def forward(self, x):
        return self.model(x)