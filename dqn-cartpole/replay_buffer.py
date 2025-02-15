import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, batch_size, maxlen=1e5, minlen=1e2):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.buffer = deque(maxlen=int(maxlen))

    def add(self, state, action, reward, next_state, done):
        state = list(state)
        next_state = list(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in batch]

        return {
            'state': torch.tensor([b[0] for b in batch]).float(),
            'action': torch.tensor([b[1] for b in batch]).long(),
            'reward': torch.tensor([b[2] for b in batch]).float(),
            'next_state': torch.tensor([b[3] for b in batch]).float(),
            'done': torch.tensor([b[4] for b in batch]).float(),
        }

    def is_ready(self):
        return len(self.buffer) >= self.batch_size and len(self.buffer) >= self.minlen
    
    def __len__(self):
        return len(self.buffer)
