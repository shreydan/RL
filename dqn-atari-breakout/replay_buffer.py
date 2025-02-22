import torch
import numpy as np

class MultiEnvReplayBuffer:
    def __init__(self, num_envs, state_shape=(4, 84, 84), max_size=1_000_000, batch_size=64, device="cuda"):
        self.num_envs = num_envs
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory with PyTorch tensors
        self.states = torch.zeros((max_size, *state_shape), dtype=torch.uint8)
        self.next_states = torch.zeros((max_size, *state_shape), dtype=torch.uint8)
        self.actions = torch.zeros((max_size,), dtype=torch.uint8)
        self.rewards = torch.zeros((max_size,), dtype=torch.float32)
        self.dones = torch.zeros((max_size,), dtype=torch.bool)

    def add(self, state, action, reward, next_state, done):
        """Adds experiences efficiently using preallocated tensors."""
        idx = np.arange(self.ptr, self.ptr + self.num_envs) % self.max_size

        self.states[idx] = state
        self.actions[idx] = torch.from_numpy(action).to(dtype=torch.uint8)
        self.rewards[idx] = torch.from_numpy(reward).float()
        self.next_states[idx] = next_state
        self.dones[idx] = torch.from_numpy(done).to(dtype=torch.bool)

        self.ptr = (self.ptr + self.num_envs) % self.max_size
        self.size = min(self.size + self.num_envs, self.max_size)

    def sample(self):
        """Samples a batch and moves it efficiently to the GPU."""
        idxs = torch.randint(0, self.size, (self.batch_size,), dtype=torch.long)

        return {
            "state": self.states[idxs].float().div_(255.).to(self.device, non_blocking=True),
            "action": self.actions[idxs].long().to(self.device, non_blocking=True),
            "reward": self.rewards[idxs].to(self.device, non_blocking=True),
            "next_state": self.next_states[idxs].float().div_(255.).to(self.device, non_blocking=True),
            "done": self.dones[idxs].float().to(self.device, non_blocking=True),
        }

    def is_ready(self):
        return self.size >= self.batch_size

    def __len__(self):
        return self.size