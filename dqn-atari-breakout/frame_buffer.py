import numpy as np
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt

class FrameBuffer:
    def __init__(self, frame_limit=4):
        self.frames = deque(maxlen=frame_limit)

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_AREA)
        return frame / 255.

    def add(self, frame):
        frame = self.preprocess(frame)
        self.frames.append(frame)

    def get_stack(self):
        stack = np.stack(self.frames)
        return torch.from_numpy(stack).float()

    def show_stack(self):
        stack = self.get_stack()
        plt.imshow(torch.hstack([x.squeeze() for x in stack.split(1)]),cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()


class MultiEnvFrameBuffer:
    def __init__(self, num_envs, frame_limit=4):
        self.num_envs = num_envs
        self.frames = {env_id: deque(maxlen=frame_limit) for env_id in range(self.num_envs)}

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_AREA)
        return frame

    def add(self, env_frames):
        for env_id in range(self.num_envs):
            frame = self.preprocess(env_frames[env_id])
            self.frames[env_id].append(frame)

    def get_stack(self):
        stack = np.stack([np.stack(frames) for _,frames in self.frames.items()])
        return torch.from_numpy(stack).to(dtype=torch.uint8)

    def show_stack(self):
        stack = self.get_stack()
        for env_id in range(self.num_envs):
            env_stack = stack[env_id] 
            im = torch.hstack([x.squeeze() for x in env_stack.split(1)])
            # print([_.shape for _ in im])
            plt.imshow(im,cmap='gray')
            plt.axis('off')
            plt.title(f'env {env_id}')
            plt.show()
            plt.close()

