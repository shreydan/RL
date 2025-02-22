import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
import torch
import numpy as np
from frame_buffer import FrameBuffer
from dqn import DQN
import cv2

def create_video(source, fps=60, output_name='breakout'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()

q_online = DQN()
q_online.load_state_dict(torch.load('breakout.pt',weights_only=True,map_location='cpu'))

eval_env = gym.make('ALE/Breakout-v5',frameskip=4,render_mode='human')
frame_buffer = FrameBuffer()
state, info = eval_env.reset()
for _ in range(4): frame_buffer.add(state)
results = [state,]
q_online.eval()
while True:
    with torch.no_grad():
        state = frame_buffer.get_stack()
        action = q_online(state.unsqueeze(0)).argmax(dim=1).cpu().flatten().item()
    next_state, reward, terminated, truncated, info = eval_env.step(action)
    results.append(next_state)
    frame_buffer.add(next_state)
    if terminated or truncated:
        break

create_video(results)