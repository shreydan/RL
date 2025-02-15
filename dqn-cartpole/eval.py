import gymnasium as gym
import torch
from dqn import DQN

env = gym.make('CartPole-v1',render_mode='human') # human, rgb_array


N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
model = DQN(state_space=STATE_DIM, action_space=N_ACTIONS, hidden_size=128)
model.load_state_dict(torch.load('cartpole.pt',weights_only=True))

def get_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    with torch.inference_mode():
        return model(state).flatten().argmax().item()

for _ in range(1):
    state, info = env.reset()
    over = False
    step=0
    while not over:
        step+=1
        action = get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        over = terminated or truncated
        state = next_state
    print(step)
    
env.close()