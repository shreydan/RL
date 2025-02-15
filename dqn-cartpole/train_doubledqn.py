import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
from dqn import DQN
from replay_buffer import ReplayBuffer

env = gym.make('CartPole-v1')
num_episodes = 1000

eps = 0.9
min_eps = 0.05
decay_rate = 0.998
discount_rate = 0.99
update_freq = 100
batches_per_ep = 8

buffer = ReplayBuffer(batch_size=64, maxlen=int(1e5), minlen=int(1e2))  # Ensure integers for maxlen and minlen

N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
HIDDEN_SIZE = 128
print(N_ACTIONS, STATE_DIM)

def update_target_model(q_online, q_target):
    online_sd = q_online.state_dict()
    q_target.load_state_dict(online_sd)
    return q_online, q_target

q_online = DQN(state_space=STATE_DIM, action_space=N_ACTIONS, hidden_size=HIDDEN_SIZE)
q_target = DQN(state_space=STATE_DIM, action_space=N_ACTIONS, hidden_size=HIDDEN_SIZE)
q_online, q_target = update_target_model(q_online, q_target)


def epsilon_greedy(state, eps):
    if np.random.uniform() < eps:  # explore
        action = env.action_space.sample()
        return action
    else:  # exploit
        state = torch.from_numpy(state).float().unsqueeze(0)  # batch, state
        q_online.eval()
        with torch.no_grad():
            action_q_values = q_online(state)
        q_online.train()
        return action_q_values.argmax(dim=1).item()  # best action with highest q-value


episode_rewards = []
episode_losses = []

optim = torch.optim.Adam(q_online.parameters(), lr=1e-4)

prog_bar = tqdm(range(int(num_episodes)))
for episode in prog_bar:
    state, info = env.reset()  # This should work with gymnasium
    done = False
    total_reward = 0.
    ep_loss = 0.
    step = 0
    while not done:

        action = epsilon_greedy(state, eps)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step += 1

        if buffer.is_ready():
            for _ in range(batches_per_ep):
                batch = buffer.sample()
                states = batch['state']
                actions = batch['action']
                rewards = batch['reward']
                next_states = batch['next_state']
                dones = batch['done']


                # DOUBLE DQN
                # - predict q-values from current states using q_online
                # - predict best actions from next_states using q_online
                # - next q-values from current states using q_target from the best actions
                # - target q-values = rewards + gamma * (1-done) * next-q-values
                # - loss: target q-values VS predicted q-values

                predicted_q_values = q_online(states).gather(1, actions.unsqueeze(1)).flatten()

                best_actions_online = q_online(next_states).argmax(dim=1)
                next_q_values = q_target(next_states).gather(1, best_actions_online.unsqueeze(1)).flatten()

                target_q_values = rewards + discount_rate * (1-dones) * next_q_values


                loss = F.smooth_l1_loss(predicted_q_values, target_q_values)
                loss.backward()

                ep_loss += loss.item()

            optim.step()
            optim.zero_grad()

    ep_loss /= (step * batches_per_ep)
    prog_bar.set_description(f"loss: {ep_loss:.3f} | reward: {total_reward} | rb_size: {len(buffer)} | eps: {eps:.3f}")

    episode_rewards.append(total_reward)
    episode_losses.append(ep_loss)

    if episode % update_freq == 0:
        q_online, q_target = update_target_model(q_online, q_target)
        
    eps = max(min_eps, eps * decay_rate)

print('reward', np.array(episode_rewards).mean(), np.array(episode_rewards).std())
print('loss', np.array(episode_losses).mean(), np.array(episode_losses).std())

# Save the model
online_sd = q_online.state_dict()
torch.save(online_sd, 'cartpole.pt')