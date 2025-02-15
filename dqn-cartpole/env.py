import gymnasium as gym

env = gym.make('CartPole-v1',render_mode='human') # human, rgb_array
state, info = env.reset()

print(state, info)

print('action space', env.action_space, env.action_space.n)
print('state space dim',env.observation_space.shape[0])
print('state space: LOW', env.observation_space.low)
print('state space: HIGH', env.observation_space.high)

over = False

while not over:
    action = env.action_space.sample() 
    next_state, reward, terminated, truncated, info = env.step(action)
    over = terminated or truncated
    

env.close()