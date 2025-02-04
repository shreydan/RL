import numpy as np
import random
from game import TreatsEnv, pygame
from tqdm.auto import tqdm
from types import SimpleNamespace
from pathlib import Path

class QLearning:
    def __init__(self, config):
        self.config = config
        self.env = TreatsEnv(render=False)

        self.state_dim = self.env.grid_size ** 2
        self.action_dim = len(self.env.actions)
        
        self.q_table = np.zeros((self.state_dim, self.action_dim))

    def sample_action(self):
        return random.randint(0,self.action_dim-1)

    def greedy_policy(self, state):
        """
        takes action based on the maximum state value
        """
        action = np.argmax(self.q_table[state])
        return action
    
    def epsilon_greedy_policy(self, state, eps):
        """
        initially, eps is high so there will be more sampling i.e. exploration
        as the q-table gets updated, the eps decays
        as it decays more and more, the likelihood of greedy_policy increases
        so it will start to select best possible action based on value of the state
        """
        random_num = random.uniform(0,1)
        if random_num > eps:
            # apply greedy policy
            return self.greedy_policy(state)
        else:
            return self.sample_action()
        

    def train(self):
        """
        eps decay formula:

        eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode)

        episodes: somewhat like epochs?
        steps: max number of actions per episode to not let it go forever
        gamma: discount rate
        lr: learning rate

        q_table updation formula:

        q[s][a] = q[s][a] + lr * (reward + gamma * max(q[next_s]) - q[s][a])
        """

        if Path('q_table.npy').exists():
            print('learned q_table exists, evaluating instead...')
            self.eval()
            return

        min_eps = self.config.min_eps
        max_eps = self.config.max_eps
        decay_rate = self.config.decay_rate
        lr = self.config.lr
        gamma = self.config.gamma

        for episode in tqdm(range(self.config.num_episodes)):

            eps = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode)

            step = 0
            state = self.env.reset()

            done = False

            for step in range(self.config.max_steps):

                action = self.epsilon_greedy_policy(state, eps)
                new_state, reward, done = self.env.step(action)

                self.q_table[state, action] = self.q_table[state, action] + lr * (
                    reward + gamma * np.max(self.q_table[new_state]) - self.q_table[state, action]
                )

                if done:
                    break

                state = new_state

        np.save('q_table.npy',self.q_table)

    def eval(self):
        self.env = TreatsEnv(render=True)

        self.q_table = np.load('q_table.npy')

        episode_rewards = []

        for episode in range(self.config.eval_episodes):
            state = self.env.reset()
            ep_rewards = 0
            for step in range(self.config.max_steps):
                self.env.show()
                action = self.greedy_policy(state)
                new_state, reward, done = self.env.step(action)

                ep_rewards += reward

                if done:
                    self.env.show()
                    pygame.time.wait(1000)
                    break

                state = new_state

            episode_rewards.append(ep_rewards)

        episode_rewards = np.array(episode_rewards)
        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)

        print(f"reward: {mean=:.3f}, {std=:.3f}")

            

if __name__ == '__main__':
    config = SimpleNamespace(
        num_episodes = 10000,
        lr = 0.7,
        eval_episodes = 25,
        max_steps = 99,
        gamma = 0.95,
        max_eps = 1.0,
        min_eps = 0.05,
        decay_rate = 0.0005
    )

    learner = QLearning(config)
    learner.train()