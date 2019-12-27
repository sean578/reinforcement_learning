import numpy as np
import gym
from q_learning import QLearning
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
actions = (0, 1, 2, 3)
env_ranges = list(zip(env.observation_space.low, env.observation_space.high))
num_observations = len(env_ranges)
print('Ob space:\t\t', env.observation_space)
print('Action space:\t', env.action_space)
print('Reward range:\t', env.reward_range)

env_ranges = [(-0.5, 0.5),
              (0, 1),
              (-0.5, 0.5),
              (-0.5, 0.5),
              (-0.4, 0.4),
              (-0.4, 0.4),
              (-1, 1),
              (-1, 1)]

print('\nObservation ranges:')
for ob in enumerate(env_ranges):
    print(ob[0], ob[1])

num_bins = [3, 20, 3, 6, 6, 6, 3, 3]
num_pos_actions = len(actions)

q_learning = QLearning(env=env,
                       num_bins=num_bins,
                       num_pos_actions=num_pos_actions,
                       env_ranges=env_ranges,
                       discount=0,
                       episodes=0,
                       epsilon=None,
                       lr=None,
                       USE=True)

env = gym.make('LunarLander-v2')
q_learning.q_table = np.load('./data_lunarlander/0_9000.npy')

for _ in range(10):

    obs = q_learning.reset_state()  # Reset the environment and get the initial

    done = False
    while not done:

        action = q_learning.action_to_maximise_q(obs)
        obs, reward, done = q_learning.perform_sim_step(action)
        print(obs, reward, done)
        q_learning.env.render()

env.close()
