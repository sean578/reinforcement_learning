import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)
from q_learning import QLearning
import math

env = gym.make('CartPole-v0')
actions = (0, 1)
env_ranges = list(zip(env.observation_space.low, env.observation_space.high))
num_observations = 2  # len(env_ranges)

obs_to_use = (2, 3)

env_ranges[2] = [-math.radians(50), math.radians(50)]  # set angle range
env_ranges[3] = [-0.5, 0.5]
env_ranges = [env_ranges[i] for i in obs_to_use]

print('Ob space:\t\t', env.observation_space)
print('Action space:\t', env.action_space)
print('Reward range:\t', env.reward_range)
print('\nObservation ranges:')
for ob in enumerate(env_ranges):
    print(ob[0], ob[1])

num_bins = [10, 3]
num_pos_actions = len(actions)
print('num_pos_actions', num_pos_actions)

# hyperparams:
discount = 1.0
episodes = 100

epsilon = [0.5, 1.0, episodes//2]  # Epsilon start, start decay index, stop decay index
lr = [0.5, 1.0, episodes//2]  # Learning rate start, start decay index, stop decay index

q_learning = QLearning(env,
                       num_bins,
                       num_pos_actions,
                       env_ranges,
                       discount,
                       episodes,
                       epsilon,
                       lr)

print('q-table shape', q_learning.q_table.shape)

obs = q_learning.reset_state()  # Reset the environment and get the initial
obs = [obs[i] for i in obs_to_use]
print('\nInitial observation:', obs)

action_to_maximise_q = q_learning.action_to_maximise_q(obs)  # Find optimal action
action = q_learning.decide_on_action(action_to_maximise_q)  # Decide whether to use optimal or random action
observation, reward_current, done = q_learning.perform_sim_step(action)  # env.step(action)  # Perform the first action

NUM_TO_SHOW = 5
rewards = []

while q_learning.episode < q_learning.episodes:

    reward_sum = 0

    if not q_learning.episode % (episodes // NUM_TO_SHOW):
        render = True
        print('episode, learning_rate, epsilon', q_learning.episode, q_learning.lr, q_learning.epsilon)
    else:
        render = False

    q_learning.episode += 1
    q_learning.perform_epsilon_decay()
    q_learning.perform_lr_decay()

    while not done:

        action_to_maximise_q = q_learning.action_to_maximise_q(obs)
        action = q_learning.decide_on_action(action_to_maximise_q)  # Decide whether to use optimal or random action
        current_q_value = q_learning.look_up_q_value(obs, action)  # Get current q value

        # Play a step in the simulation with this optimal value to get a future observation (doesn't actually occur)
        ob_future, reward, done = q_learning.perform_sim_step(action)
        ob_future = [ob_future[i] for i in obs_to_use]
        reward_sum += reward
        max_future_q = q_learning.get_max_q(ob_future)  # Get max future q

        # Update the table q value
        updated_q_value = q_learning.calc_new_q_value(current_q_value, reward, max_future_q)
        q_learning.update_q_value(action, obs, updated_q_value)

        # Make the new state (found using a random or optimal q value) the old state for the next iteration
        obs = ob_future

        if render:
            q_learning.env.render()

    rewards.append(reward_sum)
    obs = q_learning.reset_state()  # Reset the environment and get the initial state
    obs = [obs[i] for i in obs_to_use]
    done = False

q_learning.env.close()

plt.plot(rewards, linewidth=0.2)
plt.show()

