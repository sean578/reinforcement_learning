import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)
from q_learning import QLearning

env = gym.make('LunarLander-v2')
actions = (0, 1, 2, 3)
env_ranges = list(zip(env.observation_space.low, env.observation_space.high))
num_observations = len(env_ranges)
print('Ob space:\t\t', env.observation_space)
print('Action space:\t', env.action_space)
print('Reward range:\t', env.reward_range)

bin_min, bin_max = -0.7, 0.7

for i in range(num_observations):
    env_ranges[i] = (bin_min, bin_max)

print('\nObservation ranges:')
for ob in enumerate(env_ranges):
    print(ob[0], ob[1])

num_bins = [6, 6, 6, 6, 6, 6, 3, 3]
num_pos_actions = len(actions)

# hyperparams:
discount = 1.0
episodes = 100000

epsilon = [0.5, 1.0, episodes]  # Epsilon start, start decay index, stop decay index
lr = [0.5, 1.0, episodes]  # Learning rate start, start decay index, stop decay index

q_learning = QLearning(env, num_bins, num_pos_actions, env_ranges, discount, episodes, epsilon, lr)
obs = q_learning.reset_state()  # Reset the environment and get the initial
print('\nInitial observation:', obs)

action_to_maximise_q = q_learning.action_to_maximise_q(obs)  # Find optimal action
action = q_learning.decide_on_action(action_to_maximise_q)  # Decide whether to use optimal or random action
observation, reward_current, done = q_learning.perform_sim_step(action)  # env.step(action)  # Perform the first action

NUM_EPOCHS = 1
NUM_TO_SHOW = 100
rewards = []

for epoch in range(NUM_EPOCHS):

    print('EPOCH', epoch)

    q_learning.episode = 0
    q_learning.epsilon = epsilon[0]
    q_learning.lr = lr[0]

    while q_learning.episode < q_learning.episodes:

        reward_sum = 0

        if not q_learning.episode % (episodes // NUM_TO_SHOW):
            render = True
            print('min, max q table values: {0:.1f}, {1:.1f}'.format(q_learning.q_table.min(), q_learning.q_table.max()))
            print('episode, learning_rate, epsilon: {0:.0f}, {1:.2f}, {2:.2f}'.format(q_learning.episode, q_learning.lr, q_learning.epsilon))
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
            reward_sum += reward
            ob_future = ob_future
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
        done = False

q_learning.env.close()

plt.plot(rewards, linewidth=0.2)
plt.show()

