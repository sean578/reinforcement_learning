import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)
from q_learning import QLearning
from time import sleep

env = gym.make('LunarLander-v2')
actions = (0, 1, 2, 3)
env_ranges = list(zip(env.observation_space.low, env.observation_space.high))
num_observations = len(env_ranges)  # Last two obs don't do anything
print('Ob space:\t\t', env.observation_space)
print('Action space:\t', env.action_space)
print('Reward range:\t', env.reward_range)

bin_min, bin_max = -1, 1

for i in range(num_observations):
    env_ranges[i] = (bin_min, bin_max)

print('\nObservation ranges:')
for ob in enumerate(env_ranges):
    print(ob[0], ob[1])

# nb = 20
# num_bins = [nb] * num_observations
num_bins = [2, 2, 10, 10, 10, 10, 2, 2]
num_pos_actions = len(actions)

# hyperparams:
discount = 0.98
episodes = 1000

epsilon = [1.0, 1.0, episodes]  # Epsilon start, start decay index, stop decay index
lr = [0.5, 1.0, episodes]  # Learning rate start, start decay index, stop decay index

q_learning = QLearning(env, num_bins, num_pos_actions, env_ranges, discount, episodes, epsilon, lr)
obs = q_learning.reset_state()  # Reset the environment and get the initial
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
    # print('episode {0:}, epsilon {1:.2f}, lr {2:.2f}'.format(q_learning.episode, q_learning.epsilon, q_learning.lr))
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
    if reward > -100:
        print('final_reward', reward)
    obs = q_learning.reset_state()  # Reset the environment and get the initial state
    done = False

q_learning.env.close()

plt.plot(rewards, linewidth=0.2)
plt.show()

