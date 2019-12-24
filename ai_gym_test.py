import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)

env = gym.make('CartPole-v1')
initial_observation = env.reset()

ob_dict = {
    'pos': 0,
    'c_vel': 1,
    'angle': 2,
    'p_vel': 3
}

PARAM_MIN, PARAM_MAX = -5.0, 5.0
NUM_BINS = 100
BIN_SIZE = (PARAM_MAX - PARAM_MIN) / NUM_BINS
learning_rate = 0.7
discount = 0.9
EPISODES = 1000

epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def update_q_value(learning_rate, current_q, reward, discount, max_future_q):
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
    print('q, update:', current_q, new_q)
    return new_q


def get_bin_index(value, min_value, bin_size):
    return int((value - min_value) // bin_size)


def get_bin_value(index, min_value, bin_size):
    return min_value + index * bin_size


# Create & initialise a q value table
q_table = np.random.uniform(low=0, high=10, size=(2, NUM_BINS, NUM_BINS))  # actions, pole angle, pole velocity
print('q table initial\n', q_table)

# Get the initial state
action = 1
ob_current, reward_current, done, _ = env.step(action)
print('rewared_current', reward_current)

num_steps_alive = 0
num_alive = []
for episode in range(EPISODES):
    # print('episiode', episode)

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    num_steps_alive = 0
    num_random = 0

    while not done:

        # Find the action providing the maximum q value
        # First get the q-table indicies for the current observation
        ob_current_index_angle = get_bin_index(ob_current[ob_dict['angle']], PARAM_MIN, BIN_SIZE)
        ob_current_index_p_vel = get_bin_index(ob_current[ob_dict['p_vel']], PARAM_MIN, BIN_SIZE)

        # Look up in the q-table the action corresponding to the highest q-value given the current observation
        # Instead of this, choose a random action with a probability 1-epsilon to aid exploration
        if np.random.random() > epsilon:
            action_for_max_q = np.argmax(q_table[:, ob_current_index_angle, ob_current_index_p_vel])
            print('action_for_max_q', action_for_max_q)
        else:
            action_for_max_q = np.random.randint(0, 1)

        # Play a step in the simulation with this optimal value to get a future observation
        ob_future, reward_future, done, _ = env.step(action_for_max_q)
        if done:
            num_alive.append(num_steps_alive)
            # print(num_steps_alive)
            reward_future = -10
            # break
        else:
            num_steps_alive += 1
            reward_future = 1

        ob_future_index_angle = get_bin_index(ob_future[ob_dict['angle']], PARAM_MIN, BIN_SIZE)
        ob_future_index_p_vel = get_bin_index(ob_future[ob_dict['p_vel']], PARAM_MIN, BIN_SIZE)

        # Get max future q
        max_future_q = np.max(q_table[:, ob_future_index_angle, ob_future_index_p_vel])

        # Update the table q value
        future_q_value = update_q_value(learning_rate,
                                     q_table[action_for_max_q, ob_current_index_angle, ob_current_index_p_vel],
                                     reward_future,
                                     discount,
                                     max_future_q)
        q_table[action, ob_current_index_angle, ob_current_index_p_vel] = future_q_value
        # print('q table after {} steps\n'.format(i), q_table)

        ob_current = ob_future
        reward_current = reward_future
        action = action_for_max_q

    env.reset()
    ob_current, reward_current, done, _ = env.step(1)

    if not episode % 100000:
        np.save('./data/{}'.format(episode), q_table)

print(q_table)
env.close()
print(num_alive)

plt.plot(num_alive)
plt.show()

