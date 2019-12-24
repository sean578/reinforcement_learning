import numpy as np
import matplotlib.pyplot as plt
import gym
np.set_printoptions(precision=1, linewidth=np.inf)


# TODO: Include angular velocity (p_vel)


env = gym.make('CartPole-v1')
env.reset()

ob_dict = {
    'pos': 0,
    'c_vel': 1,
    'angle': 2,
    'p_vel': 3
}

ANGLE_MIN, ANGLE_MAX = -5.0, 5.0
# POLE_VEL_MIN, POLE_VEL_MAX = -2.0, 3.0
NUM_BINS = 1000
BIN_SIZE = (ANGLE_MAX - ANGLE_MIN) / NUM_BINS
learning_rate = 0.2
discount = 0.99
EPISODES = 100000

epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def update_q_value(learning_rate, current_q, reward, discount, max_future_q):
    return (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)


def get_bin_index(value, min_value, bin_size):
    return int((value - min_value) // bin_size)


def get_bin_value(index, min_value, bin_size):
    return min_value + index * bin_size


# Create & initialise a q value table # TODO: Extend this by num bins of angular velocity
q_table = np.random.uniform(low=-0.1, high=0.1, size=(2, NUM_BINS))  # actions, states (angles)

print('q table initial\n', q_table)

# Get the initial state
action = 1
ob_old, reward_old, done, _ = env.step(action)
print('rewared_old', reward_old)

num_steps_alive = 0
num_random = 0
num_alive = []
for episode in range(EPISODES):
    # print('episiode', episode)

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    num_steps_alive = 0
    num_random = 0

    while not done:

        # Find the action providing the maximum q value # TODO: Use angle and angular velocity (get 2 indicies)
        ob_old_index = get_bin_index(ob_old[ob_dict['angle']], ANGLE_MIN, BIN_SIZE)

        if np.random.random() > epsilon:
            action_for_max_q = np.argmax(q_table[:, ob_old_index])  # TODO: 2 indicies in q_table lookup
        else:
            action_for_max_q = np.random.randint(0, 1)
            num_random += 1

        # Play a step in the simulation with this optimal value to get a new observation
        ob_new, reward_new, done, _ = env.step(action_for_max_q)
        if done:
            num_alive.append(num_steps_alive)
            print(num_steps_alive)
            reward_new = 0
            # break
        else:
            num_steps_alive += 1
            reward_new = 1

        ob_new_index = get_bin_index(ob_new[ob_dict['angle']], ANGLE_MIN, BIN_SIZE)  # TODO: 2 incicies

        # Get max future q
        max_future_q = np.max(q_table[:, ob_new_index])

        # Update the table q value
        new_q_value = update_q_value(learning_rate,
                                     q_table[action_for_max_q, ob_old_index],
                                     reward_new,
                                     discount,
                                     max_future_q)
        q_table[action, ob_old_index] = new_q_value
        # print('q table after {} steps\n'.format(i), q_table)

        ob_old = ob_new
        reward_old = reward_new
        action = action_for_max_q

    env.reset()
    ob_old, reward_old, done, _ = env.step(1)

    if not episode % 1000:
        np.save('./data/{}'.format(episode), q_table)

print(q_table)
env.close()
print(num_alive)

plt.plot(num_alive)
plt.show()