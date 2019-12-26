import numpy as np
import gym
import matplotlib.pyplot as plt

# env = gym.make('CartPole-v1')
# env.reset()
#
# ob_dict = {
#     'pos': 0,
#     'c_vel': 1,
#     'angle': 2,
#     'p_vel': 3
# }
#
# ANGLE_MIN, ANGLE_MAX = -5.0, 5.0
# POLE_VEL_MIN, POLE_VEL_MAX = -5.0, 5.0
# NUM_BINS = 10
# BIN_SIZE = (ANGLE_MAX - ANGLE_MIN) / NUM_BINS
#
#
# def get_bin_index(value, min_value, bin_size):
#     return int((value - min_value) // bin_size)
#
#
# def get_bin_value(index, min_value, bin_size):
#     return min_value + index * bin_size

q_tables = []
# action, angle
for i in range(1, 182, 20):
    q_tables.append(np.load('./data/{}.npy'.format(i)))

min_view, max_view = 0, q_tables[-1].max()

num_to_plot = len(q_tables)
fig, ax = plt.subplots(2, num_to_plot)
for i in range(num_to_plot):
    ax[0, i].imshow(q_tables[i][0, :, :], vmin=min_view, vmax=max_view, origin='lower')
    ax[1, i].imshow(q_tables[i][1, :, :], vmin=min_view, vmax=max_view, origin='lower')
plt.show()

plt.plot(q_tables[-1][0, 4, :])
plt.show()


# action = 1
# num_steps_alive = 0
# while True:
#     env.render()
#     input('continue...')
#
#     # Play a step in the simulation with this optimal value to get a new observation
#     ob_new, reward_new, done, _ = env.step(action)
#     if done:
#         # input('finish...')
#         break
#     else:
#         num_steps_alive += 1
#
#     # Get the index in the q-table for the observed angle
#     angle_bin_index = get_bin_index(ob_new[ob_dict['angle']], ANGLE_MIN, BIN_SIZE)
#     p_vel_bin_index = get_bin_index(ob_new[ob_dict['p_vel']], ANGLE_MIN, BIN_SIZE)
#     # Get the action in the q-table with the highest q-value
#     action = np.argmax(q_table[:, angle_bin_index, p_vel_bin_index])
#
# env.close()
# print(num_steps_alive)
