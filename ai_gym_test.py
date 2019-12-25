import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)
from q_learning import QLearning
from time import sleep

env = gym.make('CartPole-v0')

ob_dict = {
    'pos': 0,
    'c_vel': 1,
    'angle': 2,
    'p_vel': 3
}

hyper = {
    'discount': 1.0,
    'num_bins_angle': 10,
    'num_bins_vel': 3
}

num_poss_actions = 2
episodes = 200

for _ in range(1):

    q_learning = QLearning(env, ob_dict, hyper, episodes, num_poss_actions)
    obs = q_learning.reset_state()  # Reset the environment and get the initial state

    action_to_maximise_q = q_learning.action_to_maximise_q(obs)  # Find optimal action
    action = q_learning.decide_on_action(action_to_maximise_q)  # Decide whether to use optimal or random action
    observation, reward_current, done = q_learning.perform_sim_step(action)  # env.step(action)  # Perform the first action

    num_steps_alive = 0
    num_alive = []
    while q_learning.episode < q_learning.episodes:

        if not q_learning.episode % (episodes // 10):
            render = True
        else:
            render = False

        q_learning.episode += 1
        # print('episode {0:}, epsilon {1:.2f}, lr {2:.2f}'.format(q_learning.episode, q_learning.epsilon, q_learning.lr))
        q_learning.perform_epsilon_decay()
        q_learning.perform_lr_decay()
        num_steps_alive = 0

        while not done:

            action_to_maximise_q = q_learning.action_to_maximise_q(obs)
            action = q_learning.decide_on_action(action_to_maximise_q)  # Decide whether to use optimal or random action

            current_q_value = q_learning.look_up_q_value(obs, action)  # Get current q value

            # Play a step in the simulation with this optimal value to get a future observation (doesn't actually occur)
            ob_future, reward, done = q_learning.perform_sim_step(action)

            max_future_q = q_learning.get_max_q(ob_future)  # Get max future q

            # Update the table q value
            updated_q_value = q_learning.calc_new_q_value(current_q_value, reward, max_future_q)
            q_learning.update_q_value(action, obs, updated_q_value)

            # Make the new state (found using a random or optimal q value) the old state for the next iteration
            obs = ob_future
            # print(obs)

            num_steps_alive += 1

            if render:
                pass
                # print('ep = {0:.3f}, lr = {1:.3f}'.format(q_learning.epsilon, q_learning.lr))
                q_learning.env.render()
                np.save('./data/{}'.format(q_learning.episode), q_learning.q_table)
                sleep(0.01)

        num_alive.append(num_steps_alive)

        obs = q_learning.reset_state()  # Reset the environment and get the initial state
        done = False

    q_learning.env.close()
    # print(num_alive)

    plt.plot(num_alive, linewidth=0.5)
    # plt.savefig('./data/learning.png')
    plt.show()


