import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)
import math


class QLearning:

    def __init__(self, env, ob_dict, hyper, episodes, num_poss_actions):
        self.env = env
        self.ob_dict = ob_dict
        self.hyper = hyper  # Dictionary containing hyper-parameters
        self.episodes = episodes
        self.episode = 0
        self.num_poss_actions = num_poss_actions



        ############################################
        # TIDY UP THESE VARIABLES ##################
        ############################################

        self.PARAM_RANGE_VEL = list(zip(env.observation_space.low, env.observation_space.high))[2]

        # self.PARAM_RANGE_VEL = [-2.5, 0.5]
        self.PARAM_RANGE_ANGLE = [-math.radians(50), math.radians(50)]
        self.BIN_SIZE_VEL = (self.PARAM_RANGE_VEL[1] - self.PARAM_RANGE_VEL[0]) / self.hyper['num_bins_vel']
        self.BIN_SIZE_ANGLE = (self.PARAM_RANGE_ANGLE[1] - self.PARAM_RANGE_ANGLE[0]) / self.hyper['num_bins_angle']

        self.epsilon = 0.5
        self.ep_decay_start = 1
        self.ep_decay_end = self.episodes // 2
        self.epsilon_decay_value = self.epsilon / (self.ep_decay_end - self.ep_decay_start)

        self.lr = 0.5
        self.lr_decay_start = 1
        self.lr_decay_end = self.episodes // 2
        self.lr_decay_value = self.lr / (self.lr_decay_end - self.lr_decay_start)

        print(self.PARAM_RANGE_VEL, self.PARAM_RANGE_ANGLE)

        self.q_table = self.create_q_table(low=0, high=1)

        ############################################
        ############################################
        ############################################

    def create_q_table(self, low, high):
        # TODO: Choose better initialisation range?
        # return np.random.uniform(low=10*self.PARAM_RANGE_ANGLE[0],
        #                          high=10*self.PARAM_RANGE_ANGLE[1],
        #                          size=(self.num_poss_actions, self.hyper['num_bins_angle'], self.hyper['num_bins_vel']))

        return np.zeros((self.num_poss_actions, self.hyper['num_bins_angle'], self.hyper['num_bins_vel']))

    def reset_state(self):
        return self.env.reset()

    def perform_sim_step(self, action):
        ob, reward, done, _ = self.env.step(action)
        if done:
            reward = -10
            # print('reward', reward)
        else:
            reward = 1
        return ob, reward, done

    def calc_new_q_value(self, current_q, current_reward, max_future_q):
        new_q = (1 - self.lr) * current_q + \
                self.lr * (current_reward + self.hyper['discount'] * max_future_q)
        return new_q

    def get_bin_index(self, value, angle):
        if angle:
            index = int((value - self.PARAM_RANGE_ANGLE[0]) // self.BIN_SIZE_ANGLE)
        else:
            index = int((value - self.PARAM_RANGE_VEL[0]) // self.BIN_SIZE_VEL)

        if index < 0:
            index = 0
        if angle:
            if index > self.hyper['num_bins_angle'] - 1:
                index = self.hyper['num_bins_angle'] - 1
        else:
            if index > self.hyper['num_bins_vel'] - 1:
                index = self.hyper['num_bins_vel'] - 1
        return index

    # def get_bin_value(self, index, angle):
    #     if angle:
    #         return self.PARAM_RANGE_ANGLE[0] + index * self.BIN_SIZE_ANGLE

    def get_indicies(self, obs):
        index_angle = self.get_bin_index(obs[self.ob_dict['angle']], angle=True)
        index_p_vel = self.get_bin_index(obs[self.ob_dict['p_vel']], angle=False)
        return index_angle, index_p_vel

    def look_up_q_value(self, obs, action):
        index_angle, index_p_vel = self.get_indicies(obs)
        return self.q_table[action, index_angle, index_p_vel]

    def update_q_value(self, action, obs, new_q_value):
        index_angle, index_p_vel = self.get_indicies(obs)
        self.q_table[action, index_angle, index_p_vel] = new_q_value

    def action_to_maximise_q(self, obs):
        index_angle, index_p_vel = self.get_indicies(obs)
        return np.argmax(self.q_table[:, index_angle, index_p_vel])

    def get_max_q(self, obs):
        index_angle, index_p_vel = self.get_indicies(obs)
        return np.max(self.q_table[:, index_angle, index_p_vel])

    def perform_epsilon_decay(self):
        if self.ep_decay_end >= self.episode >= self.ep_decay_start:
            self.epsilon -= self.epsilon_decay_value
        if self.epsilon < 0:
            self.epsilon = 0

    def perform_lr_decay(self):
        if self.lr_decay_end >= self.episode >= self.lr_decay_start:
            self.lr -= self.lr_decay_value
        if self.lr < 0:
            self.lr = 0

    def decide_on_action(self, action_for_max_q):
        if np.random.random() > self.epsilon:
            return action_for_max_q
        else:
            random_int = np.random.randint(0, 2)
            return random_int

