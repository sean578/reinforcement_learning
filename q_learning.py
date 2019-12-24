import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)


class QLearning:

    def __init__(self, env, ob_dict, hyper, episodes, num_poss_actions):
        self.env = env
        self.ob_dict = ob_dict
        self.hyper = hyper  # Dictionary containing hyper-parameters
        self.episodes = episodes
        self.episode = 0
        self.num_poss_actions = num_poss_actions

        self.q_table = self.create_q_table(low=0, high=1)

        ############################################
        # TIDY UP THESE VARIABLES ##################
        ############################################

        self.PARAM_MIN, self.PARAM_MAX = -5.0, 5.0
        self.BIN_SIZE = (self.PARAM_MAX - self.PARAM_MIN) / self.hyper['num_bins']

        self.epsilon = 1.0
        self.ep_decay_start = 1
        self.ep_decay_end = self.episodes // 2
        self.epsilon_decay_value = self.epsilon / (self.ep_decay_end - self.ep_decay_start)

        self.lr = 0.05
        self.lr_decay_start = 1
        self.lr_decay_end = self.episodes
        self.lr_decay_value = self.lr / (self.lr_decay_end - self.lr_decay_start)

        ############################################
        ############################################
        ############################################

    def create_q_table(self, low, high):
        # return np.random.uniform(low=low,
        #                          high=high,
        #                          size=(self.num_poss_actions, self.hyper['num_bins'], self.hyper['num_bins']))

        return np.zeros((self.num_poss_actions, self.hyper['num_bins'], self.hyper['num_bins']))

    def reset_state(self):
        return self.env.reset()

    def perform_sim_step(self, action):
        ob, reward, done, _ = self.env.step(action)
        if done:
            reward = -1
            # print('reward', reward)
        else:
            reward = 1
        return ob, reward, done

    def calc_new_q_value(self, current_q, current_reward, max_future_q):
        new_q = (1 - self.lr) * current_q + \
                self.lr * (current_reward + self.hyper['discount'] * max_future_q)
        return new_q

    def get_bin_index(self, value):
        return int((value - self.PARAM_MIN) // self.BIN_SIZE)

    def get_bin_value(self, index):
        return self.PARAM_MIN + index * self.BIN_SIZE

    def get_indicies(self, obs):
        index_angle = self.get_bin_index(obs[self.ob_dict['angle']])
        index_p_vel = self.get_bin_index(obs[self.ob_dict['p_vel']])
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

    def perform_lr_decay(self):
        if self.lr_decay_end >= self.episode >= self.lr_decay_start:
            self.lr -= self.lr_decay_value

    def decide_on_action(self, action_for_max_q):
        if np.random.random() > self.epsilon + 0.1:
            return action_for_max_q
        else:
            random_int = np.random.randint(0, 2)
            return random_int

