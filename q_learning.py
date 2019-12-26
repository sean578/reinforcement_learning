import numpy as np
np.set_printoptions(precision=1, linewidth=np.inf)


class QLearning:

    def __init__(self, env, num_bins, num_pos_actions, env_ranges, discount, episodes, epsilon, lr):
        self.env = env
        self.num_bins = num_bins
        self.num_poss_actions = num_pos_actions
        self.env_ranges = env_ranges
        self.discount = discount
        self.episodes = episodes

        self.episode = 0

        self.bin_sizes = []
        for i, r in enumerate(self.env_ranges):
            self.bin_sizes.append((r[1] - r[0]) / self.num_bins[i])

        [self.epsilon,  self.ep_decay_start, self.ep_decay_end] = epsilon
        [self.lr, self.lr_decay_start, self.lr_decay_end] = lr

        self.epsilon_decay_value = self.epsilon / (self.ep_decay_end - self.ep_decay_start)
        self.lr_decay_value = self.lr / (self.lr_decay_end - self.lr_decay_start)

        self.q_table = self.create_q_table()

    def create_q_table(self):
        # print(self.num_poss_actions)
        q_table = np.zeros((self.num_poss_actions, *self.num_bins))
        return q_table

    def reset_state(self):
        return self.env.reset()

    def perform_sim_step(self, action):
        ob, reward, done, _ = self.env.step(action)
        return ob, reward, done

    def calc_new_q_value(self, current_q, current_reward, max_future_q):
        new_q = (1 - self.lr) * current_q + self.lr * (current_reward + self.discount * max_future_q)
        return new_q

    def get_bin_indicies(self, values):
        indicies = []
        for i, value in enumerate(values):
            bin_step = (self.env_ranges[i][1] - self.env_ranges[i][0]) / self.num_bins[i]
            index = (value - self.env_ranges[i][0]) / bin_step
            if index < 0:
                index = 0
            if index >= self.num_bins[i]:
                index = self.num_bins[i] - 1
            indicies.append(int(index))
        return tuple(indicies)

    def look_up_q_value(self, obs, action):
        indicies = self.get_bin_indicies(obs)
        return self.q_table[(action, ) + indicies]

    def update_q_value(self, action, obs, new_q_value):
        indicies = self.get_bin_indicies(obs)
        self.q_table[(action, ) + indicies] = new_q_value

    def action_to_maximise_q(self, obs):
        indicies = self.get_bin_indicies(obs)
        return np.argmax(self.q_table[(slice(None), ) + indicies])

    def get_max_q(self, obs):
        indicies = self.get_bin_indicies(obs)
        return np.max(self.q_table[(slice(None), ) + indicies])

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
            random_int = np.random.randint(0, self.num_poss_actions)
            return random_int

