import gym
import numpy as np
from deep_q_learning import DeepQLearning
import os
import time
np.set_printoptions(precision=1, linewidth=np.inf)

env = gym.make('CartPole-v0')
actions = (0, 1)
env_ranges = list(zip(env.observation_space.low, env.observation_space.high))
num_observations = len(env_ranges)

print('Ob space:\t\t', env.observation_space)
print('Action space:\t', env.action_space)
print('Reward range:\t', env.reward_range)
print('\nObservation ranges:')
for ob in enumerate(env_ranges):
    print(ob[0], ob[1])

num_pos_actions = len(actions)
print('num_pos_actions', num_pos_actions)

root_log_dir = os.path.join(os.curdir, 'my_models')
run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
model_filepath = os.path.join(root_log_dir, run_id)

# hyperparams:
discount = 0.99
episodes = 5000

epsilon = 0.5  # Epsilon
epsilon_min = 0.001
epsilon_decay = 0.999
lr = 0.001  # Learning rate
reward_max = 0  # Keep track of max reward

agent = DeepQLearning(env=env,
                      num_observations=num_observations,
                      num_pos_actions=num_pos_actions,
                      lr=lr,
                      discount=discount)

NUM_TO_SHOW = 100
episode_rewards = []
step = 1
for episode in range(episodes):

    # Update tensorboard step every episode
    agent.tensorboard_callback.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1  # Step within the current episode

    current_state = agent.env.reset()  # Reset the environment and get the initial observation

    if not episode % NUM_TO_SHOW:
        render = True  # Never showing
        print('episode {0:}, epsilon {1:.2f}'.format(episode, epsilon))
    else:
        render = False

    episode += 1

    done = False
    while not done:

        # Either get the highest q action or a random one
        if np.random.random() > epsilon:
            qs = agent.get_qs(current_state)
            action = np.argmax(qs)
            max_qs = qs[action]
        else:
            action = np.random.randint(0, agent.num_pos_actions)

        future_state, reward, done, _ = agent.env.step(action)
        episode_reward += reward

        # Draw the screen
        if render:
            agent.env.render()

        agent.update_replay_memory((current_state, action, reward, future_state, done))
        agent.train(done, step)  # Will update predictor model too if it is time
        current_state = future_state
        step += 1

    episode_rewards.append(episode_reward)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)

    stats = {
        'reward': episode_reward,
        'max_qs': max_qs
    }
    agent.tensorboard_callback.update_stats(stats)

    if stats['reward'] > reward_max:
        reward_max = stats['reward']
        print('maximum reward achieved', stats['reward'])
        agent.model.save(model_filepath)


print(episode_rewards)
agent.env.close()

