import gym
env = gym.make('CartPole-v1')
initial_observation = env.reset()
# 0: left, 1: right
# print('Action space', env.action_space)
# Position, velocity, angle, pole velocity:
# print('Observation space', env.observation_space)
# print('Observation space: max values', env.observation_space.high)
# print('Observation space: min values', env.observation_space.low)
# print('Initial observation:', initial_observation)

ob_dict = {
    'pos': 0,
    'c_vel': 1,
    'angle': 2,
    'p_vel': 3
}

done = False
force = 1
angle = 0
pos = 0
while abs(angle) < 1.0 and abs(pos) < 2.5:
    env.render()
    observation, reward, done, info = env.step(force)  # take a random action
    angle = observation[ob_dict['angle']]
    pos = observation[ob_dict['pos']]
    if angle < 0.05:
        force = 0
    else:
        force = 1
env.close()

