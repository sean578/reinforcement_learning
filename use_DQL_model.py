"""
Load in a pre-trained model and use its predicted q-values to control an agent.
"""

# import tensorflow as tf
import tensorflow.keras.models as models
import os
import gym
import numpy as np

##################################################################################
# Create the environment
##################################################################################

env = gym.make('CartPole-v0')

##################################################################################
# Load the pre-trained model
##################################################################################

root_log_dir = os.path.join(os.curdir, 'my_models')
filename = 'the_model'
model_filepath = os.path.join(root_log_dir, filename)

# Load a pre-trained model
model = models.load_model(model_filepath)

##################################################################################
# Make predictions using the model and take the best actions
##################################################################################

# Reset the environment and get the first observation
state = env.reset()

total_reward = 0
done = False
render = True
while not done:

    if render:
        env.render()

    # Find the best action to take:
    qs = model.predict(np.array(state).reshape(-1, *state.shape))[0]
    action = np.argmax(qs)

    # Take the action:
    state, reward, done, _ = env.step(action)
    total_reward += reward

print('Total reward', total_reward)

