from collections import deque
import time
from modified_tensorboard import ModifiedTensorBoard  # A version of tensorboard that works well with DQN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

REPLAY_MEMORY_SIZE = 2000
REPLAY_MEMORY_SIZE_TO_START_TRAINING = 50
MINIBATCH_SIZE = 32

UPDATE_TARGET_EVERY = 10  # Number of episodes


class DeepQLearning:
    def __init__(self, env, num_observations, num_pos_actions, lr, discount):

        self.env = env
        self.num_observations = num_observations
        self.num_pos_actions = num_pos_actions
        self.lr = lr
        self.discount = discount

        # Create a model used for training
        self.model = self.create_model()

        # Create a second model used for predictions - not updated as frequently
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0  # Keep track of when to update prediction model

        # Create a double-ended queue to hold past observations to be used for training
        # Each element contains data of two point in time (S1 --> S2 via action A)
        # Includes the instant reward & wheter the episode is done
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Store results during training using a tensorboard
        self.tensorboard_callback = ModifiedTensorBoard(log_dir='logs/test4')

    def create_model(self):
        """ The model used to estimate Q-vales from the observable state """

        model = Sequential()
        model.add(Dense(32, input_shape=(self.num_observations, )))
        model.add(Activation('elu'))
        model.add(Dense(32))
        model.add(Activation('elu'))
        model.add(Dense(self.num_pos_actions))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        """ (observation_space, action, reward, new observation space, done)"""

        self.replay_memory.append(transition)

    def get_qs(self, state):
        """ Get the predicted q-values for the current state for each possible action """

        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, done, step):
        """ Perfrom a training step on the main network """

        # Only start training once we have enough samples in the replay buffer
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE_TO_START_TRAINING:
            return

        # Get a random sample of previous experiences from the replay buffer to train on
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get the current states (S1 in S1 --> S2 via A)
        current_states = np.array([transition[0] for transition in minibatch])

        # Predict the Q-values for each action from state S1 using the prediction model
        current_qs_list = self.model.predict(current_states)

        # Get the observed future states (S2 in S1 --> S2 via A)
        new_current_states = np.array([transition[3] for transition in minibatch])

        # Predict the Q-values for each action from state S2 using the target model
        future_qs_list = self.target_model.predict(new_current_states)

        ################################################################################
        # Update (refit) the model
        ################################################################################

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not done then use discounted plus current rewards, else just use current
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = -10

            # Update the q value for the current state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to the data to be used for training
            X.append(current_state)
            y.append(current_qs)

        # Fit all the above samples
        self.model.fit(np.array(X, dtype=np.float16),
                       np.array(y, dtype=np.int16),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       callbacks=None)

        # If the episode has ended, update the count of when to update the target model
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

