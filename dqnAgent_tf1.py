from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import tensorflow as tf
import os

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
"""
replay memory stores the past MEMORY_SIZE states (images) into memory
each time we are trying to fit self.model, it needs to work on an batch
and the batch is a random sampling of BATCH_SIZE elements from the replay
memory. This allows us to train the model on randomly sampled batches of 
input states and actions instead of each single sample that would cause 
the model to keep training every single step and not converge anywhere
"""


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class DQNAgent:
    def __init__(self, env):
        self.env = env
        # main model - trains on every step
        self.model = self.create_model()
        """
        self.model will likely be updated after finishing every episode
        each episode can have many steps (states) and those values are used
        for forward and backward propagation every episode/ a specified number
        of steps per episode
        """
        # target model - predicts on every step
        self.target_model = self.create_model()
        """
        target_model is used to actually predict action values during exploration
        or exploitation, and not updated as frequently as the main model. The reason
        is that otherwise the model would be predicting stuff all over the place
        and the randomness would be too much. Instead we set the target model to copy
        over the weights of the main training model after some number of episodes to 
        maintain constancy
        """
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0



    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=self.env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state0 = np.array(state).reshape(-1, *state.shape)
        prediction = self.model.predict(state0/255)
        return prediction[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        """
        X is the input to model, currently our input is images from the game
        y is the model target, namely the q values predicted by the NN, and we
        will take the argmax of the output to predict the best possible action
        for our current state
        """

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        if terminal_state is not None:
            self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        """
        Trying to determine if it's time to update our target counter yet
        If yes, then set target_model weights to the current model weights
        """
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0








