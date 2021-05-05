from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import numpy as np

REPLAY_MEMORY_SIZE = 50_000
"""
replay memory stores the past MEMORY_SIZE states (images) into memory
each time we are trying to fit self.model, it needs to work on an batch
and the batch is a random sampling of BATCH_SIZE elements from the replay
memory. This allows us to train the model on randomly sampled batches of 
input states and actions instead of each single sample that would cause 
the model to keep training every single step and not converge anywhere
"""

class DQNAgent:
    def __init__(self):
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
        model.add(Conv2D(256, (3, 3), input_shape=(env.OBSERVATION_SPACE_VALUES)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]