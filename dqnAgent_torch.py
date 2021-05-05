import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque, OrderedDict
import numpy as np
import random
import os

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

random.seed(1)
np.random.seed(1)
torch.random.manual_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class NeuralNet(nn.Module):
    def __init__(self, input_size=(3, 10, 10), output_size=4):
        super(NeuralNet, self).__init__()
        shape = np.array([input_size[1], input_size[2]]).astype(np.int32)
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=256, kernel_size=3)
        shape = shape - 2
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        shape = shape//2
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        shape = shape - 2
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        shape = shape//2
        self.drop2 = nn.Dropout(0.2)

        self.dens1 = nn.Linear(in_features=256*shape[0]*shape[1], out_features=64)
        self.dens2 = nn.Linear(64, output_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.dens1(x)

        x = self.dens2(x)

        return x


class DQNAgent:
    def __init__(self, env):
        self.env = env
        if (torch.cuda.is_available()):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(self.device)

        self.model = NeuralNet(input_size=(3, 10, 10), output_size=9)
        self.target_model = NeuralNet(input_size=(3, 10, 10), output_size=9)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state = np.rollaxis(state, 2, 0)        # shape of (H, W, C) to (C, H, W)
        state = np.array(state).reshape(-1, *state.shape)  # (C, H, W) to (1, C, H, W)
        state = state.astype(np.float32)
        state = torch.Tensor(state/255)
        prediction = self.model(state.to(self.device)).cpu().detach().numpy()
        return prediction[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_states = np.swapaxes(current_states, 1, 3)
        current_states = np.swapaxes(current_states, 2, 3)
        current_states = torch.Tensor(current_states.astype(np.float32))
        current_qs_list = self.model(current_states.to(self.device))
        current_qs_list = current_qs_list.cpu().detach().numpy()

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        new_current_states = np.swapaxes(new_current_states, 1, 3)
        new_current_states = np.swapaxes(new_current_states, 2, 3)
        new_current_states = torch.Tensor(new_current_states.astype(np.float32))
        future_qs_list = self.target_model(new_current_states.to(self.device))
        future_qs_list = future_qs_list.cpu().detach().numpy()

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(np.rollaxis(current_state, 2, 0))
            y.append(current_qs)

        X = torch.Tensor(np.array(X).astype(np.float32))
        y = torch.Tensor(np.array(y))

        if terminal_state is not None:
            y_pred = self.model(X.to(self.device)).cpu()
            loss = self.loss_function(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


# if __name__ == '__main__':
#     batch_size, C, H, W = 32, 3, 20, 40
#     model = NeuralNet(input_size=(C, H, W), output_size=8)
#
#     x = torch.randn(batch_size, C, H, W)
#
#     output = model(x)
#
#     print(output.shape)

