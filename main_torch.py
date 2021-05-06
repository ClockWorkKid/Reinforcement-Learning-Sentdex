from dqnAgent_torch import DQNAgent
from blob import BlobEnv
from tqdm import tqdm
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -100  # For model save
MEMORY_FRACTION = 0.20

checkpoint = False
model_file = "models/2x256____-1.00max__-31.82avg__-67.00min__1620213331.model"
# Environment settings
EPISODES = 30000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

agent = DQNAgent(BlobEnv())

ep_rewards = []
avg_rewards = []
max_rewards = []
min_rewards = []

if checkpoint == True:
    agent.model.load_state_dict(torch.load(model_file))

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    episode_reward = 0
    step = 1
    current_state = agent.env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            qs = agent.get_qs(current_state)
            action = np.argmax(qs)
        else:
            action = np.random.randint(0, agent.env.ACTION_SPACE_SIZE)

        new_state, reward, done = agent.env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            agent.env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_rewards.append(average_reward)
            max_rewards.append(max_reward)
            min_rewards.append(min_reward)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= max(min_rewards):
                torch.save(agent.model.state_dict(),
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pt')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


print("Finished training model")
plt.figure()
plt.subplot(211)
plt.plot(max_rewards, label="max")
plt.plot(avg_rewards, label="avg")
plt.plot(min_rewards, label="min")
plt.legend(loc=4)
plt.subplot(212)
plt.plot(ep_rewards)
plt.show()