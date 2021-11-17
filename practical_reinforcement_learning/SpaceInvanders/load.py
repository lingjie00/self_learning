"""Loads a RL model."""


import gym

from tensorflow.keras.optimizers import Adam
from build_model import build_model, build_agent

import numpy as np
##############
# env
##############
# create env
env = gym.make("SpaceInvaders-v0")

########
# CNN
########
height, width, channel = env.observation_space.shape  # env space
actions = env.action_space.n  # num of actions
model = build_model(height, width, channel, actions)


dqn = build_agent(model, actions)
# compile
dqn.compile(Adam(lr=0.01))

dqn.load_weights("models_modified/dqn.h5f")

# test
scores = dqn.test(env, nb_episodes=5, visualize=False)
print(np.mean(scores.history["episode_reward"]))
