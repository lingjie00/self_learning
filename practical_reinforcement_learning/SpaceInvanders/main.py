import gym

import numpy as np
from tensorflow.keras.optimizers import Adam
from build_model import build_model, build_agent

##############
# env
##############
# create env
env = gym.make("SpaceInvaders-v0")

# create episodes
# iteratively train agent
episodes = 10

# for episode in range(episodes):
#     # reset state
#     state = env.reset()
#     done = False
#     score = 0
# 
#     while not done:
#         #env.render()  # display env
#         state, reward, done, info = env.step(env.action_space.sample())
#         score += reward
#     print(f"{episode}\n{score=}")


########
# CNN
########
height, width, channel = env.observation_space.shape  # env space
actions = env.action_space.n  # num of actions
model = build_model(height, width, channel, actions)


######
# RL
######
dqn = build_agent(model, actions)
# compile
dqn.compile(Adam(lr=0.001))
# train
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

# save
dqn.save_weights("models_modified/dqn.h5f")

# load
# dqn.load_weigths("models/dqn.h5f")

# test
scores = dqn.test(env, nb_episodes=5, visualize=False)
print(np.mean(scores.history["episode_reward"]))


# exit env
env.close()
