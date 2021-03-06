"""Car RL."""
import gym
import highway_env
import numpy as np
from stable_baselines import HER, SAC, PPO2, DQN

# roundabout
env = gym.make("roundabout-v0")
model = PPO2("MlpPolicy", env, verbose=1)

for i in range(10):
    done = False
    env.reset()
    while not done:
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(info)
env.close()

# create model
model = PPO2("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# save and load model
model.save("roundabout")
model = PPO2.load("roundabout")

# visualize model
for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        # env.render()
        action, _states = model.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(info)

# parking agent
env = gym.make("parking-v0")
# random action

# model
model = HER("MlpPolicy", env, SAC, n_sampled_goal=4, goal_selection_strategy="future", verbose=1)
model.learn(1000)
model.save("parkingagent")
model = HER.load("parkingagent", env=env)

# Merge agent
env = gym.make("merge-v0")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(100)
model2 = PPO2("MlpPolicy", env, verbose=1)
