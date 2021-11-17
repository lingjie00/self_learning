import gym
import numpy as np
import random
import time

#env = gym.make("Taxi-v3")
env = gym.make("FrozenLake-v0")
#env = gym.make("FrozenLake8x8-v0")

# episodes = 10
# 
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0
# 
#     while not done:
#         env.render()
#         state, reward, done, info = env.step(env.action_space.sample())
#         score += reward
#     print(f"{episode=}\n{score=}")
# 
#env.close()

# create Q-Table
actions = env.action_space.n
state = env.observation_space.n
q_table = np.zeros((state, actions))

# parameters for Q-learning
num_episodes = 10000  # play around
max_steps_per_episode = 1000  # play around

learning_rate = 0.1  # alpha
discount_rate = 0.99  # gamma

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.00005  # play around

rewards_all_episodes = []  # store rewards

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration vs Exploitation trade off
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # variation of the equation
        # q_table[state, action] = q_table[state, action] * (1-learning_rate) + learning_rate * (
        #     reward + discount_rate * np.max(q_table[new_state, :])
        # )
        q_table[state, action] = q_table[state, action] + \
                learning_rate * (
                    reward + discount_rate * (np.max(q_table[new_state, :] - q_table[state, action]))
                )

        state = new_state

        reward_current_episode += reward

        if done == True:
            break

    exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(reward_current_episode)

print("Training ended")
print(q_table)

# calculate and print average rewards for thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)

print("Average per thousand episodes")
for r in rewards_per_thousand_episodes:
    print("1000", " : ", str(sum(r/1000)))

# visualize agent
for episode in range(3):
    state = env.reset()
    done = False
    print(f"{episode=}")

    for step in range(max_steps_per_episode):
        #env.render()
        #time.sleep(0.4)

        action = np.argmax(q_table[state, :])

        new_state, reward, done, info = env.step(action)

        if done:
            #env.render()
            #time.sleep(0.4)
            print(f"{reward=}")
            if reward == 1:
                print("reached Goal")
            else:
                print("Failed")

            break

        state = new_state

env.close()
