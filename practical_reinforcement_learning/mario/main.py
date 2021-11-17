"""Mario."""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Activation, Flatten, Conv2D, MaxPool2D
)
from tensorflow.keras.optimizers import Adam
import random
import gym
import numpy as np
from collections import deque
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY  # can only go right
from nes_py.wrappers import JoypadSpace
from PIL import Image


def random_run(display=False):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, RIGHT_ONLY)  # we want right only for actions

    total_reward = 0
    done = True

    for step in range(100):
        if display:
            env.render()
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
    state_size = (80, 88, 1)  # the screen size, after conversion
    action_space = env.action_space.n
    return state_size, action_space, env

def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((88, 80))
    image = image.convert("L")  # greyscale
    image = np.array(image)

    return image

class DQNAgent:
    def __init__(self, state_size, action_size):
        # create var for agent
        self.state_space = state_size
        self.action_space = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.8
        self.choosenAction = 0

        # exploration vs exploitation
        self.epsilon = 0.3
        self.max_epsilon = 0.3
        self.min_epsilon = 0.01
        self.decay_epsilon = 0.0001

        # Building NN for Agent
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(64, (4, 4), 4, "same", input_shape=self.state_space))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), 2, "same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), 1, "same"))
        model.add(Activation("relu"))
        model.add(Flatten())

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())
        print(model.summary())

        return model

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
        return self

    def act(self, state, onGround):
        if onGround < 83:
            if random.uniform(0, 1) < self.epsilon:
                # random action
                self.choosenAction = np.random.randint(self.action_space)
            else:
                Q_value = self.main_network.predict(state)
                self.choosenAction = np.argmax(Q_value[0])
            return self.choosenAction
        else:
            # not on ground
            print("not on ground")
            return self.choosenAction

    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_epsilon * episode)
        return self

    def train(self, batch_size):
        # train the network
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # get var from batch and find q-value
            target = self.main_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))

            self.main_network.fit(state, target, epochs=1, verbose=0)

    def store_trainsition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return self

    def load(self, name):
        self.main_network = tf.keras.models.load_model(name)
        self.target_network = tf.keras.models.load_model(name)
        return self

    def save(self, name):
        self.main_network.save(name)
        return self

    def get_pred_act(self, state):
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])

def train(display=False):
    # complicated network that require much larger training
    num_episode = 20
    num_timesteps = 4000
    batch_size = 64
    print("Starting training")
    state_size, action_size, env = random_run()
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    debug_length = 200

    stuck_buffer = deque(maxlen=debug_length)

    for i in range(num_episode):
        Return = 0
        done = False
        time_step = 0
        onGround = 79

        state = preprocess_state(env.reset())
        state = state.reshape((-1, 80, 88, 1))

        for t in range(num_timesteps):
            if display:
                env.render()
            time_step += 1

            if t > 1 and stuck_buffer.count(stuck_buffer[-1]) == debug_length:
                action = agent.act(state, onGround=79)
            action = agent.act(state, onGround)

            next_state, reward, done, info = env.step(action)
            onGround = info["y_pos"]
            stuck_buffer.append(onGround)

            if done:
                break

            next_state = preprocess_state(next_state)
            next_state = next_state.reshape((-1, 80, 88, 1))

            agent.store_trainsition(state, action, reward, next_state, done)
            state = next_state

            Return += reward
            print(f"episode = {i}\n{time_step=}\n{Return=}\n{reward=}\n{agent.epsilon}")

            if (len(agent.memory) > batch_size) and (i > 5):
                  agent.train(batch_size)

        agent.update_epsilon(i)
        agent.update_target_network()
    return agent

def visualize(model_path, display=False):
    state_size, action_size, env = random_run()
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.main_network = tf.keras.models.load_model(model_path)
    while True:
        done = False
        state = preprocess_state(env.reset())
        state = state.reshape((-1, 80, 88, 1))
        total_reward = 0

        while not done:
            if display:
                env.render()
            action = agent.get_pred_act(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            print(total_reward)

            next_state = preprocess_state(next_state)
            state = next_state
            state = state.reshape((-1, 80, 88, 1))

    env.close()

if __name__ == "__main__":
    agent = train()
    agent.save("mario")
