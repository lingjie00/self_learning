import random
import numpy as np
import flappy_bird_gym
from collections import deque  # storing a Q data structure
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import RMSprop


# Neural Network for Agen
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape, activation="relu",
                   kernel_initializer="he_uniform"))
    model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(output_shape, activation="linear", kernel_initializer="he_uniform"))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model

# Brain of Agent || BluePrint of Agent
class DQNAgent:
    def __init__(self):
        # Environment
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 5000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        # Hyper-parameter
        self.gamma = 0.95 # discount rate
        self.epsilon = 1 # take random action
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_number = 64 # for NN, start with 16, 32, 128, 256

        self.train_start = 1000
        self.jump_prob = 0.01
        self.model = NeuralNetwork(input_shape=(self.state_space, ), output_shape=self.action_space)

    def act(self, state):
        if np.random.random() > self.epsilon:
            # using model
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    def learn(self):
        """Training."""
        if len(self.memory) < self.train_start:
            # less than needed data
            return None

        # create minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))
        # varaibles to store minibatch info
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))
        action, reward, done = [], [], []

        # store data in variables
        for i in range(self.batch_number):
            minibatch_i = minibatch[i]
            state[i] = minibatch_i["state"]
            action.append(minibatch_i["action"])
            reward.append(minibatch_i["reward"])
            next_state[i] = minibatch_i["next_state"]
            done.append(minibatch_i["done"])

        # predict y label
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_number, verbose=0)

    def train(self):
        # n episode iterations for training
        for i in range(self.episodes):
            # Environment variables for training
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            print(state) if i ==1 else None
            done = False
            score = 0
            # decay learning rate
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                # not cleared with level
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                # reshape next state
                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1

                if done:
                    reward -= 100

                self.memory.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                })
                state = next_state

                if done:
                    print(f"Episode: {i}\n{score=}\n{self.epsilon}")
                    # save model
                    if score >= 1000:
                        self.model.save("flappy_bird/flappybrain")
                        return

                self.learn()

    def perform(self, display=False):
        """Visualize model."""
        self.model = load_model("flappybrain")
        while True:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                if display:
                    self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                print(f"{score=}")

                if done:
                    print("DEAD")


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
    #agent.perform()

