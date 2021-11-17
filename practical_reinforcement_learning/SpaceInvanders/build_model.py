from tensorflow import keras
from tensorflow.keras import layers
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


########
# CNN
########
def build_model(height, width, channel, actions):
    """Build a model based on the screen size and actions."""
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=(3, height, width, channel)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model


######
# RL
######
def build_agent(model, actions):
    """Build a RL agent."""
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps", value_max=1., value_min=.1,
        value_test=.2, nb_steps=10000
    )
    memory = SequentialMemory(limit=2000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type="avg",
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn

