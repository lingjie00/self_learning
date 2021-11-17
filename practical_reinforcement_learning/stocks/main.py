"""Stock environment."""
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features

url = """
https://www.marketwatch.com/investing/index/spx/downloaddatapartial?
startdate=06/02/1975%2000:00:00&enddate=07/09/2021%2000:00:00
&daterange=d30&frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false
""".replace("\n", "")

df = pd.read_csv(url)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by=["Date"], inplace=True)
df.set_index("Date", inplace=True)
for col in ["Open", "High", "Close", "Low"]:
    df[col] = df[col].apply(lambda x: float(x.replace(",", "")))

env = gym.make("stocks-v0", df=df, frame_bound=(5, 200), window_size=5)
env.signal_features  # extracted featuers over time, scale format of dataframe
env.action_space  # buy and sell
state = env.reset()

while True:
    action = env.action_space.sample()  # random action
    next_state, reward, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.savefig("stocks/plot1.jpg")
plt.show()

env_training = lambda: gym.make("stocks-v0", df=df, frame_bound=(5, 200), window_size=5)
env = DummyVecEnv([env_training])

# create model
# Actor-Critic: child is acting, parent is criticing the child
model = A2C("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # training

env = gym.make("stocks-v0", df=df, frame_bound=(200, 253), window_size=5)
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]  # reformat
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15, 6))
plt.savefig("stocks/plot2.jpg")
plt.show()

# load data
data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2021-01-01")
# add in technical analytis (89 of them)
df2 = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")

def my_processed_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Low"].to_numpy()[start:end]
    signal_features = env.df.loc[:, ["Close", "Volume", "momentum_rsi", "volume_obv", "trend_macd_diff"]].to_numpy()
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = my_processed_data

env2 = MyCustomEnv(df=df2, window_size=5, frame_bound=(5, 700))
env.signal_features

training_env = lambda: env2
env = DummyVecEnv([training_env])

model = A2C("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

env = gym.make("stocks-vo", df=df, frame_bound=(200, 253), window_size=5)
env = MyCustomEnv(df=df2, window_size=5, frame_bound=(700, 1000))
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]  # reformat
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print(info)
        break

plt.figure(figsize=(15, 6))
plt.savefig("stocks/plot3.jpg")
plt.show()
