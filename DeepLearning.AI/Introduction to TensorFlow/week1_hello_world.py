from tensorflow import keras
import tensorflow as tf
import numpy as np

model = keras.Sequential([
        keras.layers.Dense(
            units=1, input_shape=[1]
        )
])

model.compile(
    optimizer='sgd',  # stochastic gradient descent
    loss='mean_squared_error'
)

xs = np.array(
    [-1., 0., 1., 2., 3., 4.]
)

ys = np.array(
    [-3., -1., 1., 3., 5., 7.]
)

model.fit(xs, ys, epochs=500)

print(model.predict(
    [10.]
))