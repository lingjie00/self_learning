from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)  # single tensor to contain everything
train_images /= 255.
test_images = test_images.reshape(10000, 28, 28, 1)
test_images /= 255.

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(train_images, train_labels, epochs=5)
print(model.evaluate(test_images, test_labels))

# investigate the layers

layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

fig, axes = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IMAGE = 26
CONVOLUTION_NUMBER = 1
for x in range(0, 4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axes[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axes[0, x].grid(False)
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axes[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axes[1, x].grid(False)
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axes[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axes[2, x].grid(False)
plt.show()