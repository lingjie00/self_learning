from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape, test_images.shape)
print(train_images[0].shape)  # 28 by 28 pixels
np.unique(train_labels, return_index=True)  # 10 labels

plt.imshow(train_images[1])  # plot a sample image
plt.show()

train_images = train_images / 255.
test_images = test_images / 255.

model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(train_images, train_labels, epochs=30)

model.evaluate(test_images, test_labels)

# model evaluation

classifications = model.predict(test_images)
print(classifications[0])  # view the first prediction

wrong_predictions = (np.argmax(classifications, axis=1) != test_labels)
wrong_images = test_images[wrong_predictions]
wrong_labels = classifications[wrong_predictions]
right_labels = test_labels[wrong_predictions]

plt.imshow(wrong_images[0])
plt.show()
print(wrong_labels[0], np.argmax(wrong_labels[0]), right_labels[0])
# well our model predict the img as a Bag instead of Sneaker
# does it look like a bag???

# second model

model2 = keras.Sequential()
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(units=1024, activation=tf.nn.relu))
model2.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))

model2.compile(optimizer=tf.optimizers.Adam(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
               )

model2.fit(train_images, train_labels, epochs=5)

model2.evaluate(test_images, test_labels)