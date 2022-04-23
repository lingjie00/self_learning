from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# loading the data

dir = './projects/DeepLearning_AI/horse-or-human/'
horse_dir = dir + 'horses'
human_dir = dir + 'humans'

# import os
# import zipfile
#
#
# local_zip = dir + '/horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall(dir + '/horse-or-human')
# zip_ref.close()
#

#
# horses = os.listdir(horse_dir)
# humans = os.listdir(human_dir)
#
# print(len(horses))
# print(len(humans))
#
# # display images
#
# img = mpimg.imread(horse_dir + '/' + horses[0])
# plt.imshow(img)
# plt.show()

# building model

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu',
                              input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# building generator

train_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
    dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# training model

history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1)

# visualising the layers

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = keras.models.Model(model.input, successive_outputs)

img = load_img(dir + 'horses/horse01-5.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)  # shape (1, 150, 150, 3)
x /= 255.

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers[1:]]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size: (i+1) * size] = x
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()