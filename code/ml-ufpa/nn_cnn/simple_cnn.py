import keras
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(5, (8, 4), activation='relu',
                        strides=(4, 2), padding='same', input_shape=(200, 200, 150)))
model.add(layers.MaxPooling2D((2, 2), padding='valid', strides=(2, 4)))
model.add(layers.Conv2D(20, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((8, 8), padding='valid', strides=(2, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='linear'))
model.summary()  # display the architecture
