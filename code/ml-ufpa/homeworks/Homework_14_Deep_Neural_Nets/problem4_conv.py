from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(4,(20,10),activation='relu',strides=(4,2), input_shape=(640,480,3)))
model.add(layers.MaxPooling2D((4,2),padding='same', strides=(2,1)))
model.add(layers.Conv2D(6,(5,3),activation='relu',padding='same'))
model.add(layers.MaxPooling2D((3,2)))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation='softmax'))
model.summary()
