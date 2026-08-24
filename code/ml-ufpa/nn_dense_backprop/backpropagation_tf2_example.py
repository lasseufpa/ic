''' I created this TF v2 version as follow:
1) #From https://github.com/keras-team/keras/issues/956 I saved as
 backpropagation_tf1_example.py because it runs on Tensorflow v1
 and is not compatible with TF v2. 
2) I used
tf_upgrade_v2 --infile backpropagation_tf1_example.py --outfile backpropagation_tf2_example.py
to create a TF v2 version.
3) According to https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta
I added the line:
tf.compat.v1.disable_eager_execution()
'''
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

# from https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta
# TF 2 does not use "eager" execution, so disable it:
tf.compat.v1.disable_eager_execution()

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputs = np.random.random((1, 8))
outputs = model.predict(inputs)
targets = np.random.random((1, 8))
rmse = sqrt(mean_squared_error(targets, outputs))
loss = losses.mean_squared_error(targets, model.output)

#  ===== Symbolic Gradient =====
gradients = k.gradients(loss, model.trainable_weights)

print("===BEFORE WALKING DOWN GRADIENT===")
print("outputs:\n", outputs)
print("targets:\n", targets)

# Begin TensorFlow
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.initialize_all_variables())

steps = 100  # steps of gradient descent
for s in range(steps):

    # ===== Numerical gradient =====
    evaluated_gradients = sess.run(gradients, feed_dict={model.input: inputs})

    # Step down the gradient for each layer
    for i in range(len(model.trainable_weights)):
        sess.run(tf.compat.v1.assign_sub(model.trainable_weights[i], evaluated_gradients[i]))

    # Every 10 steps print the RMSE
    if s % 10 == 0:
        outputs = model.predict(inputs)
        rmse = sqrt(mean_squared_error(targets, outputs))
        print("step " + str(s) + " rmse:", rmse)

final_outputs = model.predict(inputs)
final_rmse = sqrt(mean_squared_error(targets, final_outputs))

print("===AFTER STEPPING DOWN GRADIENT===")
print("outputs:\n", outputs)
print("targets:\n", targets)
