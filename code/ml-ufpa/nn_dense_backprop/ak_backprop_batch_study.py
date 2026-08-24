'''
Code to study MLP training with backpropagation and the influence of 
mini batch sizes, by tracking the training steps.
The neural network is a fully-connected multilayer perceptron with 1 hidden layer
and 2 inputs, 2 neurons in the hidden layer and 2 output neurons.
Hence, the weigth matrices (also called kernels by Keras) have dimensions:
W1 for layer 1 has dimension 2 x 2 and W2 of layer 2 has dimension 2 x 2.
Recall that when implementing the equations, we use the transposes. For instance,
the net output of layer 1 is W1^T x, where x is a column vector of dimension 2 x 1.
All neurons have a bias weight.

History:
I created this TF v2 version as follow:
1) Obtained a Tensorflow v1 from
 https://github.com/keras-team/keras/issues/956 and saved it as
 backpropagation_tf1_example.py. This code was not compatible with TF v2. 
2) Then I used
tf_upgrade_v2 --infile backpropagation_tf1_example.py --outfile backpropagation_tf2_example.py
to create a TF v2 version.
3) According to https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta
I added the line:
tf.compat.v1.disable_eager_execution()
and the code executed properly.
4) I made several modifications to initialize the weights with known values
and help tracking the training procedure.
5) I then compared training with different batch sizes.

Aldebaro, UFPA, September 2022.
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

#Initialize all weights to match the values adopted in the example at
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ 
#Initialise kernel (weights matrix) to required value
def kernel_init_layer1(shape, dtype='float'):
    # In this simple example, assume that the shape is (2, 2) 
    # and create a fixed array with this dimension for layer 1
    kernel = np.array([[0.15, 0.20],[0.25, 0.30]])
    return kernel 

#Initialise kernel (weights matrix) to required value
def kernel_init_layer2(shape, dtype='float'):
    # In this simple example, assume that the shape is (2, 2)
    # and create a fixed array with this dimension for layer 2
    kernel = np.array([[0.40, 0.45], [0.50, 0.55]])
    return kernel 

#Initialise kernel (bias vector) to required value
def kernel_init_bias1(shape, dtype='float'):
    # In this simple example, assume that the shape is (1, 2)
    # and create a fixed array with this dimension for the 
    # bias vector of layer 1
    kernel = np.array([0.35, 0.35]) #both neurons with the same value
    return kernel 

#Initialise kernel (bias vector) to required value
def kernel_init_bias2(shape, dtype='float'):
    # In this simple example, assume that the shape is (1, 2)
    # and create a fixed array with this dimension for the 
    # bias vector of layer 2
    kernel = np.array([0.60, 0.60]) #both neurons with the same value
    return kernel 

def create_and_compile_model():
    #Define the neural network model with dense layers. Syntax:
    #https://keras.io/api/layers/core_layers/dense/
    #The sigmoid activation function in Keras is the standard logistic function 1/(1+exp(-x)).
    #https://keras.io/api/layers/activations/
    model = Sequential()
    model.add(Dense(2, input_dim=2, use_bias=True,  bias_initializer=kernel_init_bias1,
            kernel_initializer=kernel_init_layer1, activation='sigmoid'))
    model.add(Dense(2, use_bias=True,  bias_initializer=kernel_init_bias2, 
            kernel_initializer=kernel_init_layer2, activation='sigmoid'))
    model.summary() # display the architecture
    #now compile the model, informing loss and performanc metrics that
    #should be computed along the training
    model.compile(loss='mse', metrics=['mse'])
    return model

def get_model_weigths(model):
    # Informs weights of a given model
    num_trainable_layers = len(model.trainable_weights)
    weights = list()
    for i in range(num_trainable_layers):
        #note that model.trainable_weights[i] is an object of
        # <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable
        #therefore (see e.g. https://stackoverflow.com/questions/33679382/how-do-i-get-the-current-value-of-a-variable )
        #you need to obtain its value via a TF session:
        weights.append(sess.run(model.trainable_weights[i]))
    return weights

# from https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta
# TF 2 does not use "eager" execution, so disable it:
tf.compat.v1.disable_eager_execution()
# Begin TensorFlow
sess = tf.compat.v1.InteractiveSession()
#sess.run(tf.compat.v1.initialize_all_variables())
sess.run(tf.compat.v1.global_variables_initializer())

model = create_and_compile_model()
# Define training and test sets for this regression problem
X = np.array([[1.2, 0], [-1.2, 1], [-1.2, -1]])
y = np.array([[0, -1.3], [0, 0.9], [0, 0.9]])

# We are going to compare with Keras optimizer. Define a learning rate:
learning_rate = 1  #you can change to 0.5 or any other reasonable value

k.set_value(model.optimizer.learning_rate, learning_rate)
print('Learning rate = ', k.get_value(model.optimizer.lr))
num_epochs = 1
batch_size = 1
verbose = 2
model.fit(X, y, epochs=num_epochs, batch_size=batch_size, shuffle=False, validation_split = 0,
verbose=verbose)
print('Learning rate = ', k.get_value(model.optimizer.lr))

print("2) Weights = ", get_model_weigths(model))

model = create_and_compile_model()
#print("3) Weights = ", get_model_weigths(model))
print("===START TRAINING THE NETWORK:===")
steps = 3 # steps of gradient descent
for s in range(steps):    
    print(" ############ Step = " + str(s) + " ############")
    print("1) ===BEFORE using any GRADIENT:===")
    #define input and target vectors, and also inform it to the loss function object
    inputs = X[s][np.newaxis] #define the input for the current iteration (step)
    #define the target vector for the current iteration (step)
    targets = y[s][np.newaxis] #notice that a sigmoid can output within range [0, 1]
    outputs = model.predict(inputs) #forward pass
    mse = mean_squared_error(targets, outputs) #calculate MSE
    #initialize loss object to be later incorporated to the gradients object that
    #enables the calculation of the symbolic gradients
    loss = losses.mean_squared_error(targets, model.output)
    #  ===== Obtain symbolic gradient to calculate numerical gradients =====
    gradients = k.gradients(loss, model.trainable_weights) #inform loss and weights
    if False: #enable with True in case you want to see the objects
        print("List of tensors representing the symbolic gradients:")
        for i in range(len(gradients)):
            print('symbolic gradient[',i,']=',gradients[i])
    print('Network input [x1, x2]:\n', inputs)
    print('Network output [out_o1, out_o2]:\n', outputs)
    print("targets:\n", targets)
    #show weights at the beginning of iteration s
    print('weights at the beginning of this step')
    for i in range(len(model.trainable_weights)):
        #note that model.trainable_weights[i] is an object of
        # <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable
        #therefore (see e.g. https://stackoverflow.com/questions/33679382/how-do-i-get-the-current-value-of-a-variable )
        #you need to obtain its value via a TF session:
        print('weights[',i,']=',sess.run(model.trainable_weights[i]))
    print("MSE with the initial weights:", mse)

    print("2) ===After applying GRADIENT:===")
    print("------------- Results for step (iteration) =", s)
    # ===== Calculate numerical gradient from symbolic gradients =====
    # evaluated_gradients is a list, show its contents:
    evaluated_gradients = sess.run(gradients, feed_dict={model.input: inputs})
    print('Gradients g to be used in new_weights = current_weights - learning_rate*g')
    for i in range(len(evaluated_gradients)):
        print('gradients[',i,']=',evaluated_gradients[i])

    # Apply ("step down") the gradient for each layer, subtracting the gradients
    # from current weights scaled by the learning rate:
    for i in range(len(model.trainable_weights)):
        sess.run(tf.compat.v1.assign_sub(model.trainable_weights[i], learning_rate*evaluated_gradients[i]))

    #show weights after gradient propagation of iteration s
    print('weights after gradient propagation in this step')
    for i in range(len(model.trainable_weights)):
        print('weights[',i,']=',sess.run(model.trainable_weights[i]))

    # print the MSE with new weights:
    outputs = model.predict(inputs)
    mse = mean_squared_error(targets, outputs)
    print("MSE with the new weights:", mse)

#Collect and show final results
final_outputs = model.predict(inputs)
final_mse = mean_squared_error(targets, final_outputs)

print("\n ===AFTER executing all GRADIENT descent steps===")
final_weigths = get_model_weigths(model)
print(final_weigths)
print("outputs:\n", final_outputs)
print("targets:\n", targets)
print("Final MSE = ", final_mse)