#First from:
#https://stackoverflow.com/questions/45573582/how-to-specify-filter-in-keras-conv2d
#and later with bias initialization from:
#https://stackoverflow.com/questions/40708169/how-to-initialize-biases-in-a-keras-model/44148506
#https://keras.io/api/layers/initializers/
#Adapted by Aldebaro. Feb 2, 2021.
import numpy as np
import scipy.stats as st
from keras.models import Sequential
from keras.layers import Conv2D

def gkern():
    '''
    Generates a given 2D kernel
    '''
    my_kernel = [[1,-1],[1,-1]]
    return my_kernel

#Set Some input Image
image = [[1,0,3,0],[250,2,0,0],[3,1,255,1],[1,0,3,2]]

# Pad to "channels_last" format 
# which is [batch, width, height, channels]=[1,4,4,1]
image = np.expand_dims(np.expand_dims(np.array(image),2),0)

#Initialise to set kernel to required value
def kernel_init(shape,dtype='float'):
    kernel = np.zeros(shape)
    kernel[:,:,0,0] = gkern() #gkern([shape[0], shape[1]])
    return kernel 

print("### Result with Keras model with 1 filter and without bias (initialized as zero)")
model_without_bias = Sequential()
our_own_layer = Conv2D(1, [2,2], kernel_initializer=kernel_init,
                    bias_initializer='zeros',
                    input_shape=(4,4,1), padding="same")
our_own_layer.trainable = False
model_without_bias.add(our_own_layer)
#Add some more layers here
#model.add(Conv2D(...)
model_without_bias.build()

# To apply existing filter, we use predict with no training
out = model_without_bias.predict(image)
print(out[0,:,:,0])

print("### Result with Keras model with 1 filter and with bias (initialized as one)")
model_with_bias = Sequential()
our_own_layer = Conv2D(1, [2,2], kernel_initializer=kernel_init,
                    bias_initializer='ones',
                    input_shape=(4,4,1), padding="same")
our_own_layer.trainable = False
model_with_bias.add(our_own_layer)
#Add some more layers here
#model.add(Conv2D(...)
model_with_bias.build()

# To apply existing filter, we use predict with no training
out2 = model_with_bias.predict(image)
print(out2[0,:,:,0])