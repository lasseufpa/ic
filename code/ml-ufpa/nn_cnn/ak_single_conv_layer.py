#From
#https://stackoverflow.com/questions/45573582/how-to-specify-filter-in-keras-conv2d
#Adapted by Aldebaro. Jan 2, 2021.
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

import matplotlib.pyplot as plt
#plt.imshow(gkern())
#plt.show()
#You can then set this as an initial filter and freeze that layer so it no longer trains, which would look something like this,

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

#Build Keras model
model = Sequential()
#We would freeze training of the layers if we
# wanted to keep the given filter
our_own_layer = Conv2D(1, [2,2], kernel_initializer=kernel_init, 
                    input_shape=(4,4,1), padding="same")
our_own_layer.trainable = False
model.add(our_own_layer)
#Add some more layers here
#model.add(Conv2D(...)
model.build()

# To apply existing filter, we use predict with no training
out = model.predict(image)
print(out[0,:,:,0])