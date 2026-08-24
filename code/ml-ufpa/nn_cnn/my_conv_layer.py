#From
#https://stackoverflow.com/questions/45573582/how-to-specify-filter-in-keras-conv2d
import numpy as np
import scipy.stats as st

def gkern(kernlen=[21,21], nsig=[3, 3]):
    """Returns a 2D Gaussian kernel array."""

    assert len(nsig) == 2
    assert len(kernlen) == 2
    kern1d = []
    for i in range(2):
        interval = (2*nsig[i]+1.)/(kernlen[i])
        x = np.linspace(-nsig[i]-interval/2., nsig[i]+interval/2., kernlen[i]+1)
        kern1d.append(np.diff(st.norm.cdf(x)))

    kernel_raw = np.sqrt(np.outer(kern1d[0], kern1d[1]))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

import matplotlib.pyplot as plt
plt.imshow(gkern([7,7]), interpolation='none')
plt.show()
#You can then set this as an initial filter and freeze that layer so it no longer trains, which would look something like this,

#exit(1)

from keras.models import Sequential
from keras.layers import Conv2D

#Set Some Image
image = [[4,3,1,0],[2,1,0,1],[1,2,4,1],[3,1,0,2]]

# Pad to "channels_last" format 
# which is [batch, width, height, channels]=[1,4,4,1]
image = np.expand_dims(np.expand_dims(np.array(image),2),0)

#Initialise to set kernel to required value
def kernel_init(shape,dtype='float'):
    kernel = np.zeros(shape)
    kernel[:,:,0,0] = gkern([shape[0], shape[1]])
    return kernel 

#Build Keras model
model = Sequential()
#We would freeze training of the layers if we
# wanted to keep a Gaussian filter
Gausslayer = Conv2D(1, [3,3], kernel_initializer=kernel_init, 
                    input_shape=(4,4,1), padding="valid")
Gausslayer.trainable = False
model.add(Gausslayer)
#Add some more layers here
#model.add(Conv2D(...)
model.build()

# To apply existing filter, we use predict with no training
out = model.predict(image)
print(out[0,:,:,0])