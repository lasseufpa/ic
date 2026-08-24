import numpy as np

gradients0_example1 = np.array([[0.04, 0.05], [0.,         0.        ]])
gradients0_example2 = np.array([[-0.01,  0.02], [ 0.01, -0.02]])
gradients0_example3 = np.array([[-0.01,   0.08], [-0.01, 0.03]])

gradients0_minibatch_average = (1.0/3) * (gradients0_example1 + gradients0_example2 + gradients0_example3)
print("gradients0_minibatch_average =", gradients0_minibatch_average)

#update
weights0 = np.array([[ 0.1,  0.3], [-0.4,  0.2]])
print("weights0 =", weights0)
lr = 1 #learning rate
weights0_new = weights0 - lr * gradients0_minibatch_average
print("weights0_new =", weights0_new)