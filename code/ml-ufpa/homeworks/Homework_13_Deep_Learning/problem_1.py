'''
This code deals with problem 1 of homework Homework_13_Deep_Learning.
'''
import numpy as np
import tensorflow as tf
from cross_entropy_study import *

print("########### Categorical CE ###########")
true_pmf = np.array([0, 1, 0, 0])
estimated_pmf = np.array([0.1, 0.6, 0.1, 0.2])
cross_entropy_example = cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
print('Cross_entropy =',cross_entropy_example)

print("########### Binary CE ###########")
# This functions deals with a single pair of binomial distributions, represented by probabilities
# p and q.
p=0
q=0.8
# Function that deals with a single pair of binomial distributions,
# represented by probabilities p and q
bce1 = individual_binary_cross_entropy_v2(p, q, use_log2 = False)
print("Parcel 1 of binary cross-entropy (BCE) =", bce1)

p=1
q=0.4
bce2 = individual_binary_cross_entropy_v2(p, q, use_log2 = False)
print("Parcel 2 of binary cross-entropy (BCE) =", bce2)

print("Binary cross-entropy (BCE, average) =", 0.5*(bce1+bce2))
