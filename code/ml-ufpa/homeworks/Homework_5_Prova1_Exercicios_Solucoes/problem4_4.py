'''
From:
https://stats.stackexchange.com/questions/39243/how-does-one-interpret-svm-feature-weights

'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def convert_linear_SVM_to_perceptron(support_vectors, dual_coef):
    '''
    Use the linearity of the inner-product.
    Aldebaro. 2020-12-20
    '''
    dual_coef = np.ravel(dual_coef,order='C') #convert to a 1D vector
    num_support_vectors = len(dual_coef)
    if support_vectors.shape[0] != num_support_vectors:
        raise Exception('support_vectors.shape[0] != num_support_vectors')
    input_space_dimension = support_vectors.shape[1]
    perceptron_weights = np.zeros((input_space_dimension))
    for sv in range(num_support_vectors):
        perceptron_weights += dual_coef[sv] * support_vectors[sv]
    return perceptron_weights

def calculate_intercept(perceptron_weights,support_vectors,support_vector_indices,labels):
    '''
    From:
    https://stats.stackexchange.com/questions/211310/deriving-the-intercept-term-in-a-linearly-separable-and-soft-margin-svm
    Aldebaro. 2020-12-20
    '''
    num_support_vectors = support_vectors.shape[0]
    #input_space_dimension = support_vectors.shape[1]
    max_negative = -1e30
    min_positive = 1e30
    for i in range(num_support_vectors):
        #note that labels has all training examples, not only support vectors,
        #so use support_vector_indices to get the proper index in training set
        this_label = labels[support_vector_indices[i]]
        this_inner_product = np.inner(perceptron_weights, support_vectors[i])
        if this_label > 0:
            if this_inner_product < min_positive:
                min_positive = this_inner_product
        else:
            if this_inner_product > max_negative:
                max_negative = this_inner_product
    return -(max_negative+min_positive)/2.0

np.random.seed(30) #reproducible experiments

X=[[2,3],[6,-1]] #only the support vectors
y=[0,1] #labels

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter

svm = svm.SVC(kernel='linear', C=C, verbose=1, shrinking=False)
print('Invoke LibSVM in C/C++')
svm.fit(X,y)

print('\n#### SVM with SVC ####\n', svm.get_params())
print('svm.coef_=',svm.coef_)
print('svm.intercept_=',svm.intercept_)
print('svm.n_support_=',svm.n_support_)
print('svm.support_=',svm.support_)
print('svm.support_vectors_=',svm.support_vectors_)
print('svm.dual_coef_=',svm.dual_coef_)
print('svm.decision_function(X)=',svm.decision_function(X))
perceptron_weights = convert_linear_SVM_to_perceptron(svm.support_vectors_, svm.dual_coef_)
print('Estimated perceptron_weights=', perceptron_weights)
bias = calculate_intercept(perceptron_weights,svm.support_vectors_,svm.support_,y)
print('Estimated bias=', bias)
