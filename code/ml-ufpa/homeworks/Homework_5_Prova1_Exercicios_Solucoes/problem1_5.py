import numpy as np

kernel_order = 4

def load_lista1_dataset():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('lista1_dataset.csv', delimiter=',')
    X = my_data[:,:2] # two first parameters are input vector
    y = my_data[:,2:]
    return X,y

def polynomial_kernel(x1, x2, order):
    return np.inner(x1,x2) ** order

def svm_output(input_vector, training_set, support_vector_indices, lambdas, bias):
    #calculate summation
    score = 0
    num_support_vectors = len(support_vector_indices)
    for n in range(num_support_vectors):
        kernel_value = polynomial_kernel(input_vector, training_set[support_vector_indices[n]], kernel_order)
        lambda_value = lambdas[n]
        score += lambda_value * kernel_value
    return score + bias

X, y = load_lista1_dataset()
support_vector_indices = np.array([0,2,5], dtype=int)
lambdas = np.array([-1,1,1], dtype=float)
test_set_input = np.array([[3,-1], [0,-2], [3,2], [0,-1]])
test_set_output = np.array([0,0,1,1])
bias = -3
N=4
for i in range(N):
    input_vector = test_set_input[i]
    score = svm_output(input_vector, X, support_vector_indices, lambdas, bias)
    print('score', score)
    print('classe predita', int(score > 0))