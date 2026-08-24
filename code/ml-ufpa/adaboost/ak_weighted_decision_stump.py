'''
This code implements a Decision Stump as a DecisionTreeClassifier of depth = 1.

Aldebaro. 2020-12-19.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

print(__doc__)

def load_lista1_dataset():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('lista1_dataset.csv', delimiter=',')
    X = my_data[:,:2] # two first parameters are input vector
    #y = my_data[:,2:]
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

def load_simple():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('../code_datasets/simple.csv', delimiter=',')
    X = my_data[:,:2] # fish length and weight
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

def load_prova():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('../datasets_hw_stumps/estudante_213584053_train.txt', delimiter=',')    
    X = my_data[:,:3]
    y = np.ravel(my_data[:,3:],order='C') #convert column vector into 1D array
    return X,y

if False:
    X, y = load_lista1_dataset()
else:
    #X, y = load_simple()
    X, y = load_prova()

print('datasets:')
print(X,y)

input_dimension = X.shape
if len(input_dimension) != 2:
    raise Exception('Input must be 2D')
num_features = input_dimension[1]
for k in range(num_features):
    print('feature', k, ', values and corresponding labels')
    sorted_indices = np.argsort(X[:,k])
    print(X[sorted_indices,k])
    print(y[sorted_indices])

#Below is an example about how to use weights for each sample in the training set
sample_weight = np.ones(y.shape, dtype=float)
#sample_weight[0] = 1000 #in this case we put more weigth on feature_0

decision_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,criterion='gini')
decision_stump.fit(X, y, sample_weight)

print('\nThe fitted decision stump:')
r = export_text(decision_stump) #, feature_names=feature_names) #, class_names=target_names)
print(r)

print('correct labels=',y)
print('   predictions=',(decision_stump.predict(X)))