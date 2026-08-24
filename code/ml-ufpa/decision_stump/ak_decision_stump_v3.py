'''
This code implements a Decision Stump as a DecisionTreeClassifier of depth = 1.

Aldebaro. 2020-12-19.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import numpy as np

DIMENSION = 2 #dimension of input space (number of input features)

print(__doc__)

def load_prova():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('../datasets_hw_stumps/estudante_213584053_train.txt', delimiter=',')    
    X = my_data[:,:DIMENSION]
    y = np.ravel(my_data[:,DIMENSION:],order='C') #convert column vector into 1D array
    return X,y

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

decision_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,criterion='gini')
decision_stump.fit(X, y)

print('\nThe fitted decision stump:')
r = export_text(decision_stump) #, feature_names=feature_names) #, class_names=target_names)
print(r)

print('correct labels=',y)
print('   predictions=',(decision_stump.predict(X)))