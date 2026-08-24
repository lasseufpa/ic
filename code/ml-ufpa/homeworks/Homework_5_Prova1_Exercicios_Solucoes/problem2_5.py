'''
Problem 2.5 from https://docs.google.com/document/d/1Mb9gY4p7cjJj9pDmMq7Vxkl6VjBcmfhnSmAuyVyanAQ/edit?usp=sharing
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from graphviz import Source
from sklearn.tree import export_graphviz
import os

print(__doc__)

np.random.seed(30)

def load_simple():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('simple.csv', delimiter=',')
    X = my_data[:,:2] # fish length and weight
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

def gini(probability_mass_function):
    '''
    Calculates GINI index.
    '''
    return 1.0 - np.sum(probability_mass_function ** 2.0)

def entropy(probability_mass_function, use_log2 = True):
    '''
    Calculates entropy.
    '''
    #probability_mass_function = np.array([49, 5])/54 #to reproduce pg 177 from Geron's book
    #probability_mass_function = np.array([4, 2])/6 
    zero_prob_indices = np.argwhere(probability_mass_function == 0)
    probability_mass_function[zero_prob_indices] = 10 #any number to avoid numerical errors
    if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
        information = np.log2(1.0 / probability_mass_function)
    else: #use log_e and, therefore, entropy in neper as in Geron's book
        information = np.log(1.0 / probability_mass_function)
    probability_mass_function[zero_prob_indices] = 0 #restore zero values
    weighted_information  = probability_mass_function * information    
    entropy = np.sum(weighted_information)
    return entropy

def get_probabilities_given_predictions(predictions):
    class_0_indices = np.argwhere(predictions == 0)
    class_1_indices = np.argwhere(predictions == 1)
    num_0 = len(class_0_indices)
    num_1 = len(class_1_indices)
    num_total = num_0 + num_1
    probabilities = np.array([num_0 / num_total, num_1 / num_total])
    return probabilities

X, y = load_simple()

input_dimension = X.shape
if len(input_dimension) != 2:
    raise Exception('Input must be 2D')
num_training_examples = input_dimension[0]
num_features = input_dimension[1]
for k in range(num_features):
    print('sorted feature', k, ', values and corresponding labels')
    sorted_indices = np.argsort(X[:,k])
    print(X[sorted_indices,k])
    print(y[sorted_indices])

decision_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,criterion='gini')
#decision_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,criterion='entropy')
decision_stump.fit(X, y)

feature_names = np.array(['length', 'weight'])
target_names = np.array(['pirarucu','piranha'])
export_graphviz(
        decision_stump,
        out_file=("problem2_5_tree.dot"),
        feature_names=feature_names,
        class_names=target_names,
        rounded=True,
        filled=True
    )


print('\nThe fitted decision stump:')
r = export_text(decision_stump) #, feature_names=feature_names) #, class_names=target_names)
print(r)

print('correct labels=',y)
print('   predictions=',(decision_stump.predict(X)))

probabilities = get_probabilities_given_predictions(y)
print('probabilities=', probabilities)
print('GINI=', gini(probabilities))
print('Entropy=', entropy(probabilities))

#Tree structure:
#https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
features = decision_stump.tree_.feature
thresholds = decision_stump.tree_.threshold
print('features', features)
print('thresholds',thresholds)

leaves = decision_stump.apply(X)
print(leaves)

indices_examples_leaf_1 = np.argwhere(leaves == 1)
indices_examples_leaf_2 = np.argwhere(leaves == 2)

labels_examples_leaf_1 = y[indices_examples_leaf_1]
labels_examples_leaf_2 = y[indices_examples_leaf_2]

#Leaf 1
probabilities = get_probabilities_given_predictions(labels_examples_leaf_1)
print('probabilities=', probabilities)
print('GINI=', gini(probabilities))
print('Entropy=', entropy(probabilities))

#Leaf 2
probabilities = get_probabilities_given_predictions(labels_examples_leaf_2)
print('probabilities=', probabilities)
print('GINI=', gini(probabilities))
print('Entropy=', entropy(probabilities))

print('If you have graphviz installed, run')
print('dot -Tpng problem2_5_tree.dot -o problem2_5_tree.png')
print('to convert the dot file into a png file')

p=np.array([4/7, 3/7])
print(entropy(p,use_log2=False))