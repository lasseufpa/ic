'''
This code implements the original (discrete) AdaBoost from [1] with the notation
and initialization procedure adopted in [2].

[1] A decision-theoretic generalization of on-line learning and an application to boosting,
Yoav Freund and Robert E. Schapire. Journal of Computer and System Sciences, 55(1):119–139,
August 1997, https://www.sciencedirect.com/science/article/pii/S002200009791504X

[2]  Robust Real-Time Face Detection, Paul Viola and Michael Jones, International Journal
of Computer Vision 57(2), 137–154, 2004. https://link.springer.com/article/10.1023/B:VISI.0000013087.49260.fb 
PDF at https://www.face-rec.org/algorithms/boosting-ensemble/16981346.pdf

For more information about distinct AdaBoost implementations and the one adopted in scikit-learn, see:
https://docs.google.com/document/d/1ty3XhXlXMcJ67Rl_9DoiRUJpCSLW2C7wFLEO3RUdHi4/edit?usp=sharing 

Aldebaro. 2020-12-19.
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

T = 3 #number of weak learners
show_training_info = True

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
    my_data = np.genfromtxt('simple.csv', delimiter=',')
    X = my_data[:,:2] # fish length and weight
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

def adaboost_predict(x, weak_classifiers, alphas, num_weak_classifiers):
    decision_threshold = 0.5 * np.sum(alphas)
    score = 0
    for t in range(num_weak_classifiers):
        score += alphas[t] * weak_classifiers[t].predict(x)
    return np.array([score >= decision_threshold]).astype(int)

def weighted_error(prediction, correct_label, weights):
    error_indices = np.argwhere(prediction != correct_label)
    return np.sum(weights[error_indices])

if False:
    X, y = load_lista1_dataset()
else:
    X, y = load_simple()

num_training_examples = X.shape[0]

#Initialization of instance weights
sample_weight = np.zeros(y.shape, dtype=float)
index_of_positive_classes = np.argwhere(y==1)
index_of_negative_classes = np.argwhere(y==0)
num_positive = len(index_of_positive_classes)
num_negative = len(index_of_negative_classes)

sample_weight[index_of_positive_classes] = 1.0 / (2*num_positive)
sample_weight[index_of_negative_classes] = 1.0 / (2*num_negative)

print('Initial sample_weight', sample_weight)

weak_classifiers = list()
alphas = list()
for t in range(T):
    print("######## Iteration t=",t)

    #1) normalize weight
    weights_sum = np.sum(sample_weight)
    sample_weight = sample_weight / weights_sum

    if show_training_info:
        print('sample_weight=',sample_weight)

    #2) select the best weak classifier with respect to weighthed error
    decision_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    decision_stump.fit(X, y, sample_weight)

    #3) Define the weak classifier h_t(x)
    weak_classifiers.append(decision_stump)

    #4) Update the weights based on the decision of the classifier that
    #incorporates all previously designed weak classifiers
    y_pred = decision_stump.predict(X)

    epsilon_weighted_error = weighted_error(y_pred, y, sample_weight)
    #AK: function below return strange, large values
    #epsilon_weighted_error = decision_stump.score(X, y, sample_weight)

    beta = epsilon_weighted_error / (1.0 - epsilon_weighted_error)
    this_alpha = np.log(1.0/beta)
    alphas.append(this_alpha)

    predictions_of_weak_classifier_t = decision_stump.predict(X)

    errors_i = np.array((predictions_of_weak_classifier_t != y)).astype(int)
    sample_weight *= beta ** (1.0-errors_i)
    #sample_weight = sample_weight / np.sum(sample_weight)

    if show_training_info:
        print('predictions_of_weak_classifier_t=',predictions_of_weak_classifier_t)
        print('                  correct labels=',y)
        print('epsilon_weighted_error=', epsilon_weighted_error)
        #print('sample_weight=',sample_weight)
        print('beta_t=',beta,'alpha_t=',this_alpha)
        print('errors_i=',errors_i)

for t in range(T):
    print('t=', t, ', alpha_t=',alphas[t])
    r = export_text(weak_classifiers[t]) #, feature_names=feature_names) #, class_names=target_names)
    print(r)
print('threshold =', 0.5 * np.sum(alphas))

predictions_of_strong_classifier = adaboost_predict(X, weak_classifiers, alphas, T)
print('predictions_of_strong_classifier=',predictions_of_strong_classifier)
print('                          labels=',y)

print('\n\n#### Comparison with scikit-learn ####')
ada_discrete = AdaBoostClassifier(
    base_estimator=decision_stump,
    learning_rate=1,
    n_estimators=T,
    algorithm="SAMME")
ada_discrete.fit(X, y)
ada_discrete_pred = ada_discrete.predict(X)
error_indices = np.argwhere(ada_discrete_pred != y)
print('error_indices=',error_indices)
print('Scikit-learn misclassification error rate (%)=',100.0*len(error_indices)/num_training_examples)

print('Scikit-learn AdaBoost classifier:')
weak_learners_list = ada_discrete.estimators_
for t in range(len(weak_learners_list)):
    print('t=',t,'alpha?=',ada_discrete.estimator_weights_[t])
    r = export_text(weak_learners_list[t]) #, feature_names=feature_names) #, class_names=target_names)
    print(r)
