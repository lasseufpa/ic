'''
This code provides an example on calculating the confusion matrix
and also showing the decision regions when the number of input
features is 2 (such that they can be displayed in a 2D figure).

It uses a Decision Stump (as a DecisionTreeClassifier of depth = 1)
as classifier, but this can be changed by the programmer that wants
to adopt another classifier.

Aldebaro. 2022-March-16.
'''

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as sp
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import numpy as np


'''
Load dataset. You will need to change it to point to your input file.
'''
def load_prova():
    my_data = np.genfromtxt('MUDE_POR_SEU_ARQUIVO/estudante_201906840033_train.txt', delimiter=',')        
    X = my_data[:,:2]
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

'''
Train and test decision stump.
'''
def train_test(X, y):

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

    return decision_stump

'''
Plot confusion matrix.
'''
def plot_confusion_matrix(clf, X, y):
    y_pred   = clf.predict(X)
    conf_mtx = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10,6))
    sns.heatmap(conf_mtx, cmap=sns.cm.rocket_r, square=True, linewidths=0.1,
                annot=True, fmt='d', annot_kws={"fontsize": 8})
    plt.tick_params(axis='both', which='major', labelsize=10,
                    bottom=False, top=False, left=False,
                    labelbottom=False, labeltop=True)
    plt.yticks(rotation=0)
    plt.title("Confusion matrix")

""" Plot the classifier decision regions
"""
def plot_decision_boundary(classifier, X, y, legend=False, plot_training=True):

    num_classes = int(np.max(y))+1 #e.g. 16 for QAM-16
    axes = [np.min(X[:,0]), np.max(X[:,0]),np.min(X[:,1]), np.max(X[:,1])]
    #print(axes)
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = classifier.predict(X_new).reshape(x1.shape)

    # Set different color for each class
    custom_cmap = cm.get_cmap('tab20')
    colors = custom_cmap.colors[:num_classes]
    levels = np.arange(num_classes + 2) - 0.5

    plt.contourf(x1, x2, y_pred, levels=levels, colors=colors, alpha=0.3)

    if plot_training:
        for ii in range(num_classes):
            selected_indices = np.argwhere(y==ii)
            selected_indices = selected_indices.reshape((-1,))
            plt.plot(X[selected_indices, 0], X[selected_indices, 1], "o",
                     c=colors[ii], label=f'{ii}')
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    plt.title("Decision regions")
    if legend:
        plt.legend(title='Classes', bbox_to_anchor=(1, 1), loc='upper left',
                   ncol=2, handleheight=2, labelspacing=0.05, frameon=False)


if __name__ == '__main__':
    print(__doc__)
    X, y = load_prova() #load dataset
    classifier = train_test(X, y) #train decision stump
    plot_decision_boundary(classifier, X, y, legend=False, plot_training=True) #plot regions
    plot_confusion_matrix(classifier, X, y) #confusion matrix
    plt.show()
