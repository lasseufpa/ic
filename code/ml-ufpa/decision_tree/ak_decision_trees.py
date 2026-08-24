# Common imports
import numpy as np
import os
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text

# to make this notebook's output stable across runs
np.random.seed(100)

def load_simple():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('simple.csv', delimiter=',')
    X = my_data[:,:2] # inputs
    y = my_data[:,2:]
    return X,y

X,y = load_simple()

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42, criterion='gini') #entropy
tree_clf.fit(X, y)

ROOT_DIR = '.'
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

#Source.from_file(os.path.join(IMAGES_PATH, "tree.dot"))

feature_names = np.array(['length', 'weight'])
target_names = np.array(['pirarucu','piranha'])
export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=feature_names,
        class_names=target_names,
        rounded=True,
        filled=True
    )

plot_tree(tree_clf)

r = export_text(tree_clf) #, feature_names=feature_names) #, class_names=target_names)
print(r)

#print(X.shape)
#print(y.shape)

# # Predicting classes and class probabilities
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
