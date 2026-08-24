import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing

np.random.seed(30)

def load_homework3():
    file_name = '../bayes_datasets/estudante_201706840076_train.txt'
    my_data = np.genfromtxt(file_name, delimiter=',')
    X = my_data[:,:3] # 3 features
    y = np.ravel(my_data[:,3:],order='C') #convert column vector into 1D array
    return X,y

def load_simple():
    #from numpy import genfromtxt
    my_data = np.genfromtxt('simple.csv', delimiter=',')
    X = my_data[:,:2] # fish length and weight
    y = np.ravel(my_data[:,2:],order='C') #convert column vector into 1D array
    return X,y

C = 3 #number of classes
threshold_value = -0.75

if True:
    #X,y = load_simple()
    X,y = load_homework3()
    x_test = X #test with train set
    K = X.shape[1] #number of features
    y_test = y
else:
    N = 3000 #number of training examples
    K = 4 #number of features
    max_feature_value = 10
    X = np.random.randint(max_feature_value, size = (N, K)) #3000 x 4 matrix with elements [0, 9]
    y = np.random.randint(C, size = (N,))
    x_test = np.random.randint(max_feature_value, size = (1, K))
    y_test = np.array([0]) #random class

print('Original input features:')
print(X)
print('True labels:')
print(y)

#Make it mean=0 and variance=1
scaler = preprocessing.StandardScaler().fit(X)
print('Original means:', scaler.mean_)
print('Original standard deviations:', scaler.scale_)
X_scaled = scaler.transform(X)
print('Scaled (standardized) input features:')
print(X_scaled)

BNBclf = BernoulliNB(alpha=0.0, binarize=threshold_value, fit_prior=True)
BNBclf.fit(X_scaled, y)
print(BNBclf)

print('#Report:')
binarized_features =np.asarray(X_scaled > threshold_value, dtype=int)
print('binarized_features and their labels:\n',np.concatenate((binarized_features, np.reshape(y,(len(y),1))), axis=1))
print('class_count_',BNBclf.class_count_)
print('### Priors:')
print('class_log_prior_ (natural log)', BNBclf.class_log_prior_)
estimated_priors = np.exp(BNBclf.class_log_prior_)
print('class_prior_', estimated_priors, 'and their sum=',np.sum(estimated_priors))
print('### Features')
print('feature_count_ (Dimension: classes x features):\n',BNBclf.feature_count_)
print('feature_log_prob_',BNBclf.feature_log_prob_)
estimated_feature_prob = np.exp(BNBclf.feature_log_prob_)
print('estimated_feature_prob',estimated_feature_prob)
print('### Predictions using training set as test set')
print('predictions=',(BNBclf.predict(x_test)))
print('predict_log_proba=',(BNBclf.predict_log_proba(x_test)))
print('predict_proba=',(BNBclf.predict_proba(x_test)))
print('score (fraction of correct predictions)=',(BNBclf.score(x_test,y_test)))
