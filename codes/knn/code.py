import numpy as np

x_train = np.loadtxt("train_set.txt")  # load from file to memory
x_test = np.loadtxt("test_set.txt")    # load from file to memory

N_train, M = x_train.shape  # N_train is the number of examples and M=3
N_test, M = x_test.shape    # N_test is the number of test examples

numberOfErrors = 0  # count number of errors

for i in range(N_test):  # go over all test set

    testFeatures = x_test[i, 0:2]  # extract only features (discard label)

    # find nearest neighbor in training set
    minimumDistance = np.inf  # initialize with large number
    nearestNeighborIndex = -1

    for j in range(N_train):  # go over all training set

        trainFeatures = x_train[j, 0:2]  # extract only features (discard label)

        distance = np.sum((testFeatures - trainFeatures) ** 2)  # squared norm

        if distance < minimumDistance:  # check if better than current
            nearestNeighborIndex = j  # store the current NN
            minimumDistance = distance  # update minimum distance

    classifierOutput = x_train[nearestNeighborIndex, 2]  # decision

    if classifierOutput != x_test[i, 2]:  # compare with correct label
        numberOfErrors = numberOfErrors + 1  # update number of errors

print("Number of errors:", numberOfErrors)