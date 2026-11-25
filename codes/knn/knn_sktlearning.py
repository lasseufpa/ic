import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# carrega os dados de treino e teste
x_train = np.loadtxt("train_set.txt")
x_test = np.loadtxt("test_set.txt")

# separa as caracteristicas (as duas primeiras colunas) e as classes (ultima coluna)
X_train = x_train[:, 0:2]
y_train = x_train[:, 2]

X_test = x_test[:, 0:2]
y_test = x_test[:, 2]

# cria o classificador KNN com vizinho mais proximo
knn = KNeighborsClassifier(n_neighbors=1)

# treina o modelo com os dados de treino
knn.fit(X_train, y_train)

# faz a previsao para os dados de teste
y_pred = knn.predict(X_test)

# compara as previsoes com os labels corretos para contar erros
number_of_errors = np.sum(y_pred != y_test)

# mostra o resultado final
print("Número de erros:", int(number_of_errors))
print("Valores esperados:", y_test)
print("Valores previstos:", y_pred)
