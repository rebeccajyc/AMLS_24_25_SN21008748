from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import numpy as np
from medmnist import BloodMNIST
import matplotlib.pyplot as plt


trainSet = BloodMNIST(split="train", download="True")
valSet = BloodMNIST(split="val", download="True")
testSet = BloodMNIST(split="test", download="True")

x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()

neighbours_list = []
accuracy_list = []

for i in range(100):
    knn_model = KNeighborsClassifier(n_neighbors=i+1)
    knn_model.fit(x_train,y_train)

    y_pred = knn_model.predict(x_test)

    accuracy = accuracy_score(y_pred, y_test)

    neighbours_list.append(i+1)
    accuracy_list.append(accuracy)

    print(f'n_neighbours:{i+1}, Accuracy:{accuracy}')

plt.scatter(neighbours_list, accuracy_list)
plt.show()