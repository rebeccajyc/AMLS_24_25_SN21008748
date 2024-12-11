from medmnist import BreastMNIST
import numpy as np
from sklearn import svm 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 

categories = ["benign", "malignant"]

trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

for i in range(len(trainSet)):
    x_train = np.array(trainSet[i][0]).flatten()
    y_train = np.array(trainSet[i][1]).flatten()

x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()

kernels = ['linear', 'sigmoid' ,'poly', 'rbf']

for i, kernel in enumerate(kernels):
    model = svm.SVC( C=2.0,kernel= kernel)
    model.fit(x_train,y_train) # fit to SVC

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_pred, y_test)

    print(f"Kernel: {kernel}, Accuracy: {accuracy:.2f}")
