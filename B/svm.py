import time
from medmnist import BloodMNIST
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score 



trainSet = BloodMNIST(split="train", download="True")
valSet = BloodMNIST(split="val", download="True")
testSet = BloodMNIST(split="test", download="True")

x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()

kernels = ['linear', 'sigmoid' ,'poly', 'rbf']

print("ok")

for i, kernel in enumerate(kernels):
    startTime = time.time()
    
    model = svm.SVC( C=2.0,kernel= kernel)
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_pred, y_test)

    endTime = time.time()
    totalTime = endTime-startTime

    print(f"Kernel: {kernel}, Accuracy: {accuracy:.2f}, Time: {totalTime}")
