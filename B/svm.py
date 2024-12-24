import time
from medmnist import BloodMNIST
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT IMAGES
trainSet = BloodMNIST(split="train", download="True")
valSet = BloodMNIST(split="val", download="True")
testSet = BloodMNIST(split="test", download="True")

x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_val = np.array([np.array(valSet[i][0]).flatten() for i in range(len(valSet))])
y_val = np.array([valSet[i][1].flatten() for i in range(len(valSet))]).ravel()

x_train = np.concatenate((x_train, x_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()

# CROSS VALIDATION (KFOLD) PARAMETERS
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)

# GRID SEARCH PARAMETERS
param_grid = {'C':[0.01, 0.1, 1, 10, 100, 200],
             'gamma':['scale', 'auto'],
            'kernel':['linear', 'sigmoid' ,'poly', 'rbf']}

# GRID SEARCH MODEL
startTime = time.time()

model = svm.SVC()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf)
grid_search.fit(x_train,y_train)

endTime = time.time()
totalTime = endTime-startTime

df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
print(f'Time taken -> {totalTime}')
print(df)
print(f'Best Parameters: {grid_search.best_params_} -> Best Score: {grid_search.best_score_}')

# FINAL MODEL WITH BEST PARAMETERS 
final_svm = svm.SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'])
final_svm.fit(x_train, y_train)

y_pred = final_svm.predict(x_test)

# ACCURACY COMPARISON WITH CROSS VALIDATION
accuracy = accuracy_score(y_pred, y_test)
print(f'final_svm Accuracy: {accuracy}')

cross_val = cross_val_score(final_svm, x_train, y_train, cv=kf)
print(f'final_svm Accuracy (Cross Validation - K-Folds): {cross_val}')
print(f'final_svm Mean Accuracy (Cross Validation - K-Folds): {cross_val.mean()}')