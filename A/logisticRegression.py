from medmnist import BreastMNIST
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()


param_grid=[{'max_iter':[1,10,100,1000, 2000]}]
    
logRegress = LogisticRegression(solver='saga')

grid = GridSearchCV(estimator=logRegress,
                      param_grid=param_grid,
                      scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)


final_logRegress = LogisticRegression(solver='saga', max_iter=grid.best_params_['max_iter'])
final_logRegress.fit(x_train, y_train)

y_pred = final_logRegress.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)

print(accuracy) 
