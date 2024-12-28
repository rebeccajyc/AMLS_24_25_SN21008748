from medmnist import BreastMNIST
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT IMAGES
trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

# PREPROCESSING - FLATTEN 2D IMAGES INTO 1D ARRAYS
x_train = np.concatenate((trainSet.imgs, valSet.imgs), axis=0)
y_train = np.concatenate((trainSet.labels, valSet.labels), axis=0)

x_test = testSet.imgs
y_test = testSet.labels

x_train_hog = []
y_train_hog = []

for image,label in zip(x_train, y_train):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    x_train_hog.append(hog_features)
    y_train_hog.append(label)

x_test_hog = []
y_test_hog = []

for image,label in zip(x_test, y_test):
    
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    x_test_hog.append(hog_features)
    y_test_hog.append(label)

x_train_hog = np.array(x_train_hog)
x_test_hog = np.array(x_test_hog)
y_train_hog = np.array(y_train_hog).ravel()
y_test_hog = np.array(y_test_hog).ravel()

# CROSS VALIDATION (KFOLD) PARAMETERS
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)

# GRID SEARCH PARAMETERS
param_grid = {'C':[0.01, 0.1, 1, 10, 100, 200],
             'gamma':['scale', 'auto'],
            'kernel':['linear', 'sigmoid' ,'poly', 'rbf']}

# GRID SEARCH MODEL
model = svm.SVC()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf)
grid_search.fit(x_train_hog,y_train_hog)

df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
print(df)
print(f'Best Parameters: {grid_search.best_params_} -> Best Score: {grid_search.best_score_}')

# FINAL MODEL WITH BEST PARAMETERS 
final_svm = svm.SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'])
final_svm.fit(x_train_hog, y_train_hog)

y_pred = final_svm.predict(x_test_hog)

# ACCURACY COMPARISON WITH CROSS VALIDATION
accuracy = accuracy_score(y_pred, y_test_hog)
print(f'final_svm Accuracy: {accuracy}')

cross_val = cross_val_score(final_svm, x_train_hog, y_train_hog, cv=kf)
print(f'final_svm Accuracy (Cross Validation - K-Folds): {cross_val}')
print(f'final_svm Mean Accuracy (Cross Validation - K-Folds): {cross_val.mean()}')

# CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test_hog, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0,1])

disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix',fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_test_hog, y_pred, target_names=['0','1']))

###
_, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
plt.imshow(hog_image, cmap='gray')
plt.title("HOG Features")
plt.show()