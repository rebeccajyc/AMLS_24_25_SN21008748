from medmnist import BreastMNIST
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT IMAGES
trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

# TRAINING SET PREPROCESSING - COMBINE TRAIN AND VAL SET + FLATTEN 2D IMAGES INTO 1D ARRAYS
x_train = np.concatenate((trainSet.imgs, valSet.imgs), axis=0)
y_train = np.concatenate((trainSet.labels, valSet.labels), axis=0)

x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1).ravel()

# TEST SET PREPROCESSING - FLATTEN 2D IMAGES INTO 1D ARRAYS
x_test = testSet.imgs
y_test = testSet.labels

x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1).ravel()

# PREPROCESSING - DATA STANDARDISATION
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# CROSS VALIDATION (KFOLD) PARAMETERS
np.random.seed(42)
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)

# GRID SEARCH PARAMETERS
param_grid = {'C':[0.01, 0.1, 1, 10, 100, 200],
             'gamma':['scale', 'auto'],
            'kernel':['linear', 'sigmoid' ,'poly', 'rbf']}

# GRID SEARCH MODEL
model = svm.SVC()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf, verbose=1)
grid_search.fit(x_train,y_train)

# DATAFRAME OF GRID SEARCH RESULTS
df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df = df.sort_values(by='Accuracy', ascending=False)
print(df)
print(f'Best Parameters: {grid_search.best_params_} -> Best Score: {grid_search.best_score_}')

# FINAL MODEL
final_svm = svm.SVC(probability=True, C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'], random_state=10)

# EVALUATE FINAL MODEL ON TEST SET
final_svm.fit(x_train, y_train)
y_pred = final_svm.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {test_accuracy}')

# CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['malignant','benign'])
disp.plot(cmap=plt.cm.Oranges)
plt.title('Confusion Matrix',fontsize=13, weight='bold', pad=10)
plt.xlabel('Prediction', fontsize=11, weight='bold')
plt.ylabel('Actual', fontsize=11, weight='bold')
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_test, y_pred, target_names=['malignant','normal, benign']))

# ROC-AUC CURVE
y_proba = final_svm.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, 'darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', weight='bold')
plt.legend(loc='lower right')
plt.show()