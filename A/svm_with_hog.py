from medmnist import BreastMNIST
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT IMAGES
trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

# TRAINING SET PREPROCESSING - COMBINE TRAIN AND VAL SET
x_train = np.concatenate((trainSet.imgs, valSet.imgs), axis=0)
y_train = np.concatenate((trainSet.labels, valSet.labels), axis=0)

y_train_hog = y_train.ravel()

# TEST SET PREPROCESSING
x_test = testSet.imgs
y_test = testSet.labels


# CROSS VALIDATION (KFOLD) PARAMETERS
np.random.seed(42)
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)

# HOG GRID SEARCH PARAMETERS
best_params = None
best_score = 0
hog_grid_results = []

hog_param_grid = {
    'orientations': [6, 9, 12],
    'pixels_per_cell': [(4, 4), (6, 6), (8, 8)],
    'cells_per_block': [(1, 1), (2, 2), (3, 3)],
    'block_norm': ['L2-Hys', 'L1', 'L2'],
}

# HOG GRID SEARCH
for params in ParameterGrid(hog_param_grid):
    x_train_hog = [hog(image,  orientations=params['orientations'], 
            pixels_per_cell=params['pixels_per_cell'], 
            cells_per_block=params['cells_per_block'], 
            block_norm=params['block_norm'])
            for image in x_train]
    
    #scaler = StandardScaler()
    #x_train_hog = scaler.fit_transform(x_train_hog)

    hog_model = svm.SVC(C=1.00, gamma='scale', kernel='rbf')
    cv_scores = cross_val_score(hog_model, x_train_hog, y_train_hog, cv=kf)
    mean_cv_scores = cv_scores.mean()

    hog_grid_results.append({
        'orientations': params['orientations'],
        'pixels_per_cell': params['pixels_per_cell'],
        'cells_per_block': params['cells_per_block'],
        'block_norm': params['block_norm'],
        'mean_accuracy': mean_cv_scores
    })

    # FINDING BEST PERFORMANCE
    if cv_scores.mean()>best_score:
        best_score=cv_scores.mean()
        best_params=params

# HOG GRID SEARCH RESULTS
df_hog = pd.DataFrame(hog_grid_results)
df_hog = df_hog.sort_values(by='mean_accuracy', ascending=False)
print(df_hog)
print(f'Best parameters: {best_params}>>>Best Score: {best_score}')




# SVM GRID SEARCH WITH OPTIMUM HOG PARAMETERS

# TRAINING SET PREPROCESSING WITH HOG
x_train_svc = []
y_train_svc = []

for image, label in zip(x_train, y_train):
    hog_features = hog(image, orientations=best_params['orientations'], 
                       pixels_per_cell=best_params['pixels_per_cell'], 
                       cells_per_block=best_params['cells_per_block'],
                       visualize=False)
    x_train_svc.append(hog_features)
    y_train_svc.append(label)

# TEST SET PREPROCESSING WITH HOG
x_test_svc = []
y_test_svc = []

for image,label in zip(x_test, y_test):
    hog_features = hog(image, orientations=best_params['orientations'], 
                       pixels_per_cell=best_params['pixels_per_cell'], 
                       cells_per_block=best_params['cells_per_block'],
                       visualize=False)
    x_test_svc.append(hog_features)
    y_test_svc.append(label)

x_train_svc = np.array(x_train_svc)
x_test_svc = np.array(x_test_svc)
y_train_svc = np.array(y_train_svc).ravel()
y_test_svc = np.array(y_test_svc).ravel()


#x_train_svc = scaler.fit_transform(x_train_svc)
#x_test_svc = scaler.transform(x_test_svc)


# SVM GRID SEARCH PARAMETERS
param_grid = {'C':[0.01, 0.1, 1, 10, 100, 200],
             'gamma':['scale', 'auto'],
            'kernel':['linear', 'sigmoid' ,'poly', 'rbf']}

# GRID SEARCH MODEL
model = svm.SVC()
grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=kf)
grid_search.fit(x_train_svc, y_train_svc)

# DATAFRAME OF GRID SEARCH RESULTS
df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df = df.sort_values(by='Accuracy', ascending=False)
print(df)
print(f'Best SVC Parameters: {grid_search.best_params_}>>> Best Score: {grid_search.best_score_}')

# FINAL MODEL WITH BEST PARAMETERS FOR HOG AND SVM
final_svm = svm.SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'])
final_svm.fit(x_train_svc, y_train_svc)
y_pred = final_svm.predict(x_test_svc)

# ACCURACY COMPARISON WITH CROSS VALIDATION
accuracy = accuracy_score(y_pred, y_test_svc)
print(f'final_svm Accuracy: {accuracy}')

cross_val = cross_val_score(final_svm, x_train_svc, y_train_svc, cv=kf)
print(f'final_svm Accuracy (Cross Validation - K-Folds): {cross_val}')
print(f'final_svm Mean Accuracy (Cross Validation - K-Folds): {cross_val.mean()}')

# CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test_svc, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix',fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_test_svc, y_pred, target_names=['0','1']))