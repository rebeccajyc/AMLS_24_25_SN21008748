from medmnist import BreastMNIST
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT IMAGES
trainSet = BreastMNIST(split="train", download="True")
valSet = BreastMNIST(split="val", download="True")
testSet = BreastMNIST(split="test", download="True")

# PREPROCESSING - FLATTEN 2D IMAGES INTO 1D ARRAYS
x_train = np.concatenate((trainSet.imgs, valSet.imgs), axis=0)
y_train = np.concatenate((trainSet.labels, valSet.labels), axis=0)

y_train_hog = y_train.ravel()

x_test = testSet.imgs
y_test = testSet.labels

np.random.seed(42)

# CROSS VALIDATION (KFOLD) PARAMETERS
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
    
    scaler = StandardScaler()
    x_train_hog = scaler.fit_transform(x_train_hog)

    hog_model = RandomForestClassifier()
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





# RF GRID SEARCH WITH OPTIMUM HOG PARAMETERS

# TRAINING SET PREPROCESSING WITH HOG
y_train_rf = y_train.ravel()
y_test_rf = y_test.ravel()

x_train_rf = [hog(image,  orientations=best_params['orientations'], 
            pixels_per_cell=best_params['pixels_per_cell'], 
            cells_per_block=best_params['cells_per_block'], 
            block_norm=best_params['block_norm'])
            for image in x_train]

x_test_rf = [hog(image,  orientations=best_params['orientations'], 
            pixels_per_cell=best_params['pixels_per_cell'], 
            cells_per_block=best_params['cells_per_block'], 
            block_norm=best_params['block_norm'])
            for image in x_test]

scaler = StandardScaler()
x_train_rf = scaler.fit_transform(x_train_rf)
x_test_rf = scaler.transform(x_test_rf)

# RF GRID SEARCH PARAMETERS
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 50, 70],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# GRID SEARCH MODEL
model = RandomForestClassifier(random_state=10)
grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=kf)
grid_search.fit(x_train_rf, y_train_rf)

# DATAFRAME OF GRID SEARCH RESULTS
df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
df = df.sort_values(by='Accuracy', ascending=False)
print(df)
print(f'Best RF Parameters: {grid_search.best_params_}>>> Best Score: {grid_search.best_score_}')

# FINAL MODEL WITH BEST PARAMETERS FOR HOG AND RF
final_rf = RandomForestClassifier(random_state=10,
                                  n_estimators=grid_search.best_params_['n_estimators'], 
                                   max_depth=grid_search.best_params_['max_depth'], 
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                   max_features=grid_search.best_params_['max_features'],)
final_rf.fit(x_train_rf, y_train_rf)
y_pred = final_rf.predict(x_test_rf)
test_accuracy = accuracy_score(y_pred, y_test_rf)
print(f'Test Set Accuracy: {test_accuracy}')

# CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test_rf, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['malignant','benign'])
disp.plot(cmap=plt.cm.Oranges)
plt.title('Confusion Matrix',fontsize=13, weight='bold', pad=10)
plt.xlabel('Prediction', fontsize=11, weight='bold')
plt.ylabel('Actual', fontsize=11, weight='bold')
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_test_rf, y_pred, target_names=['0','1']))

# ROC-AUC CURVE
y_pred_prob = final_rf.predict_proba(x_test_rf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_rf, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, 'darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', weight='bold')
plt.legend(loc='lower right')
plt.show()
