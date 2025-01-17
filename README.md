# AMLS_assignment24_25
This assignment explores two tasks involving datasets from MedMNIST.

**Task A (Binary Classification)** utilises the BreastMNIST dataset to classify breast tumours as 'benign' or 'malignant'. Models used include: SVM, SVM with HOG feature extraction, random forest with HOG feature extraction, logistic regression and CNN

**Task B (Multiclass Classification)** utilises the BloodMNIST dataset to classfy blood cells into eight distinct classes.
Models used include: Custom CNN and ResNet-50 transfer learning

## Contents

The file 'main.py' outputs the final test results from all models (Task A and B) including test accuracy, confusion matirix, classification report and ROC curve for each model. 
The files in Folders 'A' and 'B' are the original files used during the course of the assignment.
### A
| FILE                   | EXPLANATION       |
|:---------------------- |:------------------|
| svm.py                 | SVM model with grid search      |
| svm_with_hog.py        | SVM with HOG model with grid searches |
| randomForest.py        | Random Forest model with grid searches |
| logisticRegression.py  | Logistic Regression model with grid search |
| CNN_A.py               | Final CNN model |
| cnn.pth                | Trained CNN model weights used for main.py |

### B  
| FILE                   | EXPLANATION       |
|:---------------------- |:------------------|
| CNN_A.py               | Final custom CNN model |
| resnet50.py            | Final ResNet-50 transfer learning model |
| cnn.pth                | Trained custom CNN model weights used for main.py |
| resnet50.pth           | Trained ResNet-50 CNN model weights used for main.py |


## Required libraries
- medmnist
- torch
- torchvision
- tqdm
- sklearn
- matplotlib
- pandas
- numpy
- copy
- sys
- random

## Set-up Instructions
1. Install all required libaries listed 
2. Ensure correct directory and run the 'main.py' file in terminal using the following:
```
python main.py
```
3. Test results from all models in Task A and B will print/display one by one 
