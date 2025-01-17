from medmnist import BreastMNIST, BloodMNIST
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

sys.path.append('./A')
from CNN_A import neuralNet_A

sys.path.append('./B')
from CNN_B import neuralNet_B

# FUNCTION TO PRINT/DISPLAY METRICS FOR ML MODELS (CONFUSION MATRIX, CLASSIFICATION REPORT, ROC CURVE)
def metrics(model, y_test, y_pred, x_test, title):
    # CONFUSION MATRIX
    print(f'{title} - Confusion Matrix:')
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['malignant','benign'])
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f'{title} Confusion Matrix',fontsize=13, weight='bold', pad=10)
    plt.xlabel('Prediction', fontsize=11, weight='bold')
    plt.ylabel('Actual', fontsize=11, weight='bold')
    plt.show()


    # CLASSIFICATION REPORT
    print(f'{title} - Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['malignant','normal, benign']))

    # ROC-AUC CURVE
    print(f'{title} - ROC Curve:')
    y_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, 'darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} Receiver Operating Characteristic (ROC) Curve', weight='bold')
    plt.legend(loc='lower right')
    plt.show()

# FUNCTION TO PRINT/DISPLAY METRICS FOR CNN MODELS (CONFUSION MATRIX, CLASSIFICATION REPORT, ROC CURVE)
def metrics_cnn(y_true, y_score, y_score_prob, title, labels, display_labels, target_names):
    # CONFUSION MATRIX
    print(f'{title} - Confusion Matrix:')
    conf_matrix = confusion_matrix(y_true, y_score, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f'{title} Confusion Matrix',fontsize=13, weight='bold', pad=10)
    plt.xlabel('Prediction', fontsize=11, weight='bold')
    plt.ylabel('Actual', fontsize=11, weight='bold')
    plt.show()

    # PRINTING METRICS (PRECISION, RECALL, F1-SCORE)
    print(f'{title} - Classification Report:')
    print(classification_report(y_true, y_score, target_names=target_names)) #as string

    if title == "Task A CNN":
        # PLOTTING ROC-AUC
        print(f'{title} - ROC Curve:')
        fpr, tpr, thresholds = roc_curve(y_true, y_score_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, 'darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} Receiver Operating Characteristic (ROC) Curve', weight='bold')
        plt.legend(loc='lower right')
        plt.show()


def svm_model(x_train, y_train, x_test, y_test):
    # STANDARISATION
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # MODEL TRAINING
    model = svm.SVC(probability=True, C=100, gamma="scale", kernel="rbf", random_state=10)
    model.fit(x_train, y_train)

    # PREDICTION + ACCURACY
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM Test Accuracy: {test_accuracy}')
    title = "SVM"
    metrics(model, y_test, y_pred, x_test, title)

def svm_hog_model(x_train, y_train, x_test, y_test):
    # FLATTEN TO 1D
    y_train_svc = y_train.ravel()
    y_test_svc = y_test.ravel()

    # Histogram of Oriented Gradients (HOG) PERFORMED ON IMAGES
    x_train_svc = [hog(image,  orientations=9, 
                pixels_per_cell=(8,8), 
                cells_per_block=(2,2), 
                block_norm='L2')
                for image in x_train]

    x_test_svc = [hog(image,  orientations=9, 
            pixels_per_cell=(8,8), 
            cells_per_block=(2,2), 
            block_norm='L2')
            for image in x_test]
    
    # STANDARISATION
    scaler = StandardScaler()
    x_train_svc = scaler.fit_transform(x_train_svc)
    x_test_svc = scaler.transform(x_test_svc)

    # MODEL TRAINING
    model = svm.SVC(probability=True, C=1, gamma='auto', kernel='rbf', random_state=10)
    model.fit(x_train_svc, y_train_svc)

    # PREDICTION + ACCURACY
    y_pred = model.predict(x_test_svc)
    test_accuracy = accuracy_score(y_test_svc, y_pred)
    print(f'\n\n\nSVM with HOG Test Accuracy: {test_accuracy}')
    title = "SVM with HOG"
    metrics(model, y_test_svc, y_pred, x_test_svc, title)

def random_forest_model(x_train, y_train, x_test, y_test):
    y_train_rf = y_train.ravel()
    y_test_rf = y_test.ravel()

    # HOG PERFORMED ON IMAGES
    x_train_rf = [hog(image,  orientations=9, 
                pixels_per_cell=(8,8), 
                cells_per_block=(2,2), 
                block_norm='L2')
                for image in x_train]

    x_test_rf = [hog(image,  orientations=9, 
                pixels_per_cell=(8,8), 
                cells_per_block=(2,2), 
                block_norm='L2')
                for image in x_test]

    # STANDARISATION
    scaler = StandardScaler()
    x_train_rf = scaler.fit_transform(x_train_rf)
    x_test_rf = scaler.transform(x_test_rf)
    
    # MODEL TRAINING
    model = RandomForestClassifier(random_state=10,
                                   n_estimators=200, 
                                   max_depth=20, 
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_features='log2')
    model.fit(x_train_rf, y_train_rf)

    # PREDICTION + ACCURACY
    y_pred = model.predict(x_test_rf)
    test_accuracy = accuracy_score(y_pred, y_test_rf)
    print(f'\n\nRandom Forest with HOG Test Accuracy: {test_accuracy}')
    title = "Random Forest"
    metrics(model, y_test_rf, y_pred, x_test_rf, title)

def logistic_regression_model(x_train, y_train, x_test, y_test):
    # STANDARISATION
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # MODEL TRAINING
    model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', class_weight=None, max_iter=1000)
    model.fit(x_train, y_train)

    # PREDICTION + ACCURACY
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n\n\nLogistic Regresssion Test Accuracy: {accuracy}')
    title = "Logistic Regression"
    metrics(model, y_test, y_pred, x_test, title)

    
def task_a_cnn(task_a_testSet):

    # INITIALISE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 25
    labels=[0, 1]
    display_labels=['malignant','benign']
    target_names=['0', '1']
    test_loader = data.DataLoader(dataset=task_a_testSet, batch_size=2*batch_size, shuffle=False)

    # LOAD MODEL AND WEIGHTS
    model = neuralNet_A(input_channels=1, no_classes=2).to(device)

    model_path = './A/cnn.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # TESTING
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    y_score_prob = torch.tensor([], device=device)
    
    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            pred = torch.sigmoid(outputs)
            rounded_pred = pred.round()

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, rounded_pred), 0)

            y_score_prob = torch.cat((y_score_prob, pred), 0)

        y_true = y_true.cpu().detach().numpy()
        y_score = y_score.cpu().detach().numpy()
        y_score_prob = y_score_prob.cpu().detach().numpy()
        
        acc = accuracy_score(y_true, y_score)

        print(f'\n\n\nTask A CNN - Test Accuracy: {acc}')
        title = 'Task A CNN'
        metrics_cnn(y_true, y_score, y_score_prob, title, labels, display_labels, target_names)


def task_b_cnn(task_b_testSet):
    # INITIALISE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 25
    labels=[0, 1, 2, 3, 4, 5, 6, 7]
    display_labels=['0', '1', '2', '3', '4', '5', '6', '7']
    target_names=['0', '1', '2', '3', '4', '5', '6', '7']
    test_loader = data.DataLoader(dataset=task_b_testSet, batch_size=2*batch_size, shuffle=False)

    # LOAD MODEL AND WEIGHTS
    model = neuralNet_B(input_channels=3, no_classes=8).to(device)

    model_path = './B/cnn.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # TESTING
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    y_score_prob = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            pred = torch.argmax(outputs, dim=1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, pred), 0)

            y_score_prob = torch.cat((y_score_prob, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().numpy()
        y_score_prob = y_score_prob.cpu().numpy()
        
        acc = accuracy_score(y_true, y_score)

        print(f'\n\n\nTask B CNN (Custom) - Test Accuracy: {acc}')
        title = 'Task B CNN (Custom)'
        metrics_cnn(y_true, y_score, y_score_prob, title, labels, display_labels, target_names)

def task_b_resnet50(task_b_testSet):
    # INITIALISE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 25
    labels=[0, 1, 2, 3, 4, 5, 6, 7]
    display_labels=['0', '1', '2', '3', '4', '5', '6', '7']
    target_names=['0', '1', '2', '3', '4', '5', '6', '7']
    test_loader = data.DataLoader(dataset=task_b_testSet, batch_size=2*batch_size, shuffle=False)

    # MODEL
    resnet50 = models.resnet50(pretrained=True)

    num_classes = 8
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    model = resnet50.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # LOAD MODEL AND WEIGHTS
    model_path = './B/resnet50.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # TESTING
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    y_score_prob = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            pred = torch.argmax(outputs, dim=1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, pred), 0)

            y_score_prob = torch.cat((y_score_prob, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().numpy()
        y_score_prob = y_score_prob.cpu().numpy()
        
        acc = accuracy_score(y_true, y_score)

        print(f'\n\nCNN (ResNet50) Accuracy: {acc}')
        title = 'Task B CNN (ResNet50)'
        metrics_cnn(y_true, y_score, y_score_prob, title, labels, display_labels, target_names)

        
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # IMPORT IMAGES
    trainSet = BreastMNIST(split="train", download="True")
    valSet = BreastMNIST(split="val", download="True")
    testSet = BreastMNIST(split="test", download="True")

    x_train = np.concatenate((trainSet.imgs, valSet.imgs), axis=0)
    y_train = np.concatenate((trainSet.labels, valSet.labels), axis=0)
    x_test = testSet.imgs
    y_test = testSet.labels

    x_train_raw = x_train.reshape(x_train.shape[0], -1)
    y_train_raw = y_train.reshape(y_train.shape[0], -1).ravel()
    x_test_raw = x_test.reshape(x_test.shape[0], -1)
    y_test_raw = y_test.reshape(y_test.shape[0], -1).ravel()
    
    # AUGMENT IMAGES FOR CNN
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # important
        transforms.RandomRotation(10),
        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformation
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    task_a_testSet = BreastMNIST(split='test', transform=data_transform, download="True")
    task_b_testSet = BloodMNIST(split='test', transform=data_transform, download="True")

    print("\n\n***********TASK A***********")

    svm_model(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    svm_hog_model(x_train, y_train, x_test, y_test)
    random_forest_model(x_train, y_train, x_test, y_test)
    logistic_regression_model(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    task_a_cnn(task_a_testSet)

    
    print("\n\n\n\n\n***********TASK B***********")
    task_b_cnn(task_b_testSet)
    task_b_resnet50(task_b_testSet)


if __name__ == '__main__':
    main()