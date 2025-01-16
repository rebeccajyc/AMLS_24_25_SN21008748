from medmnist import BreastMNIST
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,  ExponentialLR, CosineAnnealingLR
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

# MODEL
class neuralNet_A(nn.Module):
    def __init__(self, input_channels, no_classes):
        super(neuralNet_A, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16 , kernel_size=3)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.conv4 =  nn.Conv2d(64, 128, kernel_size=3)
        self.batchNorm4 = nn.BatchNorm2d(128)

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1))
        
    def forward(self,x):
       
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    # SEEDED FOR REPRODUCEABILITY AND DEBUGGING
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # GPU - DEVICE AVAILABILITY
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # INITIALISE HYPERPARAMETERS
    lr = 0.001
    no_epochs = 100
    batch_size = 25

    # DATA AUGMENTATION
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # important
        transforms.RandomRotation(10),
        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformation
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # DOWNLOAD DATASETS
    trainSet = BreastMNIST(split='train', transform=data_transform, download="True")
    valSet = BreastMNIST(split='val', transform=data_transform, download="True")
    testSet = BreastMNIST(split='test', transform=data_transform, download="True")

    train_loader = data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=valSet, batch_size=2*batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=testSet, batch_size=2*batch_size, shuffle=False)

    # CLASS IMBALANCE - CLASS WEIGHTS
    class_weights = class_weight.compute_class_weight('balanced',
                                                        classes = np.unique(trainSet.labels[:,0]),
                                                        y = trainSet.labels[:, 0])
    weights = { 0 : class_weights[0], 1 : class_weights[1] }
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)



    model = neuralNet_A(input_channels=1, no_classes=2).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # SIGMOID FOR BINARY CLASSIFICATION

    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,eps=1e-08)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #+
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    #scheduler = CosineAnnealingLR(optimizer, T_max=20)

    # INITIALISE FOR EARLY STOPPING
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 5
    wait = 0

    # INITIALISE METRICS
    trainAccList = []
    valAccList = []
    testAccList = []

    trainLossList = []
    valLossList = []

    for epoch in range(no_epochs):
        # TRAINING
        totalTrainLoss = 0
        
        train_y_true = torch.tensor([], device=device)
        train_y_score = torch.tensor([], device=device)

        model.train()

        for inputs, targets in tqdm(train_loader):
        
            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.view(-1, 1).float() #

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss.item()

            trainPred = torch.sigmoid(outputs).round() ###
            train_y_true = torch.cat((train_y_true, targets), 0)
            train_y_score = torch.cat((train_y_score, trainPred), 0)

        train_y_true = train_y_true.cpu().detach().numpy() ##detach
        train_y_score = train_y_score.cpu().detach().numpy()

        trainAcc = accuracy_score(train_y_true, train_y_score)
        trainAccList.append(trainAcc)

        avgTrainLoss = totalTrainLoss/len(train_loader)
        trainLossList.append(avgTrainLoss)

        print(f'Epoch:{epoch}, Train Loss: {avgTrainLoss}, Accuracy: {trainAcc}')
        
        # VALIDATION
        with torch.no_grad():
            model.eval()
            totalValLoss = 0
            
            val_y_true = torch.tensor([], device=device)
            val_y_score = torch.tensor([], device=device)
        
            for inputs, targets in val_loader:

                inputs, targets = inputs.to(device), targets.to(device)

                targets = targets.view(-1, 1).float() #
                
                outputs = model(inputs)
                
                valLoss = criterion(outputs, targets)
                totalValLoss += valLoss.item()

                valPred = torch.sigmoid(outputs).round()
                val_y_true = torch.cat((val_y_true, targets), 0)
                val_y_score = torch.cat((val_y_score, valPred), 0)
            
            val_y_true = val_y_true.cpu().detach().numpy()
            val_y_score = val_y_score.cpu().detach().numpy()

            valAcc = accuracy_score(val_y_true, val_y_score)
            valAccList.append(valAcc)

            avgValLoss = totalValLoss/len(val_loader)
            valLossList.append(avgValLoss)

            print(f'Epoch:{epoch}, Validation Loss: {avgValLoss}, Accuracy: {valAcc}')

            # EARLY STOPPPING
            if avgValLoss < best_loss:
                    best_loss = avgValLoss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    wait = 0
            else:
                wait += 1
                print(f"No improvement in validation loss for {wait} epoch(s).")

            if wait >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
        
        scheduler.step(avgValLoss) # reducelronplateau
        #scheduler.step()

    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'cnn.pth')
            
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    y_score_prob = torch.tensor([], device=device)

    # TESTING
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
        testAccList.append(acc)

        print(f'Test Accuracy: {acc}')


    # PLOTTING ACCURACY GRAPH
    epochList = np.arange(no_epochs)
    plt.plot(epochList, trainAccList, 'r', label='Train Accuracy')
    plt.plot(epochList, valAccList, 'b', label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=11, weight='bold')
    plt.ylabel('Accuracy', fontsize=11, weight='bold')
    plt.legend(loc="upper left")
    plt.show()  

    # PLOTTING LOSS GRAPH
    plt.plot(epochList, trainLossList, 'r', label='Train Loss')
    plt.plot(epochList, valLossList, 'b', label='Validation Loss')
    plt.xlabel('Epoch', fontsize=11, weight='bold')
    plt.ylabel('Loss', fontsize=11, weight='bold')
    plt.legend(loc="upper left")
    plt.show()

    # PLOTTING CONFUSION MATRIX
    conf_matrix = confusion_matrix(y_true, y_score, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['malignant','benign'])
    disp.plot(cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix',fontsize=13, weight='bold', pad=10)
    plt.xlabel('Prediction', fontsize=11, weight='bold')
    plt.ylabel('Actual', fontsize=11, weight='bold')
    plt.show()

    # PRINTING METRICS (PRECISION, RECALL, F1-SCORE)
    print(classification_report(y_true, y_score, target_names=['0', '1'])) #as string

    # PLOTTING ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, 'darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', weight='bold')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()

