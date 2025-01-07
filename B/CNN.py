from medmnist import BloodMNIST
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,  ExponentialLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

# SEEDED FOR REPRODUCEABILITY AND DEBUGGING
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# GPU - DEVICE AVAILABILITY
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# INITIALISE HYPERPARAMETERS
lr = 0.001
no_epochs = 50
batch_size = 150
weight_decay = 1e-4

# DATA AUGMENTATION
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformation
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# DOWNLOAD DATASETS
trainSet = BloodMNIST(split='train', transform=data_transform, download="True")
valSet = BloodMNIST(split='val', transform=data_transform, download="True")
testSet = BloodMNIST(split='test', transform=data_transform, download="True")

train_loader = data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dataset=valSet, batch_size=2*batch_size, shuffle=False)
test_loader = data.DataLoader(dataset=testSet, batch_size=2*batch_size, shuffle=False)

# CLASS IMBALANCE - CLASS WEIGHTS
class_weights = class_weight.compute_class_weight('balanced',
                                                    classes = np.unique(trainSet.labels[:,0]),
                                                    y = trainSet.labels[:, 0])
class_weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

# CNN MODEL
class neuralNet(nn.Module):
    def __init__(self, input_channels, no_classes):
        super(neuralNet, self).__init__()

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
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(256)


        self.fc = nn.Sequential(
            #nn.Linear(128 * 3 * 3, 128),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, no_classes))
        
        
    def forward(self,x):
       
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)


        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = neuralNet(input_channels=3, no_classes=8).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,eps=1e-08)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        optimizer.zero_grad()

        outputs = model(inputs)
        targets = targets.squeeze().long()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        totalTrainLoss += loss.item()

        trainPred = torch.argmax(outputs, dim=1)
        train_y_true = torch.cat((train_y_true, targets), 0)
        train_y_score = torch.cat((train_y_score, trainPred), 0)

    train_y_true = train_y_true.cpu().numpy()
    train_y_score = train_y_score.cpu().numpy()

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
            
            outputs = model(inputs)
            targets = targets.squeeze().long()
            valLoss = criterion(outputs, targets)

            totalValLoss += valLoss.item()
            valPred = torch.argmax(outputs, dim=1)

            val_y_true = torch.cat((val_y_true, targets), 0)
            val_y_score = torch.cat((val_y_score, valPred), 0)
        

        val_y_true = val_y_true.cpu().numpy()
        val_y_score = val_y_score.cpu().numpy()

        valAcc = accuracy_score(val_y_true, val_y_score)
        valAccList.append(valAcc)

        avgValLoss = totalValLoss/len(val_loader)
        valLossList.append(avgValLoss)

        print(f'Epoch:{epoch}, Validation Loss: {avgValLoss}, Accuracy: {valAcc}')

        # EARLY STOPPING
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
    
        
model.eval()
y_true = torch.tensor([], device=device)
y_score = torch.tensor([], device=device)
y_score_prob = torch.tensor([], device=device)

# TESTING
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

    testAccList.append(acc)

    print(f'Accuracy: {acc}')

# PLOTTING ACCURACY GRAPH
epochList = np.arange(no_epochs)
plt.plot(epochList, trainAccList, 'r', label='train accuracy')
plt.plot(epochList, valAccList, 'b', label='validation accuracy')
plt.legend(loc="upper left")
plt.show()

# PLOTTING LOSS GRAPH
plt.plot(epochList, trainLossList, 'r', label='train loss')
plt.plot(epochList, valLossList, 'b', label='validation loss')
plt.legend(loc="upper left")
plt.show()

# PLOTTING CONFUSION MATRIX
conf_matrix = confusion_matrix(y_true, y_score, labels=[0, 1, 2, 3, 4, 5, 6, 7])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3, 4, 5, 6, 7])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix',fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.show()

# PRINTING METRICS (PRECISION, RECALL, F1-SCORE, AUC-ROC)
print(classification_report(y_true, y_score, target_names=['0', '1', '2', '3', '4', '5', '6', '7'])) #as string


# lr
# batch size
# dropout (input) (inbetween layers - acc drops) -> 0.1-0.3 for conv layers, 0.5 connected
# optimizer
# .to(device) .cpu()
# early stopping -> 10 not strong enough overfits despite stopping -> decrease
# seeded for reproduceability/ debugging -> test across other seeds
# transforms
# adam+
# high batch size -> overfitting
# weight_decay