import torchvision.models as models
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
torch.cuda.manual_seed_all(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


# RESNET 50 - FREEZING EARLIER LAYERS
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

criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #+

# INITIALISE FOR EARLY STOPPING
best_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())
patience = 5
wait = 0

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
    
torch.save(model.state_dict(), 'resnet50.pth')

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
conf_matrix = confusion_matrix(y_true, y_score, labels=[0, 1, 2, 3, 4, 5, 6, 7])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['0', '1', '2', '3', '4', '5', '6', '7'])
disp.plot(cmap=plt.cm.Oranges)
plt.title('Confusion Matrix',fontsize=13, weight='bold', pad=10)
plt.xlabel('Prediction', fontsize=11, weight='bold')
plt.ylabel('Actual', fontsize=11, weight='bold')
#plt.xticks(rotation=70)
plt.show()

# PRINTING METRICS (PRECISION, RECALL, F1-SCORE, AUC-ROC)
print(classification_report(y_true, y_score, target_names=['0', '1', '2', '3', '4', '5', '6', '7'])) #as string

# more layer unfreeze better results