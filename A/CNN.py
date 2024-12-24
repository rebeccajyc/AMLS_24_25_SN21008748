from medmnist import BreastMNIST
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


lr = 0.001
no_epochs = 100
batch_size = 50

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

trainSet = BreastMNIST(split='train', transform=data_transform, download="True")
valSet = BreastMNIST(split='val', transform=data_transform, download="True")
testSet = BreastMNIST(split='test', transform=data_transform, download="True")

train_loader = data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dataset=valSet, batch_size=2*batch_size, shuffle=False)
test_loader = data.DataLoader(dataset=testSet, batch_size=2*batch_size, shuffle=False)

class neuralNet(nn.Module):
    def __init__(self, input_channels, no_classes):
        super(neuralNet, self).__init__()

    
        self.conv1 = nn.Conv2d(input_channels, 16 , kernel_size=3)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.batchNorm2 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 64, kernel_size=3)
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.conv4 =  nn.Conv2d(64, 64, kernel_size=3)
        self.batchNorm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(64)

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, no_classes))
        
        
    def forward(self,x):
       
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)

        x = self.dropout(x)

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

        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = neuralNet(input_channels=1, no_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,eps=1e-08)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

trainAccList = []
valAccList = []
testAccList = []

trainLossList = []
valLossList = []

for epoch in range(no_epochs):
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
    
        
model.eval()
y_true = torch.tensor([], device=device)
y_score = torch.tensor([], device=device)

with torch.no_grad():
    for inputs, targets in test_loader:

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        targets = targets.squeeze().long()
        outputs = outputs.softmax(dim=-1)
        pred = torch.argmax(outputs, dim=1)

        y_true = torch.cat((y_true, targets), 0)
        y_score = torch.cat((y_score, pred), 0)

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().numpy()
    
    acc = accuracy_score(y_true, y_score)

    testAccList.append(acc)

    print(f'Accuracy: {acc}')

epochList = np.arange(no_epochs)
plt.plot(epochList, trainAccList, 'r', label='train acc')
plt.plot(epochList, valAccList, 'b', label='val acc')
plt.legend(loc="upper left")
plt.show()

plt.plot(epochList, trainLossList, 'r', label='train loss')
plt.plot(epochList, valLossList, 'b', label='val loss')
plt.legend(loc="upper left")
plt.show()

# lr
# batch size
# dropout (input) (inbetween layers - acc drops)
# optimizer
# .to(device) .cpu()
