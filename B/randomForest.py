from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import numpy as np
from medmnist import BloodMNIST
import time

trainSet = BloodMNIST(split="train", download="True")
valSet = BloodMNIST(split="val", download="True")
testSet = BloodMNIST(split="test", download="True")


# RF
x_train = np.array([np.array(trainSet[i][0]).flatten() for i in range(len(trainSet))])
y_train = np.array([trainSet[i][1].flatten() for i in range(len(trainSet))]).ravel()

x_test = np.array([np.array(testSet[i][0]).flatten() for i in range(len(testSet))])
y_test = np.array([testSet[i][1].flatten() for i in range(len(testSet))]).ravel()

startTime = time.time()

rf_classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=5)
rf_classifier.fit(x_train, y_train)

endTime = time.time()

print(f'rf TIme:{endTime-startTime}')

# HOG + RF
x_train_hog = []
y_train_hog = []

for image,label in trainSet:
    image = np.mean(image, axis=2) # greyscaling the image for HOG
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    x_train_hog.append(hog_features)
    y_train_hog.append(label)

x_test_hog = []
y_test_hog = []

for image,label in testSet:
    image = np.mean(image, axis=2)
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    x_test_hog.append(hog_features)
    y_test_hog.append(label)
 
rf_classifier_hog = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=5)
rf_classifier_hog.fit(x_train_hog, np.array(y_train_hog).ravel()) # 1D vector

# PREDICT
y_pred = rf_classifier.predict(x_test)
y_pred_hog = rf_classifier_hog.predict(x_test_hog)

accuracy = accuracy_score(y_pred, y_test)
accuracy_hog = accuracy_score(y_pred_hog, y_test_hog)

print(accuracy)
print(accuracy_hog)