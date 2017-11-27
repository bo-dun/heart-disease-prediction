import sklearn.ensemble
from read_data import *
import numpy as np

X_train, X_test, Y_train, Y_test = split_data()

clf = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(np.transpose(X_train), Y_train.ravel())
counter = 0
total = 0

X_test = np.transpose(X_test)
Y_test = Y_test.ravel()
prediction1 = clf.predict(X_test)
answer = Y_test
for i in range(len(prediction1)):
    total = total + 1
    if (answer[i] == prediction1[i]):
        counter = counter + 1

print("RANDOM FOREST STATS")
print('Predicted Correctly: ' + str(counter))
print('Total: ' + str(total))
print('Accuracy: ' + str(counter/total))

gbc = sklearn.ensemble.GradientBoostingClassifier()
gbc.fit(np.transpose(X_train), Y_train.ravel())
counter = 0
total = 0
prediction2 = gbc.predict(X_test)
for i in range(len(prediction2)):
    total = total + 1
    if (answer[i] == prediction2[i]):
        counter = counter + 1

print("GRADIENT BOOSTED TREE STATS")
print('Predicted Correctly: ' + str(counter))
print('Total: ' + str(total))
print('Accuracy: ' + str(counter/total))

