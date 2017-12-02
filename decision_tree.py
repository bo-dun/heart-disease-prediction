import sklearn.ensemble
from sklearn.model_selection import GridSearchCV
from read_data import *
import numpy as np

# STRATIFY BY DIABETES

X_train, X_test, Y_train, Y_test = split_data()

randomForest = sklearn.ensemble.RandomForestClassifier(random_state=0)
# Performed iterative linear search for optimal hyperparameters
parameters = {
    'n_estimators': [46],
    'criterion': ['gini'],
    'max_depth': [11],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}
clf = GridSearchCV(randomForest, parameters)
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
print("Best Params: {}".format(clf.best_params_))

gradientTree = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
parameters = {
    'loss': ['deviance'],
    'n_estimators': [25],#range(1,100),#[50],
    'max_depth': [1],
    'criterion': ['friedman_mse'],#['friedman_mse', 'mse', 'mae'],
    'min_samples_leaf': [1],#range(1,10),
    'max_features': ['sqrt']#'auto', 'sqrt', 'log2', None]
}
gbc = GridSearchCV(gradientTree, parameters)
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
print('Best Params: {}'.format(gbc.best_params_))
