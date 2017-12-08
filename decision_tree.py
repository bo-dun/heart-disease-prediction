import sklearn.ensemble
from sklearn.model_selection import GridSearchCV
from data_entry import *
import numpy as np

# STRATIFY BY DIABETICS
# print(test_frame)
# X_train, X_test, Y_train, Y_test = split_data()

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
print(np.asarray(y_train_set[0]))
clf.fit(x_train_set[0], y_train_set[0])
counter = 0
total = 0

prediction1 = clf.predict(x_dev_set[0])#X_test.T)
answer = y_dev_test[0].ravel()#Y_test.ravel()
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
gbc.fit(x_train_set[0], y_train_set[0].ravel())#X_train.T, Y_train.ravel())
counter = 0
total = 0
prediction2 = gbc.predict(x_dev_set[0])
for i in range(len(prediction2)):
    total = total + 1
    if (answer[i] == prediction2[i]):
        counter = counter + 1

print("GRADIENT BOOSTED TREE STATS")
print('Predicted Correctly: ' + str(counter))
print('Total: ' + str(total))
print('Accuracy: ' + str(counter/total))
print('Best Params: {}'.format(gbc.best_params_))
