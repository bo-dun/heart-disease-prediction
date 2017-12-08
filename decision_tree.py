import sklearn.ensemble
from sklearn.model_selection import GridSearchCV
from data_entry import *
import numpy as np

# STRATIFY BY DIABETICS
# print(test_frame)
# X_train, X_test, Y_train, Y_test = split_data()

kNum = len(x_dev_set)
max_total = 0
max_i = -1
for i in range(1,2):
    print(i)
    set_total = 0
    for set_num in range(kNum):
        clf = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=75, criterion='entropy', max_depth=11, min_samples_leaf=1)
# Performed iterative linear search for optimal hyperparameters
#parameters = {
#    'n_estimators': [50],#[94],
#    'criterion': ['gini'],
#    'max_depth': [11],
#    'min_samples_leaf': [1],
#    'max_features': ['sqrt']
#}
#clf = GridSearchCV(randomForest, parameters)
        clf.fit(x_train_set[set_num], y_train_set[set_num])
        
        counter = 0
        prediction1 = clf.predict(x_dev_set[set_num])
        answer = y_dev_set[set_num]
        for j in range(len(prediction1)):
            if (answer[j] == prediction1[j]):
                counter = counter + 1
        set_total = set_total + counter
        print(np.sort(clf.feature_importances_))
    if (set_total > max_total):
        max_i = i
        max_total = set_total
        

total = len(x_dev_set[0])
accuracy = max_total / kNum
print("RANDOM FOREST STATS")
print('Best Estimator: ' + str(max_i))
print('Average Predicted Correctly: ' + str(accuracy))
print('Total: ' + str(total))
print('Accuracy: ' + str(accuracy/total))
#print("Best Params: {}".format(clf.best_params_))







'''
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
gbc.fit(x_train_set[0], y_train_set[0].ravel())
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
'''
