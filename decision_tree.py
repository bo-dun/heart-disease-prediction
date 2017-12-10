import sklearn.ensemble
from sklearn.model_selection import GridSearchCV
from data_entry_peter import *
from sklearn.metrics import confusion_matrix
import numpy as np

# STRATIFY BY DIABETICS
# print(test_frame)
# X_train, X_test, Y_train, Y_test = split_data()

kNum = len(x_dev_set)
numFeatures = len(keys)
max_total = 0
max_i = -1
feature_importances = 0
for i in ['entropy']:
    print(i)
    set_total = 0
    for set_num in range(kNum):
        clf = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=73, criterion='entropy', max_depth=4, min_samples_leaf=12)
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
        
        feature_importances = feature_importances + clf.feature_importances_
        #important = clf.feature_importances_.argsort()[-55:][::-1]
        #print(keys[important])
        #print(clf.feature_importances_[important])
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
important = feature_importances.argsort()[-numFeatures:][::-1]
print(keys[important])
print(important)

optimal_dev_x = []
optimal_train_x = []
max_i = -1
max_total = 0
for i in range(1,numFeatures):
    print(i)
    
    new_dev_x = []
    new_dev_y = []
    new_train_x = []
    new_train_y = []
    print(important[:i])
    for j in range(kNum):
        new_dev_x.append(x_dev_set[j][:,important[:i]])#np.delete(arr=x_dev_set[j], obj=important[:i], axis=1))
        new_train_x.append(x_train_set[j][:,important[:i]])#np.delete(arr=x_train_set[j], obj=important[:i], axis=1))

    set_total = 0
    for set_num in range(kNum):
        clf = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=73, criterion='entropy', max_depth=4, min_samples_leaf=12)
        clf.fit(new_train_x[set_num], y_train_set[set_num])

        counter = 0
        prediction1 = clf.predict(new_dev_x[set_num])
        answer = y_dev_set[set_num]
        for j in range(len(prediction1)):
            if (answer[j] == prediction1[j]):
                counter = counter + 1
        set_total = set_total + counter

    if (set_total > max_total):
        max_i = i
        max_total = set_total
        optimal_dev_x = new_dev_x
        optimal_train_x = new_train_x
    print(set_total/kNum/total)

accuracy = max_total / kNum
print("RANDOM FOREST STATS")
print('Optimal Number of Features: ' + str(max_i))
print('Average Predicted Correctly: ' + str(accuracy))
print('Total: ' + str(total))
print('Accuracy: ' + str(accuracy/total))



max_total = 0
max_i2 = -1
feature_importances = 0
for i in range(1):
    print(i)
    set_total = 0
    for set_num in range(kNum):
        clf = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=70, criterion='entropy', max_depth=4, min_samples_leaf=12)
# Performed iterative linear search for optimal hyperparameters
#parameters = {
#    'n_estimators': [50],#[94],
#    'criterion': ['gini'],
#    'max_depth': [11],
#    'min_samples_leaf': [1],
#    'max_features': ['sqrt']
#}
#clf = GridSearchCV(randomForest, parameters)
        clf.fit(optimal_train_x[set_num], y_train_set[set_num])

        counter = 0
        prediction1 = clf.predict(optimal_dev_x[set_num])
        answer = y_dev_set[set_num]
        for j in range(len(prediction1)):
            if (answer[j] == prediction1[j]):
                counter = counter + 1
        set_total = set_total + counter

        feature_importances = feature_importances + clf.feature_importances_
        #important = clf.feature_importances_.argsort()[-55:][::-1]
        #print(keys[important])
        #print(clf.feature_importances_[important])
    if (set_total > max_total):
        max_i2 = i
        max_total = set_total

total = len(x_dev_set[0])
accuracy = max_total / kNum
print("RANDOM FOREST STATS")
print('Best Estimator: ' + str(max_i2))
print('Average Predicted Correctly: ' + str(accuracy))
print('Total: ' + str(total))
print('Accuracy: ' + str(accuracy/total))




predictor = sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=70, criterion='entropy', max_depth=4, min_samples_leaf=12)
predictor.fit(x_train_dev[:,important[:max_i]], y_train_dev)
total = 0
counter = 0
prediction1 = predictor.predict(x_test[:,important[:max_i]])
for j in range(len(prediction1)):
    total = total + 1
    if (y_test[j] == prediction1[j]):
        counter = counter + 1
    else:
        print("Error: " + str(j))
print('Accuracy: ' + str(counter/total))
print(confusion_matrix(y_test, prediction1))
print(prediction1)

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
