from sklearn.ensemble import RandomForestClassifier
from read_data import *
import numpy as np

#filename = "cleveland.data"

#to_int = np.vectorize(float)

#with open(filename, 'r', encoding='ascii') as f:
#    values = [word for line in f for word in line.split()]
#    data_processed = np.reshape(np.array(values), (-1, 76))

# exclude the predictor
#X = to_int(np.delete(data_processed, [58 - 1, 76-1], 1))
#Y = to_int(data_processed[:, 58 - 1])
X_train, X_test, Y_train, Y_test = split_data()
#print(X.shape, Y.shape)
# diabetic?
#d = data_processed[:, 17 - 1]

clf = RandomForestClassifier(max_depth=10, random_state=0)
#half_x = int(X.shape[0] / 2)
#clf.fit(X[:half_x], Y[:half_x])
clf.fit(np.transpose(X_train), Y_train.ravel())
counter = 0
total = 0

X_test = np.transpose(X_test)
Y_test = Y_test.ravel()
prediction = clf.predict(X_test)
answer = Y_test
for i in range(len(prediction)):
    total = total + 1
    if (answer[i] == prediction[i]):
        counter = counter + 1

print(counter)
print(total)
