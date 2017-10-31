from sklearn.ensemble import RandomForestClassifier

import numpy as np

filename = "cleveland.data"

to_int = np.vectorize(float)

with open(filename, 'r', encoding='ascii') as f:
        values = [word for line in f for word in line.split()]
        data_processed = np.reshape(np.array(values), (-1, 76))

# exclude the predictor
X = to_int(np.delete(data_processed, [58 - 1, 76-1], 1))
Y = to_int(data_processed[:, 58 - 1])

# diabetic?
d = data_processed[:, 17 - 1]

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X[:X.shape[0] - 1], Y[:Y.shape[0] - 1])
print(clf.predict([X[X.shape[0] - 2]]))
print(Y[Y.shape[0] - 2])
