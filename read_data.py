import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def read_data(filename="cleveland.data"):
    to_int = np.vectorize(float)

    with open(filename, 'r', encoding='ascii') as f:
        values = [word for line in f for word in line.split()]
        data_processed = np.reshape(np.array(values), (-1, 76))

    # exclude the predictor
    X = to_int(np.delete(data_processed, [58 - 1, 76-1], 1))
    Y = to_int(data_processed[:, 58 - 1])

    # diabetic?
    d = data_processed[:, 17 - 1]
    return X.T, np.reshape(Y, (Y.shape[0], 1)).T


def filter_data(filename="cleveland.data"):
    X, y = read_data(filename)
    y = np.where(y > 0, 1, 0)

    selections = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50]  # the 14 used attributes
    X = X[selections, :]

    # add more attributes

    X = normalize(X, axis=0)

    return X, y


def split_data():
    X, y = filter_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, stratify=y.T, random_state = 0
    )
    return X_train.T, X_test.T, y_train.T, y_test.T
