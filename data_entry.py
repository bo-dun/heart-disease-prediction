import numpy as np

fheader = "headernames.data"
fdata = [
         "cleveland.data",
         "hungarian.data",
         "long-beach-va.data",
         "switzerland.data",
        ]

with open(fheader, 'r', encoding='ascii') as f:
    headers = [word for line in f for word in line.split()]

len(headers)

import pandas as pd

data_processed = []

for i in range(len(fdata)):
    with open(fdata[i], 'r', encoding='ascii') as f:
        values = [word for line in f for word in line.split()]
        data_processed.append(np.reshape(np.array(values), (-1, 76)))
    n = np.shape(data_processed[i])[0]
    data_processed[i] = np.column_stack((np.repeat(i, n), data_processed[i]))

for d in data_processed:
    print(np.shape(d))

data_processed = np.vstack(data_processed)

frame = pd.DataFrame.from_records(data_processed, columns=np.array(['dataset'] + headers))
frame

# delete names
del frame['name']

# convert to numbers
frame = frame.applymap(float)
# responses
Y = frame['num'].map(lambda x: 1 if x != 0 else 0)

# delete "garbage columns" that only have one value for the healthy
# as they probably just describe the response

for k in frame.keys():
    if frame[k][Y == 0].nunique() == 1:
        print('deleted ' + k)
        del frame[k]

# break down categorical variables
flatten = lambda x: 1 if x else 0
factors = ['cp', 'restecg', 'proto', 'slope', 'restwm', 'thal', 'dataset']

for factor in factors:
    if factor in frame:
        for value in frame[factor].unique():
            frame[factor + str(int(value))] = (frame[factor] == value).map(flatten)
        del frame[factor]

# create new column to indicate whether a variable is -9

problematic_cols = [k for k in frame.keys() if any(frame[k] == -9)]

for k in frame.keys():
    if frame[k].isin([-9])[0]:
        frame[k + '_invalid'] = frame[k].map(lambda x: 1 if x == -9 else 0)

frame.to_csv('./229_processed_cleveland_full.data')
Y.to_csv('./229_processed_cleveland_Y_full.data')

N = len(Y)
test_i = np.random.permutation(np.arange(N))[0:(N//6)]
train_i = np.random.permutation(np.arange(N))[(N//6):]

test_frame = frame[frame.index.isin(test_i)]
train_frame = frame[frame.index.isin(train_i)]

Y_test_frame = Y[frame.index.isin(test_i)]
Y_train_frame = Y[frame.index.isin(train_i)]

test_frame.to_csv('./new_cleveland_test_X.data')
train_frame.to_csv('./new_cleveland_train_X.data')

Y_test_frame.to_csv('./new_cleveland_test_Y.data')
Y_train_frame.to_csv('./new_cleveland_train_Y.data')
