import numpy as np
import pandas as pd
import sklearn.model_selection as sk

fheader = "headernames.data"
fdata = [
         "cleveland.data",
         "hungarian.data",
         "long-beach-va.data",
         "switzerland.data",
        ]
good_features = [
  "age",
  "sex",
  "painloc",
  "painexer",
  "relrest",
  "pncaden",
  "cp",
  "trestbps",
  "htn",
  "chol",
  "smoke",
  "cigs",
  "years",
  "fbs",
  "dm",
  "famhist",
  "restecg",
  "dig",
  "prop",
  "nitr",
  "pro",
  "diuretic",
  "proto",
  "thaldur",
  "thaltime",
  "met",
  "thalach",
  "thalrest",
  "tpeakbps",
  "tpeakbpd",
  "dummy",
  "trestbpd",
  "exang",
  "xhypo",
  "oldpeak",
  "slope",
  "rldv5",
  "rldv5e",
  "restef",
  "restwm",
  "exeref",
  "exerwm",
  "thal"
]

with open(fheader, 'r', encoding='ascii') as f:
    headers = [word for line in f for word in line.split()]

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

# DON'T KEEP THE FOUR LABEL CATEGORIES
Y = frame['num'].map(lambda x: 1 if float(x) != 0 else 0)

frame = frame[good_features]
# Added in two features, one corresponding to sex and one to age.
# Both are aggregate features corresponding to percentage frequency
# of coronary heart disease
age_series = frame['age']
sex_series = frame['sex']

sex_feature = [(7.2 if x == "1" else 4.3) for x in sex_series]
age_feature = []
for age in age_series:
    num = int(age)
    if (num < 18):
        age_feature.append(0)
    elif (num < 45):
        age_feature.append(0.8)
    elif (num < 65):
        age_feature.append(6.1)
    elif (num < 75):
        age_feature.append(16.4)
    else:
        age_feature.append(23.3)
frame['age_feature'] = age_feature
frame['sex_feature'] = sex_feature

# convert to numbers
frame = frame.applymap(float)

# break down categorical variables
flatten = lambda x: 1 if x else 0
factors = ['cp', 'restecg', 'proto', 'slope', 'restwm', 'thal', 'dataset']

for factor in factors:
    if factor in frame:
        for value in frame[factor].unique():
            frame[factor + str(int(value))] = (frame[factor] == value).map(flatten)
        del frame[factor]

# impute -9 values
problematic_cols = [k for k in frame.keys() if any(frame[k] == -9)]

for k in frame.keys():
    if frame[k].isin([-9])[0]:
        ser = frame[k][frame[k] != -9]
        mean = 0
        if ser.shape[0] != 0:
            mean = ser.mean()
        frame[k] = frame[k].map(lambda x: mean if x == -9 else 0)
        
X = pd.DataFrame.as_matrix(frame)

x_train_dev, x_test, y_train_dev, y_test = sk.train_test_split(X, Y, test_size = 0.2, random_state = 0, stratify=Y)
kf = sk.StratifiedKFold(n_splits=5, random_state = 0, shuffle=True)
y_train_dev = np.asarray(y_train_dev)
x_train_dev = np.asarray(x_train_dev)
y_test = np.asarray(y_test)
x_test = np.asarray(x_test)

x_train_set = []
y_train_set = []
x_dev_set = []
y_dev_set = []
for train_index, dev_index in kf.split(x_train_dev, y_train_dev):
    x_train_set.append(x_train_dev[train_index])
    y_train_set.append(y_train_dev[train_index])
    x_dev_set.append(x_train_dev[dev_index])
    y_dev_set.append(y_train_dev[dev_index])

# 5 SETS OF K-FOLD SPLIT TRAINING DATA AND LABELS
x_train_set
y_train_set

# 5 CORRESPONDING SETS OF DEV DATA AND LABELS
x_dev_set
y_dev_set

# TESTING DATA AND LABELS
x_test
y_test

# Feature Names
keys = np.asarray(frame.keys())

print("Features:", frame.keys())
