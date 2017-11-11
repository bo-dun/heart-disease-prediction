from read_data import *
from keras.models import Sequential
from keras.layers import Dense

X_train, X_test, y_train, y_test = split_data()

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[0]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.T, y_train.T, epochs=100)

training_score = model.evaluate(X_train.T, y_train.T)
testing_score = model.evaluate(X_test.T, y_test.T)

print(training_score)
print(testing_score)