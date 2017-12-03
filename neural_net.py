from read_data import *
from keras.models import Sequential
from keras.layers import Dense

X_train, X_test, y_train, y_test = split_data()

model = Sequential()

model.add(Dense(units=64,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu',
                input_dim=X_train.shape[0]))
model.add(Dense(units=32,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu'))
model.add(Dense(units=16,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu'))
model.add(Dense(units=1,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.T, y_train.T, epochs=1000)

training_score = model.evaluate(X_train.T, y_train.T)
print(training_score)

testing_score = model.evaluate(X_test.T, y_test.T)
print(testing_score)