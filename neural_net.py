from data_entry import *
from keras.models import Model, Input
from keras.layers import Dense

for i in range(len(y_train_set)):
    np.expand_dims(y_train_set[i], axis=1)

inputs = Input((x_train_set[0].shape[1], ))
X = Dense(units=64,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          activation='relu')(inputs)
X = Dense(units=1,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          activation='sigmoid')(X)

model = Model(inputs=inputs, outputs=X)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_set[0], y_train_set[0], epochs=10000, batch_size=32)
