from data_entry import *
from keras.models import Model, Input
from keras.layers import Dense, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os

tb = TensorBoard(log_dir='./logs')
inputs = Input((x_train_set[0].shape[1], ))

early_stopper = EarlyStopping(patience=100)

X = Dense(units=64,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l1_l2(0.003, 0.01),
          activation='relu')(inputs)
X = BatchNormalization()(X)
X = Dense(units=32,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l1_l2(0.003, 0.01),
          activation='relu')(X)
X = Dense(units=16,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l1_l2(0.003, 0.01),
          activation='relu')(X)
X = Dense(units=1,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l1_l2(0.003, 0.01),
          activation='sigmoid')(X)

model = Model(inputs=inputs, outputs=X)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.save_weights('./checkpoints/nn_init_weights.hdf5')

for i in range(len(x_train_set)):
    best_weights_filepath = os.path.join('./checkpoints', 'nn_best_weights' + str(i) + '.hdf5')
    checkpointer = ModelCheckpoint(filepath=best_weights_filepath,
                                   verbose=1,
                                   monitor='val_loss',
                                   save_best_only=True)
    model.fit(x_train_set[i],
              y_train_set[i],
              validation_data=(x_dev_set[i], y_dev_set[i]),
              epochs=1000,
              callbacks=[checkpointer, early_stopper])
    model.load_weights('./checkpoints/nn_init_weights.hdf5')

best_weights_filepath = os.path.join('./checkpoints', 'nn_final_weights' + '.hdf5')
checkpointer = ModelCheckpoint(filepath=best_weights_filepath,
                               verbose=1,
                               monitor='loss',
                               save_best_only=True)

model.fit(x_train_dev,
          y_train_dev,
          epochs=1000,
          callbacks=[tb, checkpointer, early_stopper])
model.load_weights(best_weights_filepath)
score = model.evaluate(x_test, y_test)
print(score)
print(1 * (model.predict(x_test) > 0.5))

def predictor(x):
    model.load_weights(best_weights_filepath)
    return 1 * (model.predict(x) > 0.5)
