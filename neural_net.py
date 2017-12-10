from data_entry import *
from keras.models import Model, Input
from keras.layers import Dense, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from scipy.stats import mode
import os

tb = TensorBoard(log_dir='./logs')
inputs = Input((x_train_set[0].shape[1], ))

early_stopper = EarlyStopping(patience=200)
X = Dense(units=64,
          kernel_initializer='glorot_uniform',
          kernel_regularizer=l2(0.0001),
          bias_initializer='zeros',
          activation='relu')(inputs)
X = BatchNormalization()(X)
X = Dense(units=32,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l2(0.0001),
          activation='relu')(X)
X = Dense(units=16,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l2(0.0001),
          activation='relu')(X)
X = Dense(units=1,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=l2(0.0001),
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
              callbacks=[tb, checkpointer, early_stopper])
    model.load_weights('./checkpoints/nn_init_weights.hdf5')

weights = []
predictions = []
for i in range(len(x_train_set)):
    model.load_weights('./checkpoints/nn_best_weights' + str(i) + '.hdf5')
    preds = model.predict(x_test)
    preds = 1 * (preds > 0.5)
    predictions.append(preds)

voting_preds = mode(predictions, axis=0)[0]
print("Test accuracy: " + str(np.mean((np.squeeze(voting_preds) - np.squeeze(y_test)) == 0)))
