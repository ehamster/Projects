#****************************
# Auxiliary, Convolutional and LSTM
# for Customer
# Attrition/Churn Prediction
# by Yi Wu
# Nov 28, 2017
#****************************


#****import modules****
import h5py
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, accuracy_score
from sklearn import cross_validation
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.layers import Input, concatenate
from keras.models import Sequential, load_model, Model
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import regularizers


from utility import *
#**********************


#****import data****

DATA_PATH = '../data/'

X, y, id = load_data(DATA_PATH + 'd_train.h5')
X = np.transpose(X, (1, 2, 0))

#X = normal_X(X)


feature = ['is_auto_renew_mean', 'is_cancel_mean', 'is_auto_renew', 'is_cancel', \
          'registered_via', 'payment_method_id', 'city', 'payment_plan_days', \
          'plan_list_price', 'month', 'year', 'bd']

#feature = ['is_auto_renew_mean', 'is_cancel_mean', 'is_auto_renew', 'is_cancel']
aux, y_aux, id = load_aux_data(DATA_PATH + 's_train.h5', feature)
print(aux.shape)
print(np.sum(y - y_aux))
print 'loaded done'


idx_pos = np.where(y == 1)[0]
idx_neg = np.where(y == 0)[0]
idx_neg = idx_neg[0:idx_pos.shape[0]]
idx = np.concatenate((idx_pos, idx_neg))

X = X[idx,:]
y = y[idx]
aux = aux[idx,:]
id = id[idx]


#split data into train, valid, test
ratio1 = 0.5
ratio2 = 0.5

X_train, X_valid_test, y_train, y_valid_test = cross_validation.train_test_split(X, y,
                                                                test_size=ratio1, random_state=0,stratify=y)
X_valid, X_test, y_valid, y_test = cross_validation.train_test_split(X_valid_test, y_valid_test,
                                                                test_size=ratio2, random_state=0, stratify=y_valid_test)

aux_train, aux_valid_test, _, _ = cross_validation.train_test_split(aux, y,
                                                                test_size=ratio1, random_state=0,stratify=y)
aux_valid, aux_test, _, _ = cross_validation.train_test_split(aux_valid_test, y_valid_test,
                                                                test_size=ratio2, random_state=0, stratify=y_valid_test)

print_id_label('train', id, y_train)
print_id_label('valid', id, y_valid)
print_id_label('test', id, y_test)
#**********************


#****model building****

#input layers
main_input1 = Input(shape=(X_train.shape[1], 1), name='main_input1')
main_input2 = Input(shape=(X_train.shape[1], 1), name='main_input2')
main_input3 = Input(shape=(X_train.shape[1], 1), name='main_input3')
main_input4 = Input(shape=(X_train.shape[1], 1), name='main_input4')
main_input5 = Input(shape=(X_train.shape[1], 1), name='main_input5')
main_input6 = Input(shape=(X_train.shape[1], 1), name='main_input6')
main_input7 = Input(shape=(X_train.shape[1], 1), name='main_input7')

aux_input = Input(shape=(aux_train.shape[1],), name='aux_input')


#main paths
main_path1 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input1)
main_path1 = BatchNormalization()(main_path1)
main_path1 = Activation('relu')(main_path1)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path1 = Dropout(0.2)(main_path1)
main_path1 = LSTM(64, recurrent_dropout=0.5, return_sequences=True)(main_path1)
main_path1 = Flatten()(main_path1)


main_path2 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input2)
main_path2 = BatchNormalization()(main_path2)
main_path2 = Activation('relu')(main_path2)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path2 = Dropout(0.2)(main_path2)
main_path2 = Flatten()(main_path2)

main_path3 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input3)
main_path3 = BatchNormalization()(main_path3)
main_path3 = Activation('relu')(main_path3)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path3 = Dropout(0.2)(main_path3)
main_path3 = Flatten()(main_path3)


main_path4 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input4)
main_path4 = BatchNormalization()(main_path4)
main_path4 = Activation('relu')(main_path4)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path4 = Dropout(0.2)(main_path4)
main_path4 = Flatten()(main_path4)

main_path5 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input5)
main_path5 = BatchNormalization()(main_path5)
main_path5 = Activation('relu')(main_path5)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path5 = Dropout(0.2)(main_path5)
main_path5 = Flatten()(main_path5)


main_path6 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input6)
main_path6 = BatchNormalization()(main_path6)
main_path6 = Activation('relu')(main_path6)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path6 = Dropout(0.2)(main_path6)
main_path6 = Flatten()(main_path6)


main_path7 = Conv1D(filters=128, kernel_size=2, strides = 1, padding='valid')(main_input7)
main_path7 = BatchNormalization()(main_path7)
main_path7 = Activation('relu')(main_path7)
#main_path = MaxPooling1D(pool_size=2)(main_path)
main_path7 = Dropout(0.2)(main_path7)

main_path7 = Flatten()(main_path7)


main_path = concatenate([main_path1, main_path2, main_path3, main_path4, main_path5, main_path6, main_path7])


#aux path
aux_path = Dense(units=100, kernel_initializer='he_normal')(aux_input)
aux_path = BatchNormalization()(aux_path)
aux_path = Activation('relu')(aux_path)
aux_path = Dropout(0.5)(aux_path)

#merge paths
merge_path = concatenate([main_path, aux_path])

#stack deep densely-connected networks on top 
merge_path = Dense(units=100, kernel_initializer='he_normal')(merge_path)
merge_path = BatchNormalization()(merge_path)
merge_path = Activation('relu')(merge_path)
merge_path = Dropout(0.5)(merge_path)

merge_path = Dense(units=100, kernel_initializer='he_normal')(merge_path)
merge_path = BatchNormalization()(merge_path)
merge_path = Activation('relu')(merge_path)
merge_path = Dropout(0.5)(merge_path)

merge_output = Dense(units=1, activation='sigmoid', name='main_output')(merge_path)
model = Model(inputs = [main_input1, main_input2, main_input3, main_input4, \
               main_input5, main_input6, main_input7, aux_input], outputs = merge_output)
#*******************************


#****model compiling****
adam = Adam(lr=0.01, decay=1e-03)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='2_aux_conv_lstm.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#***********************


#****model fit****
print(model.summary())

History = model.fit([X_train[:,:, 0:1], X_train[:,:, 1:2], X_train[:,:, 2:3], \
                    X_train[:,:, 3:4], X_train[:,:, 4:5], X_train[:,:, 5:6], X_train[:,:, 6:7], aux_train], y_train, \
                    batch_size=300, epochs=500, shuffle=True, \
                    validation_data = ([X_valid[:, :, 0:1], X_valid[:, :, 1:2], X_valid[:, :, 2:3], \
		    X_valid[:, :, 3:4], X_valid[:, :, 4:5], X_valid[:, :, 5:6], X_valid[:, :, 6:7], aux_valid], y_valid), \
                    initial_epoch = 0, verbose=2, callbacks=[checkpointer,earlystopper])

save_history('history_1', History)
#**********************



#****model evaluation****
model = load_model('2_aux_conv_lstm.hdf5')
prob_pred = model.predict([X_test[:, :, 0:1],  X_test[:, :, 1:2], X_test[:, :, 2:3], X_test[:, :, 3:4], \
                          X_test[:, :, 4:5], X_test[:, :, 5:6], X_test[:, :, 6:7], aux_test], verbose=1, batch_size = 1000)
class_pred = (prob_pred > 0.5).astype(int)


auc_ROC = roc_auc_score(y_test, prob_pred)
mcc = matthews_corrcoef(y_test, class_pred)
prfs = precision_recall_fscore_support(y_test, class_pred)
acc = accuracy_score(y_test, class_pred)
auc_PR = average_precision_score(y_test, prob_pred, average="micro")


print(tabulate([['auc@roc', auc_ROC], \
                ['mcc', mcc], \
                ['precision', prfs[0][1]], \
                ['recall', prfs[1][1]], \
                ['f1 score', prfs[2][1]],\
                ['support', prfs[3][1]],\
                ['accuray', acc], \
                ['auc@pr', auc_PR]]))
#************************


