#****************************
# Logistic Regression
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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import pickle
from sklearn.metrics import log_loss

import os

from utility import *
#**********************


#****import data****
DATA_PATH = '../data/'


X, _, _ = load_data(DATA_PATH + 'd_train.h5')
X = np.transpose(X, (1,2,0))
print(X.shape)
X = np.mean(X, axis = 1)
print(X.shape)

#feature = ['is_auto_renew_mean', 'is_cancel_mean', 'is_auto_renew', 'is_cancel', \
#          'registered_via', 'payment_method_id', 'city', 'payment_plan_days', 'plan_list_price']

feature = ['is_auto_renew_mean', 'is_cancel_mean', 'is_auto_renew', 'is_cancel']
aux, y, id = load_aux_data(DATA_PATH + 's_train.h5', feature)
print 'loaded done'

#aux = np.concatenate((aux, X), axis = 1)
print(aux.shape)

idx_pos = np.where(y == 1)[0]
idx_neg = np.where(y == 0)[0]
idx_neg = idx_neg[0:idx_pos.shape[0]]
idx = np.concatenate((idx_pos, idx_neg))

"""y = y[idx]
aux = aux[idx,:]
id = id[idx]"""

#split data into train, valid, test
ratio1 = 0.5
ratio2 = 0.5



aux_train, aux_valid_test, y_train, y_valid_test = cross_validation.train_test_split(aux, y,
                                                                test_size=ratio1, random_state=0,stratify=y)
aux_valid, aux_test, y_valid, y_test = cross_validation.train_test_split(aux_valid_test, y_valid_test,
                                                                test_size=ratio2, random_state=0, stratify=y_valid_test)

print_id_label('train', id, y_train)
print_id_label('valid', id, y_valid)
print_id_label('test', id, y_test)
#**********************

print 'training LR model'
#clf = LogisticRegressionCV(Cs=10.0**-np.arange(-1,1), cv=3, penalty="l2", solver="sag", n_jobs=-1, max_iter=2000, verbose=1)
#LR_fit = clf.fit(aux_train, y_train)

clf = LogisticRegression(penalty='l2', C = 0.5)
LR_fit = clf.fit(aux_train, y_train)


filename = 'finalized_model.pkl'
pickle.dump(LR_fit, open(filename, 'wb'))#函数的功能：将obj对象序列化为string形式，而不是存入文件中

#****model evaluation****
LR_fit = pickle.load(open(filename, 'r')) #函数的功能：从string中读出序列化前的obj对象。
prob_pred =  LR_fit.predict_proba(aux_test)
prob_pred = prob_pred[:,1].flatten()
class_pred = (prob_pred > 0.5).astype(int)

print(type(prob_pred))
print(y_test.shape)
print(prob_pred.shape)
print(y_test)
print(prob_pred)

loss = log_loss(y_test, prob_pred)
auc_ROC = roc_auc_score(y_test, prob_pred)
mcc = matthews_corrcoef(y_test, class_pred)
prfs = precision_recall_fscore_support(y_test, class_pred)
acc = accuracy_score(y_test, class_pred)
auc_PR = average_precision_score(y_test, prob_pred, average="micro")


print(tabulate([['log_loss', loss], \
        ['auc@roc', auc_ROC], \
                ['mcc', mcc], \
                ['precision', prfs[0][1]], \
                ['recall', prfs[1][1]], \
                ['f1 score', prfs[2][1]],\
                ['support', prfs[3][1]],\
                ['accuray', acc], \
                ['auc@pr', auc_PR]]))
#************************

