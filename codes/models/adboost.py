# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# TODO 调参，完善一下数据

xstr = 'adboost_80_all3'
df_train = pd.read_csv('../../datas/all3.csv')
df_train = df_train.fillna(df_train.mean())
#df_train = pd.read_csv('../../datas/login_trade_train_filter') # 0.6 filter

df_train.fillna(0,inplace=True)

test = df_train[df_train['time']>='2015-07-01 00:00:00']

df_train = df_train[df_train['time']<'2015-07-01 00:00:00']

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
df_train = df_train[df_train['time']<'2015-06-01 00:00:00']
# 构造训练集
#df_train = df_train[df_train['time']>='2015-05-01 00:00:00']



# 构造测试集
# 构造训练集

df_train = df_train.drop('time',axis=1)
df_val = df_val.drop('time',axis=1)

#test = pd.read_csv('../../datas/login_trade_test_filter')
test.fillna(0,inplace=True)
test = test.drop('time',axis=1)

y_train = df_train['is_risk'].values
y_val = df_val['is_risk'].values

dftrain = df_train[['is_risk','rowkey']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
dfval = df_val[['is_risk','rowkey']]
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

dtest = test[['is_risk','rowkey']]
X_test = test.drop(['is_risk','rowkey'], axis=1).values

# 0.97760362301
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=150, learning_rate=0.01)
clf.fit(X_train,y_train)


# 预测验证集
ytest_pred = clf.predict_proba(X_test)[:,1]
dtest['y'] = ytest_pred
dtest = dtest.sort_values('y',ascending=False).reset_index(drop=True)
dtest.to_csv('../../datas/{0}_ytest_pred'.format(xstr),index=None)

# 预测验证集
yval_pred = clf.predict_proba(X_val)[:,1]
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../../datas/{0}_yval_pred'.format(xstr),index=None)

# 预训练证集
ytrain_pred = clf.predict_proba(X_train)[:,1]
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../../datas/{0}_ytrain_pred'.format(xstr),index=None)

fpr, tpr, _ = roc_curve(y_train, ytrain_pred)

roc_auc = auc(fpr, tpr)
print 'train:',roc_auc

fpr, tpr, _ = roc_curve(y_val, yval_pred)

roc_auc = auc(fpr, tpr)
print 'valid:',roc_auc
