# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# TODO 数据优化，调参，完善一下数据
#df_train = pd.read_csv('../../datas/login_trade_train')
#df_train = pd.read_csv('../../datas/login_trade_train')
df_train = pd.read_csv('../../datas/trade_login_all')  # 0.6 filter

#test = pd.read_csv('../datas/baseline_feas_test')
test = df_train[df_train['time']>='2015-07-01 00:00:00']
test = test.drop('time',axis=1)

df_train = df_train[df_train['time']<'2015-07-01 00:00:00']

#for i in [0,1,2]:
#    df_train['is_scan_{0}'.format(i + 1)] = df_train['is_scan_{0}'.format(i + 1)].map(lambda x: 1 if x else 0)
#    df_train['is_sec_{0}'.format(i + 1)] = df_train['is_sec_{0}'.format(i + 1)].map(lambda x: 1 if x else 0)

df_train.fillna(np.nan,inplace=True)

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
df_train = df_train[df_train['time']<'2015-06-01 00:00:00']
# 构造训练集
df_train = df_train[df_train['time']>='2015-05-01 00:00:00']

df_train = df_train.drop('time',axis=1)
df_val = df_val.drop('time',axis=1)


#test = pd.read_csv('../../datas/login_trade_test')
#test = pd.read_csv('../../datas/baseline_test')

#for i in [0,1,2]:
#    test['is_scan_{0}'.format(i + 1)] = test['is_scan_{0}'.format(i + 1)].map(lambda x: 1 if x else 0)
#    test['is_sec_{0}'.format(i + 1)] = test['is_sec_{0}'.format(i + 1)].map(lambda x: 1 if x else 0)

test.fillna(np.nan,inplace=True)

#test = test.drop('time',axis=1)

y_train = df_train['is_risk'].values
y_val = df_val['is_risk'].values

dftrain = df_train[['is_risk','rowkey']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
dfval = df_val[['is_risk','rowkey']]
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

xtest = test[['is_risk','rowkey']]
d_test = test.drop(['is_risk','rowkey'], axis=1)


import xgboost as xgb

dtrain = xgb.DMatrix(df_train,y_train,missing=np.nan)
dval = xgb.DMatrix(df_val,y_val,missing=np.nan)
dtest = xgb.DMatrix(d_test,missing=np.nan)

params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':2,
    'eval_metric':'auc',
    'gamma':0.1,
    'max_depth':8,
    'lambda':5,
    'subsample':0.7,
    'colsample_bytree':0.4,
    'eta':0.05,
    'seed':123,
    'nthreed':8
}

watchlist = [(dtrain,'train'),(dval,'valid')]
bst = xgb.train(params,dtrain,num_boost_round=50,evals=watchlist)
xstr = 'xgb_100_time'

# 预测验证集
ytest_pred = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
xtest['y'] = ytest_pred
xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)
xtest.to_csv('../../datas/{0}_ytest_pred'.format(xstr),index=None)

# 预测验证集
yval_pred = bst.predict(dval,ntree_limit=bst.best_ntree_limit)
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../../datas/{0}_yval_pred'.format(xstr),index=None)

# 预训练证集
ytrain_pred = bst.predict(dtrain,ntree_limit=bst.best_ntree_limit)
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../../datas/{0}_ytrain_pred'.format(xstr),index=None)

fpr, tpr, _ = roc_curve(y_train, ytrain_pred)

roc_auc = auc(fpr, tpr)
print 'train:',roc_auc

fpr, tpr, _ = roc_curve(y_val, yval_pred)

roc_auc = auc(fpr, tpr)
print 'valid:',roc_auc