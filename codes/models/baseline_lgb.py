# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle

"""
learning_rate = 0.1
num_leaves = 255
num_trees = 500
num_threads = 16
min_data_in_leaf = 0
min_sum_hessian_in_leaf = 100
"""

# TODO 调参(lgb,xgb,adboost)，正则，融合
# TODO 更换训练数据
df_train = pd.read_csv('../../datas/baseline')
# 构造测试集
#df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
# 构造训练集
#df_train = df_train[df_train['time']<'2015-06-01 00:00:00'][df_train['time']>='2015-04-10 00:00:00']

df_val = df_train[df_train['rowkey']>=801668]
df_train = df_train[df_train['rowkey']<801668][df_train['rowkey']>=633047]


#df_train = df_train.drop('time',axis=1)
#df_val = df_val.drop('time',axis=1)

#test = pd.read_csv('../datas/baseline_feas_test')
#test = pd.read_csv('../../datas/login_trade_test')
#test = test.drop('time',axis=1)


# data_view 33 cell，正样本 203 个
#df_test = df_train[df_train['rowkey']>=736366]
#df_train = df_train[df_train['rowkey']<736366]#[df_train['rowkey']>=633047]

y_train = df_train['is_risk'].values
y_val = df_val['is_risk'].values

#y_test = df_test['is_risk'].values
dftrain = df_train[['is_risk','rowkey']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
dfval = df_val[['is_risk','rowkey']]
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

#test['is_risk'] = -1
#dtest = test[['is_risk','rowkey']]
#X_test = test.drop(['is_risk','rowkey'], axis=1).values


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss,auc', # auc
    'num_leaves': 61,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

#lgb_test = lgb.Dataset(X_test)

evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,
                lgb_train,
                feature_name=df_train.columns.tolist(),
                num_boost_round=80,                                # 80
                evals_result=evals_result,
                learning_rates=lambda iter: 0.1 * (0.99 ** iter), # 0.1 替换为 0.08
                valid_sets=[lgb_train,lgb_val])
# 预测验证集
#ytest_pred = gbm.predict(X_test)
#dtest['y'] = ytest_pred
#dtest = dtest.sort_values('y',ascending=False).reset_index(drop=True)
#dtest.to_csv('../datas/lgbbase_ytest_pred',index=None)

# 预测验证集
yval_pred = gbm.predict(X_val)
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../datas/lgbbase_yval_pred',index=None)

# 预训练证集
ytrain_pred = gbm.predict(X_train)
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../datas/lgbbase_ytrain_pred',index=None)
#y_pred = gbm.predict(X_test)
#xtest['y'] = y_pred
#xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)

#print xtest.head()
#xtest.to_csv('../datas/xtest',index=None)
#print evals_result
lgb.plot_metric(evals_result,metric='auc')
#lgb.plot_metric(evals_result,metric='binary_logloss')
#lgb.plot_importance(gbm, max_num_features=20)
plt.show()
