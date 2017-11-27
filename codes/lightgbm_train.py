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


df_train = pd.read_csv('../datas/baseline')
# data_view 33 cell，正样本 203 个
df_test = df_train[df_train['rowkey']>=801668]
df_train = df_train[df_train['rowkey']<801668][df_train['rowkey']>=633047]

y_train = df_train['is_risk'].values
y_test = df_test['is_risk'].values
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
X_train = df_train.values
xtest = df_test[['is_risk','rowkey']]
X_test = df_test.drop(['is_risk','rowkey'], axis=1).values

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss,auc', # auc
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)

evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,
                lgb_train,
                feature_name=df_train.columns.tolist(),
                num_boost_round=150,
                evals_result=evals_result,
                valid_sets=[lgb_train,lgb_test])

y_pred = gbm.predict(X_test)
xtest['y'] = y_pred
xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)

print xtest.head()
xtest.to_csv('../datas/xtest',index=None)
#print evals_result
lgb.plot_metric(evals_result,metric='auc')
#lgb.plot_metric(evals_result,metric='binary_logloss')
#ax = lgb.plot_importance(gbm, max_num_features=20)
plt.show()
