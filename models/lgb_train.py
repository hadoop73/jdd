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
# TODO 更换训练数据，xgb 效果好，需要提升 lgb,adboost 的效果
# TODO 检查数据，完善一下数据
# TODO 模型融合改进
# TODO 用 month_login 数据进行训练,调参，筛数据
xstr = 'lgb60_month_'

df_train = pd.read_csv('../datas/all.csv')  # 0.6 filter
acl = [ci for ci in df_train.columns if 'timestamp_online' in ci]
print(acl)
df_train = df_train.drop(['trade_stamp']+acl,axis=1)
#df_train = df_train.fillna(0)
#df_train = df_train.drop(['hour','weekday'],axis=1)

#test = pd.read_csv('../datas/baseline_feas_test')
test = df_train[df_train['time']>='2015-07-01 00:00:00']
test = test.drop('time',axis=1)

df_train = df_train[df_train['time']<'2015-07-01 00:00:00']

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
# 构造训练集
#df_train = df_train[df_train['time']>='2015-05-01 00:00:00']
df_train = df_train[df_train['time']<'2015-06-01 00:00:00']#[df_train['time']>='2015-05-01 00:00:00']

df_train = df_train.drop('time',axis=1)
df_val = df_val.drop('time',axis=1)



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
dtest = test[['is_risk','rowkey']]
X_test = test.drop(['is_risk','rowkey'], axis=1).values

params = {
    #'max_depth':8,
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc', # auc
    'num_leaves': 61,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.9,    # 随机选 90% 的特征用于训练
    'bagging_fraction': 0.8,    # 随机选择 80% 的数据进行 bagging
    'bagging_freq': 5,          # bagging 没 5 次进行
    'max_bin':100,              # 特征最大分割
    'min_data_in_leaf':50,      # 每个叶子节点最少样本
    'verbose': 0
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

lgb_test = lgb.Dataset(X_test)

evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,
                lgb_train,
                feature_name=df_train.columns.tolist(),
                num_boost_round=70,                                # 80
                evals_result=evals_result,
                learning_rates=lambda iter: 0.1 * (0.99 ** iter), # 0.1 替换为 0.08
                valid_sets=[lgb_train,lgb_val])

# 预测验证集
ytest_pred = gbm.predict(X_test)
dtest['y'] = ytest_pred
dtest = dtest.sort_values('y',ascending=False).reset_index(drop=True)
dtest.to_csv('../datas/{0}_ytest_pred'.format(xstr),index=None)

# 预测验证集
yval_pred = gbm.predict(X_val)
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../datas/{0}_yval_pred'.format(xstr),index=None)

n = 0
while dfval.loc[n]['is_risk'] == 1:
    n += 1
print("top n: ",n)

# 预训练证集
ytrain_pred = gbm.predict(X_train)
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../datas/{0}_ytrain_pred'.format(xstr),index=None)
#y_pred = gbm.predict(X_test)
#xtest['y'] = y_pred
#xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)

#print xtest.head()
#xtest.to_csv('../datas/xtest',index=None)
#print evals_result
#lgb.plot_metric(evals_result,metric='auc')
#lgb.plot_metric(evals_result,metric='binary_logloss')
lgb.plot_importance(gbm, max_num_features=20)
graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree0')
graph.render(view=True)
plt.show()
