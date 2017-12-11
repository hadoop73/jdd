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
xstr = 'lgb60_more'

#df_train = pd.read_csv('../datas/feas_new')
#df_train = df_train.drop('is_risk',axis=1)

df_train = pd.read_csv('../datas/feasb1')
#feas = feas.drop(['is_risk'],axis=1)
#df_train = df_train.merge(feas,on='rowkey',how='left')

feas = pd.read_csv('../datas/all3.csv')
acl = ['id','time','is_risk']
feas = feas.drop(acl,axis=1)

tis = [60, 2 * 60, 30 * 60, 60 * 60, 24 * 60 * 60, 2 * 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60,
        60 * 24 * 60 * 60]

for ti in tis:
    acl += ['trade_{0}_hour'.format(ti),'trade_{0}_cnt'.format(ti),'logintrade_{0}_rate'.format(ti),
    'iptrade_{0}_rate'.format(ti),'trade_time_diff_{0}_max'.format(ti),'trade_time_diff_{0}_min'.format(ti),
    'trade_time_diff_{0}_mean'.format(ti),'trade_{0}_cnt{1}'.format(ti, 60),'trade_{0}_cnt{1}'.format(ti, 180),
    'trade_{0}_cnt{1}'.format(ti, 300),'trade_{0}_cnt{1}'.format(ti, 900),'trade_{0}_cnt{1}'.format(ti, 901)]


#feas = feas.drop(acl,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')


feas = pd.read_csv('../datas/feasb2')
ack = [i for i in ['is_risk','id','time','hour'] if i in feas.columns]
feas = feas.drop(ack,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')

acl = [ci for ci in df_train.columns if 'timestamp_online' in ci]
acl = acl + [ci for ci in df_train.columns if 'timestamp' in ci]

acl = acl + ['trade_stamp']#,'hour','id','weekday' ,'city0','city1','city2','log_from0','log_from1','log_from2']

print(acl)
#df_train = df_train.drop(acl,axis=1)

feas = pd.read_csv('../datas/feas_login_new31')
cl = ['hour','is_risk','time','id']
#
feas = feas.drop(cl,axis=1)

# feas = pd.read_csv('../datas/feas_new_more')
# ack = [i for i in ['is_risk','id','time','hour'] if i in feas.columns]
# feas = feas.drop(ack,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')




test = df_train[df_train['time']>='2015-07-01 00:00:00']
dtest = test[['is_risk','rowkey','time','id']]

test = test.drop('time',axis=1)

df_train = df_train[df_train['time']<'2015-07-01 00:00:00']

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
# 构造训练集
#df_train = df_train[df_train['time']>='2015-05-01 00:00:00']
df_train = df_train[df_train['time']<'2015-06-01 00:00:00']#[df_train['time']>='2015-05-01 00:00:00']

df_train = df_train.drop('time',axis=1)
dfval = df_val[['is_risk','rowkey','time','id']]
df_val = df_val.drop('time',axis=1)

y_train = df_train['is_risk'].values
y_val = df_val['is_risk'].values

#y_test = df_test['is_risk'].values
dftrain = df_train[['is_risk','rowkey']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

#test['is_risk'] = -1
X_test = test.drop(['is_risk','rowkey'], axis=1).values

params = {
    #'max_depth':8,
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc', # auc
    'num_leaves': 81,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.6,    # 随机选 90% 的特征用于训练
    'bagging_fraction': 0.9,    # 随机选择 80% 的数据进行 bagging
    'bagging_freq': 50,          # bagging 每 5 次进行
    'max_bin':80,              # 特征最大分割
    'min_data_in_leaf':20,      # 每个叶子节点最少样本
    'verbose': 0
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

lgb_test = lgb.Dataset(X_test)

evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,
                lgb_train,
                feature_name=df_train.columns.tolist(),
                num_boost_round=95,                                # 80
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
yi = 1
while dfval.loc[n]['is_risk'] == 1:
    yi = dfval.loc[n]['y']
    n += 1

print(df_train.shape)
print("top n: ",n)
print('test:',dtest[dtest['y']>=yi].shape[0])

ans = pd.read_csv('../datas/ans')
ans = ans[ans['p']==1]

dfval.rename(columns={'y':'y1'},inplace=True)
ans = ans.merge(dfval[['rowkey','y1','is_risk','id','time']],on='rowkey',how='left')
ans['sum'] = ans['y1']

ans = ans.sort_values('sum',ascending=False).reset_index(drop=True)
ans.to_csv('../datas/improve',index=None)
print ans.head(50)
print ans.loc[50:100]

cols = df_train.columns.tolist()
scores = gbm.feature_importance()
df = pd.DataFrame({'cols':cols,'scores':scores})
df = df.sort_values('scores',ascending=False).reset_index(drop=True)

df.head(300).to_csv('../datas/a',index=None,header=None)

