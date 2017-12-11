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
xstr = 'lgb60_month2'

feas = pd.read_csv('../datas/feas_login_new3')
cl = ['hour','time','id','is_risk']

feas = feas.drop(cl,axis=1)

cl = ['city_cnt120','city_cnt120rate','city_cnt172800','city_cnt172800rate','city_cnt1800','city_cnt1800rate','city_cnt300','city_cnt300rate','city_cnt3600','city_cnt3600rate','city_cnt60','city_cnt600','city_cnt600rate','city_cnt60rate','device_id_15','device_id_2','device_id_30','device_id_5','device_id_7','device_ip_id_15','device_ip_id_2','device_ip_id_30','device_ip_id_5','device_ip_id_7','ip_id_15','ip_id_2','ip_id_30','ip_id_5','ip_id_7','login2_time_diff','login_120_cnt120','login_120_cnt180','login_120_cnt60','login_172800_cnt120','login_172800_cnt180','login_172800_cnt60','login_1800_cnt120','login_1800_cnt180','login_1800_cnt60','login_300_cnt120','login_300_cnt180','login_300_cnt60','login_3600_cnt120','login_3600_cnt180','login_3600_cnt60','login_600_cnt120','login_600_cnt180','login_600_cnt60','login_60_cnt120','login_60_cnt180','login_60_cnt60','result31_1','result_31_and_1cnt120','result_31_and_1cnt172800','result_31_and_1cnt1800','result_31_and_1cnt300','result_31_and_1cnt3600','result_31_and_1cnt60','result_31_and_1cnt600','rowkey','timelong','trade_120_cnt','trade_15_cnt_pre','trade_1800_cnt','trade_2_cnt_pre','trade_300_cnt','trade_30_cnt_pre','trade_3600_cnt','trade_5_cnt_pre','trade_600_cnt','trade_60_cnt','trade_7_cnt_pre','trade_86400_cnt','trade_login_hour_cnt','trade_login_time_diff','trade_login_time_diff1','trade_login_time_diff2']
#feas = feas[cl]

df_train = pd.read_csv('../datas/feas_month_3_7')



#feas3['login_type3_hour_cnt2'] = feas3['login_type3_hour_cnt']*feas3['trade_hour_cnt']
#feas3 = feas3.drop(['time','id'],axis=1)
#feas3 = feas3.drop(['login_type3_result1','login_type3_hour_cnt'],axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')

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
dftrain = df_train[['is_risk','rowkey','id']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
dfval = df_val[['is_risk','rowkey','id']]
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

#test['is_risk'] = -1
dtest = test[['is_risk','rowkey','id']]
X_test = test.drop(['is_risk','rowkey'], axis=1).values

params = {
    #'max_depth':8,
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc', # auc
    'num_leaves': 81,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.6,    # 随机选 90% 的特征用于训练
    'bagging_fraction': 0.8,    # 随机选择 80% 的数据进行 bagging
    'bagging_freq': 50,         # bagging 每 5 次进行
    'max_bin':100,              # 特征最大分割
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
                num_boost_round=100,                                # 80
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
# lgb.plot_metric(evals_result,metric='auc')
# # # #lgb.plot_metric(evals_result,metric='binary_logloss')
#lgb.plot_importance(gbm, max_num_features=50)
# # #
# # # graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree0')
# # # graph.render(view=True)
#plt.show()
