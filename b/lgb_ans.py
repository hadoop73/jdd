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
xstr = 'lgb60_mk'

#df_train = pd.read_csv('../datas/feas_new')
#df_train = df_train.drop('is_risk',axis=1)

df_train = pd.read_csv('../datas/feasb1')
#feas = feas.drop(['is_risk'],axis=1)
#df_train = df_train.merge(feas,on='rowkey',how='left')

feas = pd.read_csv('../datas/all3.csv')
acl = ['rowkey','hour','id_diff10','trade_weekday_cnt_rate','trade_cnt','idtype_diff10','idtype_time_min0','hour_v_cnt_rate','idtype_time_min1','idcity_diff10','id_diff20','idtype_diff20','id_time_min0','iddevice_diff20','weekday','id_diff30','id_time_max0','id_diff32','idtype_city0','id_time_min2','ip_time_min0','iddevice_diff10','idtype_diff30','idcity_time_max0','idtype_time_mean0','ip_diff10','idcity_time_mean0','idip_diff10','ip_time_max0','idcity_time_min1','device_time_min0','id_time_mean0','iptype_diff10','idcity_diff32','ip_time_mean0','id_city0','id_cnt_1rate0','id_diff11','devicetype_time_min1','idcity_cnt_rate0','idcity_diff20','device_diff10','device_diff11','idcity_time_min0','idtype_diff11','idtype_time_max0','idtype_diff32','iddevice_time_min0','devicetype_time_min0','hour_v_cnt','iptype_time_mean0','id_time_min1','hour_v','idtype_diff22','id_diff31','iddevice_diff30','idtype_time_min2','devicetype_time_mean2','idcity_time_mean1','idcity_time_max2','iddevice_time_max0','device_diff32','idcity_time_min2','idcity_diff11','idtype_diff31','device_time_max0','iddevice_diff21','iptype_time_max0','iptype_time_min0','id_diff22','device_diff20','ip_diff11','device_diff12','id_cnt_rate0','idcity_cnt_00','idcity_time_max1','iddevice_diff32','idtype_diff21','idtype_time_max2','devicetype_diff11','idcity_time_mean2','idcity_diff21','devicetype_diff10','idtype_cnt_1rate0','iptype_diff11','idip_time_mean0','device_id1','idip_time_min0','iptype_id0','ip_id0','device_time_mean0','id_diff21','id_time_mean1','id_cnt_1rate2','iddevice_time_min1','device_time_min1','device_id2','id_diff12','idcity_diff22','idtype_diff12','trade_weekday_cnt','ip_time_min1','iddevice_diff11','idtype_time_max1','device_cnt_1rate1','device_diff22','ipdevice_diff10','devicetype_time_min2','idtype_cnt_rate0','id_time_max2','idcity_cnt_rate1','iddevice_time_mean1','device_time_max1','ip_time_mean2','devicetype_id1','iddevice_city0','idcity_diff12','iddevice_time_mean0','device_cnt_scan0rate2','iddevice_time_mean2','devicetype_cnt_1rate2','device_time_mean1','idip_time_max0','iptype_time_min1','iptype_id1','idtype_cnt_1rate2','devicetype_time_mean0','id_cnt2','devicetype_diff21','iddevice_cnt_1rate2','idtype_city1','device_cnt_1rate2','idtype_cnt_1rate1','iddevice_diff22','device_time_mean2','id_city2','ip_diff30','idip_diff20','devicetype_time_mean1','idip_diff11','idtype_time_mean2','iddevice_diff31','iddevice_time_max1','id_time_max1','iddevice_cnt_1rate0','idip_time_mean1','ip_time_mean1','idtype_time_mean1','id_cnt_1rate1','idcity_diff30','devicetype_cnt_scan0rate1','ip_diff20','devicetype_diff20','devicetype_diff22','idcity_cnt_1rate0','ip_diff12','iddevice_time_min2','devicetype_id2','ip_diff32','idtype_city2','idtype_cnt0','ip_time_min2','devicetype_diff31','idip_cnt_00','devicetype_time_max0','device_diff21','ip_time_max2','id_cnt_00','idcity_cnt_02','ipdevice_time_max2','idip_time_min1','idcity_cnt_1rate1','idip_time_max1','id_ip0','id_time_mean2','idcity_cnt_10','idip_diff30','iddevice_diff12','device_diff30','devicetype_diff30','ipdevice_time_mean0','idtype_device1','id_device0','idtype_cnt_10','idcity_cnt0','devicetype_cnt_1rate0','device_diff31','devicetype_diff12','devicetype_cnt_02','idtype_cnt1','devicetype_cnt_rate0','iddevice_cnt0','idcity_cnt_12','iptype_time_mean1','device_cnt_rate2','ip_diff21','devicetype_diff32','ipdevice_diff12','devicetype_cnt_1rate1','devicetype_ip2']
# acl += [ci for ci in df_train.columns if 'timestamp_online' in ci]
# acl = acl + [ci for ci in df_train.columns if 'timestamp' in ci]
# #alist = ['iptype_cnt_rate2','idip_cnt_scan0rate2','idip_cnt_rate2','devicetype_cnt_scan00','idtype_ip0','iptype_cnt_1rate2','ip_cnt_scan10','iddevice_cnt_00','iddevice_cnt_11','ipdevice_cnt_12','idcity_device1','idtype_device1','ipdevice_city1','iptype_cnt_00','idcity_type1','id_cnt_scan11','ip_cnt_scan01','id_cnt_scan12','idip_cnt1','ipdevice_cnt_scan01','idip_cnt_11','idcity_cnt_scan12','idip_city2','device_from2','device_from0','ip_cnt_sec01','iddevice_cnt_rate0','idip_cnt_1rate1','idcity_from2','idcity_from0','iptype_cnt_scan00','idip_cnt_02','iddevice_type2','iddevice_type1','iptype_cnt_1rate1','idcity_type2','idip_cnt_scan00','iddevice_cnt_scan0rate2','iddevice_cnt_02','idip_cnt_scan02','idcity_cnt_sec00','idcity_cnt_sec01','ipdevice_cnt_02','ip_cnt_scan0rate2','ip_cnt_scan0rate1','iddevice_from2','idip_device1','idip_from2','idip_from0','ipdevice_diff21','ipdevice_diff22','iptype_cnt0','iptype_cnt1','idip_cnt_1rate2','iptype_cnt_scan0rate0','ipdevice_diff30','iddevice_ip2','idtype_ip1','devicetype_cnt_scan11','iddevice_cnt_10','ipdevice_cnt_11','id_from0','id_from1','ipdevice_city0','ip_cnt_12','ipdevice_time_mean1','idtype_cnt_scan02','ipdevice_diff11','iddevice_cnt1','device_ip1','idcity_device2','idip_cnt0','ipdevice_cnt_scan02','devicetype_from1','idip_cnt_12','devicetype_cnt_01','idtype_cnt_scan01','idip_cnt_1rate0','iddevice_cnt_rate2','idcity_from1','devicetype_ip0','iptype_cnt_scan02','id_from2','ip_device2','ip_city0','ipdevice_cnt2','iddevice_cnt_01','idip_cnt_scan01','iddevice_diff31','iddevice_from0','ip_cnt_02','ip_cnt_00','idtype_cnt_scan0rate1','idip_from1','ipdevice_diff31','iddevice_ip0','device_cnt_sec02','ipdevice_cnt_rate2','ip_cnt_scan11','idip_cnt_scan12','idcity_cnt_11','id_device1','idtype_cnt_scan00','idcity_cnt_scan00','device_cnt_scan02','device_cnt_scan00','ip_from1','id_cnt_scan0rate1','idcity_ip1','iptype_cnt_rate1','id_cnt_sec00','idtype_cnt_00','id_ip1','idtype_cnt1','ip_cnt_1rate2','iddevice_cnt_scan00','id_cnt_scan01','devicetype_ip1','ip_type0','idcity_cnt_rate1','ipdevice_cnt0','ipdevice_cnt_00','ipdevice_diff20','id_cnt_scan0rate2','iddevice_cnt_1rate0','iddevice_cnt_1rate1','iddevice_cnt_sec00','idtype_from2','ip_cnt2','iptype_diff32','idtype_cnt_sec02','ip_cnt_11','ipdevice_diff10','iddevice_cnt2','id_cnt_scan10','idcity_cnt_scan0rate1','devicetype_from2','ip_cnt_scan12','idip_city1','id_cnt_sec02','idtype_cnt0','iddevice_cnt_rate1','iddevice_cnt_scan02','ipdevice_diff32','ipdevice_time_mean2','devicetype_cnt1','idtype_cnt_10','iptype_device1','idcity_cnt_01','iptype_cnt_rate0','iddevice_ip1','iddevice_cnt_12','iptype_diff30','ip_cnt_10','idcity_cnt_scan02','ipdevice_diff12','idtype_cnt_rate2','device_cnt_rate1','idip_diff30','iptype_time_mean2','idip_diff20','ip_cnt_scan00','idcity_ip0','iptype_cnt_scan12','idip_cnt_10','ip_cnt_1rate1','idip_diff31','idip_diff32','devicetype_cnt0','ip_time_mean2','ipdevice_time_max2','devicetype_city0','ip_cnt_01','id_cnt_scan00','id_cnt_rate1','devicetype_cnt_1rate0','iddevice_cnt_1rate2','ip_time_max1','idcity_cnt_12','id_device2','idip_diff12','idcity_cnt_scan01','ip_cnt_rate1','idtype_cnt_rate0','idtype_cnt_rate1','iptype_cnt_10','ip_id2','devicetype_cnt_00','idtype_cnt_1rate2','idcity_cnt_1rate0','device_cnt_scan0rate0','idtype_cnt_11','devicetype_cnt_1rate1','idtype_ip2','devicetype_cnt_scan12','device_type2','device_ip0','ip_cnt_rate2','ipdevice_id1','iptype_time_mean1','idcity_cnt_scan0rate2','iptype_time_min2','idtype_cnt_01','device_cnt_scan10','iptype_time_max2','device_cnt_00','devicetype_diff30','id_cnt_rate2','idcity_cnt_rate2','ip_diff12','ip_cnt1','idtype_cnt_12','ip_device1','idip_time_mean2','idcity_cnt_10','id_type0','id_type1','idtype_device2','id_cnt_scan0rate0','idip_diff22','idcity_cnt_scan0rate0','idcity_cnt_scan10','id_ip2','ipdevice_time_min2','ip_cnt0','devicetype_city1','idip_time_max1','idtype_from0','iddevice_diff12','iptype_diff31','device_cnt_scan01','idcity_ip2','iptype_diff20','idcity_cnt_scan11','devicetype_cnt_scan0rate2','id_ip0','ipdevice_time_max1']
# #acl = acl + alist
#
# acl = acl + ['trade_stamp']#,'hour','id','weekday' ,'city0','city1','city2','log_from0','log_from1','log_from2']

print(acl)
feas = feas[acl]

#feas = feas.drop(acl,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')


feas = pd.read_csv('../datas/feasb2')
ack = [i for i in ['is_risk','id','time','hour'] if i in feas.columns]
feas = feas.drop(ack,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')

#df_train = df_train.drop(acl,axis=1)

feas = pd.read_csv('../datas/feas_login_new31')
cl = ['hour','is_risk','time','id']
#
feas = feas.drop(cl,axis=1)

# feas = pd.read_csv('../datas/feas_new_more')
# ack = [i for i in ['is_risk','id','time','hour'] if i in feas.columns]
# feas = feas.drop(ack,axis=1)
df_train = df_train.merge(feas,on='rowkey',how='left')


feas = pd.read_csv('../datas/feas_month2')
cls = ['rowkey','id_diff2','id_5184000_max','device_diff2','id_diff3','ip_diff2','id_604800_max','id_1296000_max','trade_5184000_hour','trade_5184000_cnt','ip_diff3','id_86400_max','device_diff3','id_5184000_cnt','trade_1296000_hour','trade_30day_cnt','idtype_1296000_3cnt','id_city5184000','idtype_5184000_3cnt','trade_86400_hour','trade_1800_hour','idresult_5184000_-2cnt','id_1800_max','idresult_5184000_1cnt','trade_60_hour','idlog_5184000_1cnt','ip_5184000_max','ip_86400_max','device_5184000_max','id_172800_max','idresult_5184000_31cnt','idlog_86400_1cnt','trade_60_cnt','idtype_604800_3cnt','device_604800_max','device_86400_max','idtype_86400_3cnt','trade_604800_hour','id_ip5184000','ip_1800_max','idtype_5184000_1cnt','idlog_5184000_2cnt','id_device5184000','idlog_604800_1cnt','idlog_1296000_2cnt','trade_604800_cnt','idtype_604800_1cnt','ip_1296000_max','ip_5184000_cnt','device_1296000_max']
feas = feas[cls]
print(feas.shape)

df_train = df_train.merge(feas,on='rowkey')



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

ans_test = pd.read_csv('../datas/ans_test')
ans_test = ans_test[ans_test['p']==1]
ans_test = ans_test.merge(dtest,on='rowkey',how='left')
print 'ans test: ',ans_test[ans_test['y']>=0.592296].shape
alist = ans_test[ans_test['y']>=0.592296]['rowkey'].values.tolist()
dtest['sub'] = dtest['rowkey'].map(lambda x:1 if x in alist else 0)
print '===',dtest[dtest['sub']==1].shape
dtest = dtest.sort_values('rowkey').reset_index(drop=True)
dtest[['rowkey','sub']].to_csv('../datas/sub.csv',index=None,header=None)


ans = pd.read_csv('../datas/ans')
ans = ans[ans['p']==1]

dfval.rename(columns={'y':'y1'},inplace=True)
ans = ans.merge(dfval[['rowkey','y1','is_risk','id','time']],on='rowkey',how='left')
ans['sum'] = ans['y1']

ans = ans.sort_values('sum',ascending=False).reset_index(drop=True)
ans.to_csv('../datas/{0}ans'.format(xstr),index=None)
print ans.head(50)
print ans.loc[50:100]

cols = df_train.columns.tolist()
scores = gbm.feature_importance()
df = pd.DataFrame({'cols':cols,'scores':scores})
df = df.sort_values('scores',ascending=False).reset_index(drop=True)

df.head(300).to_csv('../datas/a',index=None,header=None)

