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
xstr = 'lgb60_month_6a'

feas = pd.read_csv('../datas/feas_new_month3')
feas = feas.drop(['trade_cnt','trade_weekday_cnt','trade_weekday_cnt_rate','hour_v_cnt','hour_v_cnt_rate','is_risk'],axis=1)
#
# print feas.shape

df_train = pd.read_csv('../datas/all3.csv')  # 0.6 filter
df_train = df_train.merge(feas,on='rowkey')

acl = [ci for ci in df_train.columns if 'timestamp_online' in ci]
acl = acl + [ci for ci in df_train.columns if 'timestamp' in ci]
#alist = ['iptype_cnt_rate2','idip_cnt_scan0rate2','idip_cnt_rate2','devicetype_cnt_scan00','idtype_ip0','iptype_cnt_1rate2','ip_cnt_scan10','iddevice_cnt_00','iddevice_cnt_11','ipdevice_cnt_12','idcity_device1','idtype_device1','ipdevice_city1','iptype_cnt_00','idcity_type1','id_cnt_scan11','ip_cnt_scan01','id_cnt_scan12','idip_cnt1','ipdevice_cnt_scan01','idip_cnt_11','idcity_cnt_scan12','idip_city2','device_from2','device_from0','ip_cnt_sec01','iddevice_cnt_rate0','idip_cnt_1rate1','idcity_from2','idcity_from0','iptype_cnt_scan00','idip_cnt_02','iddevice_type2','iddevice_type1','iptype_cnt_1rate1','idcity_type2','idip_cnt_scan00','iddevice_cnt_scan0rate2','iddevice_cnt_02','idip_cnt_scan02','idcity_cnt_sec00','idcity_cnt_sec01','ipdevice_cnt_02','ip_cnt_scan0rate2','ip_cnt_scan0rate1','iddevice_from2','idip_device1','idip_from2','idip_from0','ipdevice_diff21','ipdevice_diff22','iptype_cnt0','iptype_cnt1','idip_cnt_1rate2','iptype_cnt_scan0rate0','ipdevice_diff30','iddevice_ip2','idtype_ip1','devicetype_cnt_scan11','iddevice_cnt_10','ipdevice_cnt_11','id_from0','id_from1','ipdevice_city0','ip_cnt_12','ipdevice_time_mean1','idtype_cnt_scan02','ipdevice_diff11','iddevice_cnt1','device_ip1','idcity_device2','idip_cnt0','ipdevice_cnt_scan02','devicetype_from1','idip_cnt_12','devicetype_cnt_01','idtype_cnt_scan01','idip_cnt_1rate0','iddevice_cnt_rate2','idcity_from1','devicetype_ip0','iptype_cnt_scan02','id_from2','ip_device2','ip_city0','ipdevice_cnt2','iddevice_cnt_01','idip_cnt_scan01','iddevice_diff31','iddevice_from0','ip_cnt_02','ip_cnt_00','idtype_cnt_scan0rate1','idip_from1','ipdevice_diff31','iddevice_ip0','device_cnt_sec02','ipdevice_cnt_rate2','ip_cnt_scan11','idip_cnt_scan12','idcity_cnt_11','id_device1','idtype_cnt_scan00','idcity_cnt_scan00','device_cnt_scan02','device_cnt_scan00','ip_from1','id_cnt_scan0rate1','idcity_ip1','iptype_cnt_rate1','id_cnt_sec00','idtype_cnt_00','id_ip1','idtype_cnt1','ip_cnt_1rate2','iddevice_cnt_scan00','id_cnt_scan01','devicetype_ip1','ip_type0','idcity_cnt_rate1','ipdevice_cnt0','ipdevice_cnt_00','ipdevice_diff20','id_cnt_scan0rate2','iddevice_cnt_1rate0','iddevice_cnt_1rate1','iddevice_cnt_sec00','idtype_from2','ip_cnt2','iptype_diff32','idtype_cnt_sec02','ip_cnt_11','ipdevice_diff10','iddevice_cnt2','id_cnt_scan10','idcity_cnt_scan0rate1','devicetype_from2','ip_cnt_scan12','idip_city1','id_cnt_sec02','idtype_cnt0','iddevice_cnt_rate1','iddevice_cnt_scan02','ipdevice_diff32','ipdevice_time_mean2','devicetype_cnt1','idtype_cnt_10','iptype_device1','idcity_cnt_01','iptype_cnt_rate0','iddevice_ip1','iddevice_cnt_12','iptype_diff30','ip_cnt_10','idcity_cnt_scan02','ipdevice_diff12','idtype_cnt_rate2','device_cnt_rate1','idip_diff30','iptype_time_mean2','idip_diff20','ip_cnt_scan00','idcity_ip0','iptype_cnt_scan12','idip_cnt_10','ip_cnt_1rate1','idip_diff31','idip_diff32','devicetype_cnt0','ip_time_mean2','ipdevice_time_max2','devicetype_city0','ip_cnt_01','id_cnt_scan00','id_cnt_rate1','devicetype_cnt_1rate0','iddevice_cnt_1rate2','ip_time_max1','idcity_cnt_12','id_device2','idip_diff12','idcity_cnt_scan01','ip_cnt_rate1','idtype_cnt_rate0','idtype_cnt_rate1','iptype_cnt_10','ip_id2','devicetype_cnt_00','idtype_cnt_1rate2','idcity_cnt_1rate0','device_cnt_scan0rate0','idtype_cnt_11','devicetype_cnt_1rate1','idtype_ip2','devicetype_cnt_scan12','device_type2','device_ip0','ip_cnt_rate2','ipdevice_id1','iptype_time_mean1','idcity_cnt_scan0rate2','iptype_time_min2','idtype_cnt_01','device_cnt_scan10','iptype_time_max2','device_cnt_00','devicetype_diff30','id_cnt_rate2','idcity_cnt_rate2','ip_diff12','ip_cnt1','idtype_cnt_12','ip_device1','idip_time_mean2','idcity_cnt_10','id_type0','id_type1','idtype_device2','id_cnt_scan0rate0','idip_diff22','idcity_cnt_scan0rate0','idcity_cnt_scan10','id_ip2','ipdevice_time_min2','ip_cnt0','devicetype_city1','idip_time_max1','idtype_from0','iddevice_diff12','iptype_diff31','device_cnt_scan01','idcity_ip2','iptype_diff20','idcity_cnt_scan11','devicetype_cnt_scan0rate2','id_ip0','ipdevice_time_max1']
#acl = acl + alist

acl = acl + ['trade_stamp']#,'hour','id','weekday' ,'city0','city1','city2','log_from0','log_from1','log_from2']

print(acl)
df_train = df_train.drop(acl,axis=1)
#df_train = df_train.fillna(0)
#df_train = df_train.drop(['hour','weekday'],axis=1)
# df_train = df_train[ac]
print df_train.shape

feas = pd.read_csv('../datas/feasb1')
feas3 = pd.read_csv('../datas/feasb1_month3')
feas = pd.concat([feas,feas3])
feas = feas.drop(['time','is_risk','id'],axis=1)
print '--',feas.shape

df_train = df_train.merge(feas,on='rowkey')

# feas = pd.read_csv('../datas/feas_month_3_7')
# alist = ['rowkey','device_diff1','id_diff1','device_diff2','timelong_5184000_max','id_5184000_max','timelong_1296000_max','ip_diff1','ip_5184000_max','timelong_604800_max','ip_diff2','device_5184000_max','id_city5184000','login_time_diff_5184000_max','ip_diff3','id_ip5184000','ip_id5184000','id_diff3','login_time_diff_5184000_min','login_time_diff_1296000_min','id_diff2','iplogin_5184000_rate','device_5184000_cnt','id_device5184000']
# df_train = df_train.merge(feas[alist],on='rowkey')

feas = pd.read_csv('../datas/feasb2')
feas3 = pd.read_csv('../datas/feasb2_month3')
feas = pd.concat([feas,feas3])
feas = feas.drop(['time','is_risk','id'],axis=1)
print '--',feas.shape
df_train = df_train.merge(feas,on='rowkey')

feas = pd.read_csv('../datas/feas_a')
df_train = df_train.merge(feas,on='rowkey')


feas = pd.read_csv('../datas/feasb3_all')
feas = feas.drop(['time','is_risk','id'],axis=1)
df_train = df_train.merge(feas,on='rowkey')

feas = pd.read_csv('../datas/feas_login_new31')
feas3 = pd.read_csv('../datas/feas_login_new31_month3')
feas = pd.concat([feas,feas3])
#feas = feas[cls]
print '--',feas.shape
feas = feas.drop(['time','is_risk','id','hour'],axis=1)
df_train = df_train.merge(feas,on='rowkey')

droplist = ['hour2_cnt','all_cnt','trade_cnt','trade_weekday_cnt','trade_weekday_cnt_rate','hour_v_cnt',
            'hour_v_cnt_rate','idcnt','time_1_diff0','time_1_diff1','ipcnt','iprate',
            'ip_log_from_time_diff_min2','device_log_from_time_diff_min2','device_log_from_time_diff_min0',
            'ip_log_from_time_diff_min0','device_log_from_time_diff_min1','ip_log_from_time_diff_min1',
            'trade_1800_cnt','trade_60_cnt','trade_120_cnt','trade_300_cnt','trade_86400_cnt','trade_3600_cnt','trade_600_cnt',
            'timelong_max','timelong0_max','login2_time_diff','id_cnt_scan12','id_cnt_scan11','id_cnt_scan10'
            # time mean
            # 'idcity_time_mean0','ip_time_mean0','idtype_time_mean0',
            # 'ip_time_mean2','iptype_time_mean0','idtype_time_mean1','iddevice_time_mean0','idip_time_mean0','id_time_mean0',
            # 'id_time_mean2','idtype_time_mean2','idcity_time_mean1','id_time_mean1','device_time_mean1','iddevice_time_mean1',
            # 'devicetype_time_mean1','devicetype_time_mean0','devicetype_time_mean2','ipdevice_time_mean0','device_time_mean0',
            # 'iddevice_time_mean2','ip_time_mean1','idip_time_mean1','iptype_time_mean1','device_time_mean2','idcity_time_mean2',
            # 'idip_time_mean2','ipdevice_time_mean2','iptype_time_mean2','ipdevice_time_mean1'
            ]

df_train['login2dayipcnt_left'] = df_train['login2daycnt'] - df_train['ip_size_u']
df_train['login2daylogfromcnt_left'] = df_train['login2daycnt'] - df_train['log_from_size_u']

tis = [60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60, 48 * 60 * 60]
sis = [0, 60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60]
for si,ti in zip(sis,tis):
    for i in [1, 2, 3, 8, 10, 11, 16, 18, 21]:
        droplist.append('log_from_{0}_{1}'.format(ti, i))

    for i in [-4, -2, -1, 5, 1, 6, 22, 31]:
        droplist.append('result_{0}_{1}'.format(ti, i))

    for i in [1, 2, 3]:
        droplist.append('type_{0}_{1}'.format(ti, i))

for i in [0,1,2]:
    for ci in [('id', 'ip'),('id', 'device'),('id', 'type'),
                   ('ip', 'device'),('ip', 'type'),('device', 'type'),('id','city')]:
        ci = ci[0] + ci[1]
        droplist.append(ci + '_cnt_sec1{0}'.format(i))
        droplist.append(ci + '_cnt_sec0{0}'.format(i))
        droplist.append(ci + '_cnt_scan1{0}'.format(i))
        droplist.append(ci + '_cnt_scan0{0}'.format(i))
        droplist.append(ci + '_cnt_scan0rate{0}'.format(i))
df_train = df_train.drop(droplist,axis=1)

#test = pd.read_csv('../datas/baseline_feas_test')
test = df_train[df_train['time']>='2015-07-01 00:00:00']
dtest = test[['is_risk','rowkey','time']]

test = test.drop('time',axis=1)

df_train = df_train[df_train['time']<'2015-07-01 00:00:00']

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
# 构造训练集
#df_train = df_train[df_train['time']>='2015-04-01 00:00:00']
df_train = df_train[df_train['time']<'2015-06-01 00:00:00']#[df_train['time']>='2015-05-01 00:00:00']

df_train = df_train.drop('time',axis=1)
dfval = df_val[['is_risk','rowkey','time','id']]
df_val = df_val.drop('time',axis=1)



# data_view 33 cell，正样本 203 个
#df_test = df_train[df_train['rowkey']>=736366]
#df_train = df_train[df_train['rowkey']<736366]#[df_train['rowkey']>=633047]

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
    'bagging_fraction': 0.8,    # 随机选择 80% 的数据进行 bagging
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
                num_boost_round=88,                                # 80
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
print("top n: ",n)
print('test:',dtest[dtest['y']>=yi].shape[0])
# 预训练证集
ytrain_pred = gbm.predict(X_train)
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../datas/{0}_ytrain_pred'.format(xstr),index=None)
#y_pred = gbm.predict(X_test)
#xtest['y'] = y_pred
#xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)


cols = df_train.columns.tolist()
scores = gbm.feature_importance()
df = pd.DataFrame({'cols':cols,'scores':scores})
df = df.sort_values('scores',ascending=False).reset_index(drop=True)

df.to_csv('../datas/aaaa',index=None,header=None)


#print xtest.head()
#xtest.to_csv('../datas/xtest',index=None)
#print evals_result
#lgb.plot_metric(evals_result,metric='auc')
# #lgb.plot_metric(evals_result,metric='binary_logloss')
#lgb.plot_importance(gbm, max_num_features=50)
#
# graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree0')
# graph.render(view=True)
#plt.show()
