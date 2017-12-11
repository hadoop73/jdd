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
xstr = 'lgb60_month_3'

feas = pd.read_csv('../datas/feas_new_month3')
feas = feas.drop('is_risk',axis=1)

print feas.shape

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
# ac = ['id','rowkey','time','is_risk','hour_x','time_diff1','trade_cnt_x','idtype_time_min0','trade_weekday_cnt_rate_x','id_diff10','idtype_diff10','time_diff2','idcity_diff10','idtype_city0','ip_diff10','time_diff12','trade_cnt_y','iddevice_diff20','idtype_diff20','iddevice_diff10','idtype_time_mean0','device_cnt_trade_rate1','id_time_min0','weekday_x','idcity_diff20','id_diff20','idtype_time_min1','id_time_max0','hour_y','iddevice_time_mean1','idcity_time_mean0','device_diff10','iptype_time_min0','idcity_cnt_rate0','idcity_diff11','idcity_time_max0','id_time_min2','time_cnt3600','iptype_time_mean0','ip_time_min0','idcity_time_min0','idcity_time_mean1','idtype_time_max0','id_diff32','hour_v_cnt_rate_x','iptype_diff10','idtype_diff32','device_diff11','idtype_diff21','idtype_time_min2','idtype_time_max2','time_cnt300','time_cnt1800','ip_id0','ip_time_max0','idip_diff10','weekday_y','idip_diff11','idcity_time_mean2','device_id2','iptype_id0','ip_cnt_trade_rate1','id_city0','idtype_time_max1','idtype_diff30','idtype_diff31','devicetype_time_mean1','devicetype_time_min1','iddevice_time_max1','id_diff12','idtype_time_mean1','id_diff30','idtype_time_mean2','idcity_time_max2','ipdevice_diff10','idip_time_min0','iddevice_time_mean0','idtype_cnt_1rate2','idtype_cnt_rate0','idcity_diff12','idtype_diff12','id_time_mean0','devicetype_diff12','devicetype_diff10','device_diff22','iddevice_diff11','devicetype_diff21','device_time_min0','id_diff22','ip_diff11','id_time_max1','iddevice_time_min1','idtype_cnt_1rate0','iddevice_time_min0','iptype_time_max0','iddevice_time_max2','id_diff11','iddevice_time_max0','ip_diff21','idcity_time_max1','devicetype_time_mean2','idip_cnt_rate0','ip_time_mean1','ip_device_cnt_trade_rate1','device_diff20','trade_weekday_cnt_rate_y','id_diff31','devicetype_id2','ip_time_mean0','idcity_diff32','device_time_max1','id_time_max2','devicetype_diff11','device_cnt_trade_rate2','idip_time_mean1','device_time_mean2','id_time_mean2','devicetype_ip2','devicetype_time_mean0','idcity_diff22','idcity_time_min2','device_cnt2_y','idcity_cnt_scan0rate2','idcity_time_min1','device_time_max0','idip_diff12','device_diff32','id_cnt_00','iptype_time_min1','ip_device_cnt_trade_rate2','id_cnt_rate0','ipdevice_time_min0','idcity_cnt2','id_time_mean1','devicetype_time_min0','idtype_diff11','id_time_min1','ip_time_max2','iddevice_diff21','idip_time_mean2','id_device0','idtype_diff22','ip_id1','id_cnt_1rate2','device_time_mean0','idtype_cnt_rate2','idip_time_mean0','devicetype_diff20','iptype_diff11','device_id1','idcity_cnt_1rate0','hour_v_x','iddevice_diff12','device_cnt1_y','idip_diff22','id_cnt_1rate0','devicetype_time_max1','devicetype_time_max2','ip_time_mean2','idcity_diff30','iddevice_time_min2','devicetype_ip1','id_diff21','ip_time_min2','iddevice_diff22','ipdevice_diff30','idtype_ip0','iptype_time_min2','idtype_city1','devicetype_cnt_1rate0','devicetype_cnt_1rate2','devicetype_time_min2','idtype_city2','device_time_min1','idtype_cnt_10','idcity_diff21','devicetype_time_max0','id_city1','device_diff21','id_cnt2','devicetype_id1','iddevice_cnt0','idcity_diff31','ip_time_max1','idip_time_min2','device_time_max2','ip_time_min1','iddevice_diff32','idip_time_max1','idip_time_max0','ip_cnt_trade_rate2','devicetype_diff31','devicetype_diff32','idcity_cnt_11','idtype_device1','idip_diff21','ipdevice_time_max0','iptype_id1','device_diff12','id_cnt_12','iptype_time_mean1','idtype_cnt_1rate1','device_diff31','devicetype_cnt_rate2','idcity_cnt_scan0rate1','devicetype_cnt_1rate1','idcity_cnt_scan02','iptype_diff21','device_cnt_1rate2','iptype_diff30','device_time_min2','iddevice_time_mean2','iddevice_city1','ip_cnt1_y','idip_time_max2','id_cnt_10','devicetype_diff22','device_time_mean1','idtype_cnt_rate1','idtype_device0','iddevice_cnt_rate2','idtype_cnt2','idcity_cnt_12','idtype_device2','iptype_time_mean2','idip_cnt_00','idip_cnt_rate2','id_cnt_scan0rate1','device_cnt_02','ip_diff12','iptype_cnt_rate0','device_id0','idip_diff20','devicetype_cnt_scan0rate2','devicetype_cnt_scan0rate1','devicetype_cnt_rate1','idtype_cnt_12','iddevice_diff30','ip_cnt_rate2','device_cnt_1rate1','ip_cnt2_x','ip_cnt_1rate0','iddevice_cnt_1rate0','idip_cnt_1rate0','ip_diff30','idtype_cnt_scan0rate0','idcity_cnt_1rate2','devicetype_cnt_02','iptype_time_max2','id_cnt_scan0rate2','hour_v_cnt_y','id_ip2','device_city1','ipdevice_time_min2','id_city2','idcity_cnt0','devicetype_diff30','ipdevice_time_min1','ip_cnt2_y','id_cnt_rate2','idcity_cnt_00','idcity_cnt_01','idcity_cnt_scan01','ipdevice_time_mean0','idcity_cnt_scan00','id_cnt_1rate1','idip_time_min1','device_diff30','idtype_cnt0','iddevice_diff31','id_from0','device_cnt_00','idcity_device0','ip_cnt_rate0','idcity_cnt_1rate1','idcity_ip1','idtype_from1','ipdevice_time_mean1','id_cnt_rate1','idcity_type0','id_cnt0','idip_diff31','trade_weekday_cnt_x','ipdevice_diff20','iddevice_cnt_10','ip_cnt_11','device_cnt_rate0','devicetype_cnt_scan0rate0','device_cnt_rate2','ip_diff20','iddevice_ip0','idtype_from2','iptype_diff20','idcity_cnt_02','id_device2','id_cnt_scan0rate0','device_cnt_12','idtype_cnt_02','hour_v_cnt_x','devicetype_cnt0','devicetype_city2','iddevice_cnt_1rate2','idtype_cnt1']
# df_train = df_train[ac]
print df_train.shape

feas = pd.read_csv('../datas/feasb1')
feas3 = pd.read_csv('../datas/feasb1_month3')
feas = pd.concat([feas,feas3])
feas = feas.drop(['time','is_risk','id'],axis=1)
print '--',feas.shape

df_train = df_train.merge(feas,on='rowkey')

feas = pd.read_csv('../datas/feasb2')
feas3 = pd.read_csv('../datas/feasb2_month3')
feas = pd.concat([feas,feas3])
feas = feas.drop(['time','is_risk','id'],axis=1)
print '--',feas.shape
df_train = df_train.merge(feas,on='rowkey')

feas = pd.read_csv('../datas/feas_login_new31')
feas3 = pd.read_csv('../datas/feas_login_new31_month3')
feas = pd.concat([feas,feas3])
cls = ['rowkey','result31_1','result_31_and_1cnt0','result_timestamp_max0','result_timestamp_min0','result_timestamp_mean0']
#feas = feas[cls]
print '--',feas.shape
feas = feas.drop(['time','is_risk','id','hour'],axis=1)
df_train = df_train.merge(feas,on='rowkey')

droplist = ['hour2_cnt','all_cnt','hour_x','trade_cnt_y','time_cnt3600','time_cnt300','time_cnt1800',
            'weekday_x','trade_weekday_cnt_rate_x','trade_cnt_x','timelong_max','timelong0_max',
            'hour_v_cnt_rate_x','hour_v_x','hour_v_cnt_x','device_cnt2_x',
            'device_cnt1_x','trade_weekday_cnt_x','trade_weekday_cnt_rate_y','hour_v_cnt_y','hour_v_cnt_rate_y',
            'trade_weekday_cnt_y','trade_1_cnt_pre','trade_login_hour_cnt',
            'trade_1800_cnt','trade_120_cnt','trade_60_cnt','trade_86400_cnt',
            'trade_3600_cnt','trade_300_cnt','trade_600_cnt'
            # 评分为 0
            #'iddevice_type0','device_from1','time_1_diff0','iddevice_type2','login_120_cnt180','login_120_cnt60','login_3600_cnt120','device_from2','iddevice_cnt_00','iddevice_type1','ip_log_from_time_diff_min1','idip_cnt_12','login_1800_cnt120','idip_cnt_rate1','device_ip_id_7','device_from0','city_cnt600','device_ip_id_1','idip_device1','login_600_cnt60','login_60_cnt120','device_cnt_sec10','login_600_cnt120','login_60_cnt60','iddevice_from1','city_cnt3600rate','iddevice_cnt_11','idip_cnt_rate2','idip_cnt_01','idip_cnt2','device_cnt_sec11','login_120_cnt120','device_cnt_sec12','idip_from1','idip_device0','city_cnt60','ip_city1','ip_log_from_time_diff_mean0','iptype_cnt_rate2','iptype_city0','iptype_city1','iptype_city2','iptype_cnt1','iptype_cnt_02','iptype_cnt_1rate0','iptype_device0','ipdevice_type1','devicetype_from0','id_cnt_sec02','ip_cnt_sec12','ip_cnt_sec11','ip_cnt_sec10','iptype_from1','ipdevice_type2','ip_from2','ip_cnt_sec02','ipdevice_from2','ipdevice_cnt_rate2','ipdevice_cnt_10','ipdevice_cnt_02','ipdevice_cnt_01','ipdevice_cnt2','ipdevice_from1','ipdevice_cnt0','ip_type0','ipdevice_city2','ipdevice_id2','ipdevice_city1','ipdevice_city0','ip_type2','ip_type1','iptype_from2','ip_cnt_sec01','idip_type0','id_cnt_sec11','time_diff12','time_diff2','ip_city2','ip_city0','device_type2','device_type1','id_cnt_sec12','ip_device_cnt_trade_rate2','idtype_ip1','id_type2','idcity_from2','idtype_cnt_02','idcnt','idip_type2','time_diff1','ip_device_cnt_trade_rate1','ip_cnt_scan10','device_cnt_trade_rate1','ip_cnt_scan0rate2','iptype_time_mean2','ip_cnt_scan01','id_cnt_sec10','device_cnt1_y','device_cnt2_y','device_cnt_trade_rate2','ip_device_cnt2','ip_cnt1_y','ip_cnt2_y','ip_cnt_trade_rate1','ip_cnt_trade_rate2','ip_cnt2_x','ip_device_cnt1',
            #'ipdevice_diff31'
            ]

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
#df_train = df_train[df_train['time']>='2015-05-01 00:00:00']
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
    'num_leaves': 61,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.9,    # 随机选 90% 的特征用于训练
    'bagging_fraction': 0.5,    # 随机选择 80% 的数据进行 bagging
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

df.to_csv('../datas/a',index=None,header=None)


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
