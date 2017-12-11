# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
import copy
from multiprocessing import Pool

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
# TODO 用 month_login 数据进行训练,调参框架，筛数据
xstr = 'lgb60_month_all1'


feas = pd.read_csv('../datas/feas_new')
feas = feas.drop('is_risk',axis=1)

df_train = pd.read_csv('../datas/all3.csv')  # 0.6 filter
df_train = df_train.merge(feas,on='rowkey')

acl = [ci for ci in df_train.columns if 'timestamp_online' in ci]
acl = acl + [ci for ci in df_train.columns if 'timestamp' in ci]
# alist = ['iptype_cnt_rate2','idip_cnt_scan0rate2','idip_cnt_rate2','devicetype_cnt_scan00','idtype_ip0','iptype_cnt_1rate2','ip_cnt_scan10','iddevice_cnt_00','iddevice_cnt_11','ipdevice_cnt_12','idcity_device1','idtype_device1','ipdevice_city1','iptype_cnt_00','idcity_type1','id_cnt_scan11','ip_cnt_scan01','id_cnt_scan12','idip_cnt1','ipdevice_cnt_scan01','idip_cnt_11','idcity_cnt_scan12','idip_city2','device_from2','device_from0','ip_cnt_sec01','iddevice_cnt_rate0','idip_cnt_1rate1','idcity_from2','idcity_from0','iptype_cnt_scan00','idip_cnt_02','iddevice_type2','iddevice_type1','iptype_cnt_1rate1','idcity_type2','idip_cnt_scan00','iddevice_cnt_scan0rate2','iddevice_cnt_02','idip_cnt_scan02','idcity_cnt_sec00','idcity_cnt_sec01','ipdevice_cnt_02','ip_cnt_scan0rate2','ip_cnt_scan0rate1','iddevice_from2','idip_device1','idip_from2','idip_from0','ipdevice_diff21','ipdevice_diff22','iptype_cnt0','iptype_cnt1','idip_cnt_1rate2','iptype_cnt_scan0rate0','ipdevice_diff30','iddevice_ip2','idtype_ip1','devicetype_cnt_scan11','iddevice_cnt_10','ipdevice_cnt_11','id_from0','id_from1','ipdevice_city0','ip_cnt_12','ipdevice_time_mean1','idtype_cnt_scan02','ipdevice_diff11','iddevice_cnt1','device_ip1','idcity_device2','idip_cnt0','ipdevice_cnt_scan02','devicetype_from1','idip_cnt_12','devicetype_cnt_01','idtype_cnt_scan01','idip_cnt_1rate0','iddevice_cnt_rate2','idcity_from1','devicetype_ip0','iptype_cnt_scan02','id_from2','ip_device2','ip_city0','ipdevice_cnt2','iddevice_cnt_01','idip_cnt_scan01','iddevice_diff31','iddevice_from0','ip_cnt_02','ip_cnt_00','idtype_cnt_scan0rate1','idip_from1','ipdevice_diff31','iddevice_ip0','device_cnt_sec02','ipdevice_cnt_rate2','ip_cnt_scan11','idip_cnt_scan12','idcity_cnt_11','id_device1','idtype_cnt_scan00','idcity_cnt_scan00','device_cnt_scan02','device_cnt_scan00','ip_from1','id_cnt_scan0rate1','idcity_ip1','iptype_cnt_rate1','id_cnt_sec00','idtype_cnt_00','id_ip1','idtype_cnt1','ip_cnt_1rate2','iddevice_cnt_scan00','id_cnt_scan01','devicetype_ip1','ip_type0','idcity_cnt_rate1','ipdevice_cnt0','ipdevice_cnt_00','ipdevice_diff20','id_cnt_scan0rate2','iddevice_cnt_1rate0','iddevice_cnt_1rate1','iddevice_cnt_sec00','idtype_from2','ip_cnt2','iptype_diff32','idtype_cnt_sec02','ip_cnt_11','ipdevice_diff10','iddevice_cnt2','id_cnt_scan10','idcity_cnt_scan0rate1','devicetype_from2','ip_cnt_scan12','idip_city1','id_cnt_sec02','idtype_cnt0','iddevice_cnt_rate1','iddevice_cnt_scan02','ipdevice_diff32','ipdevice_time_mean2','devicetype_cnt1','idtype_cnt_10','iptype_device1','idcity_cnt_01','iptype_cnt_rate0','iddevice_ip1','iddevice_cnt_12','iptype_diff30','ip_cnt_10','idcity_cnt_scan02','ipdevice_diff12','idtype_cnt_rate2','device_cnt_rate1','idip_diff30','iptype_time_mean2','idip_diff20','ip_cnt_scan00','idcity_ip0','iptype_cnt_scan12','idip_cnt_10','ip_cnt_1rate1','idip_diff31','idip_diff32','devicetype_cnt0','ip_time_mean2','ipdevice_time_max2','devicetype_city0','ip_cnt_01','id_cnt_scan00','id_cnt_rate1','devicetype_cnt_1rate0','iddevice_cnt_1rate2','ip_time_max1','idcity_cnt_12','id_device2','idip_diff12','idcity_cnt_scan01','ip_cnt_rate1','idtype_cnt_rate0','idtype_cnt_rate1','iptype_cnt_10','ip_id2','devicetype_cnt_00','idtype_cnt_1rate2','idcity_cnt_1rate0','device_cnt_scan0rate0','idtype_cnt_11','devicetype_cnt_1rate1','idtype_ip2','devicetype_cnt_scan12','device_type2','device_ip0','ip_cnt_rate2','ipdevice_id1','iptype_time_mean1','idcity_cnt_scan0rate2','iptype_time_min2','idtype_cnt_01','device_cnt_scan10','iptype_time_max2','device_cnt_00','devicetype_diff30','id_cnt_rate2','idcity_cnt_rate2','ip_diff12','ip_cnt1','idtype_cnt_12','ip_device1','idip_time_mean2','idcity_cnt_10','id_type0','id_type1','idtype_device2','id_cnt_scan0rate0','idip_diff22','idcity_cnt_scan0rate0','idcity_cnt_scan10','id_ip2','ipdevice_time_min2','ip_cnt0','devicetype_city1','idip_time_max1','idtype_from0','iddevice_diff12','iptype_diff31','device_cnt_scan01','idcity_ip2','iptype_diff20','idcity_cnt_scan11','devicetype_cnt_scan0rate2','id_ip0','ipdevice_time_max1']
# acl = acl + alist
df_train = df_train.drop(['trade_stamp']+acl,axis=1)

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
    'max_depth':8,
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc', # auc
    'num_leaves': 61,           # 31 替换为 61
    #'learning_rate': 0.05,
    'feature_fraction': 0.7,    # 随机选 90% 的特征用于训练
    'bagging_fraction': 0.8,    # 随机选择 80% 的数据进行 bagging
    'bagging_freq': 5,          # bagging 没 5 次进行
    'max_bin':80,              # 特征最大分割
    'min_data_in_leaf':50,      # 每个叶子节点最少样本
    'verbose': 0
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

lgb_test = lgb.Dataset(X_test)

def train(dp,dfval=dfval):
        print dp['params']
        gbm = lgb.train(dp['params'],
                        lgb_train,
                        feature_name=df_train.columns.tolist(),
                        num_boost_round= dp['num'],                                # 80
                        learning_rates=lambda iter: dp['rate'] * (0.99 ** iter) # 0.1 替换为 0.08
                        )
        # 预测验证集
        yval_pred = gbm.predict(X_val)
        dfval['y'] = yval_pred
        dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)

        n = 0
        while dfval.loc[n]['is_risk'] == 1:
            n += 1
        print "top n: ",n
        return n


p = {
        'max_bin': 100,  # 特征最大分割
        'bagging_freq': 50,  # bagging 没 5 次进行
        'min_data_in_leaf': 50,  # 每个叶子节点最少样本
        'max_depth': 9,
        'num_leaves': 61,  # 31 替换为 61
        'feature_fraction': 0.7,  # 随机选 90% 的特征用于训练
        'bagging_fraction': 0.8,  # 随机选择 80% 的数据进行 bagging

    }

ans = 0
bp = None

def fun(ki,i=3):
    if i<1: return
    ps,dp = [],{}
    global p
    for r in [0.5,  1.2,  1.5]:
        if type(p[ki]) == int:
            p[ki] = int(r * p[ki])
        elif r * p[ki] >= 1:
            continue
        else:
            p[ki] = r * p[ki]
        params.update(p)
        dp['params'] = params
        for num in [60, 80, 100]:
            for rate in [0.1, 0.08]:
                dp['num'] = num
                dp['rate'] = rate
                dp['r'] = r
                ti = copy.deepcopy(dp)
                ps.append(ti)
    pool = Pool(3)
    rs = pool.map(train,ps)
    rmax = max(rs)
    global ans
    if rmax > ans:
        for ai,bi in zip(rs,ps):
            if ai == rmax:
                with open('./ans/res.txt','a') as f :
                    f.writelines('\nans：{0}'.format(rmax))
                    f.writelines('{0}'.format(bi))
                p = bi['params']
        ans = rmax

    for ai, bi in zip(rs, ps):
        if ai==rmax and bi['r']!=1.2:
             # 继续调参
             fun(ki,i-1)

def rand_params():
    cols = ['max_bin', 'max_depth', 'num_leaves', 'feature_fraction',
            'bagging_fraction', 'bagging_freq', 'min_data_in_leaf']
    for ki in cols:
        fun(ki)


rand_params()


