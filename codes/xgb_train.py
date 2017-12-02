# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import operator

# TODO 数据优化，调参，完善一下数据
#df_train = pd.read_csv('../../datas/login_trade_train')
#df_train = pd.read_csv('../../datas/login_trade_train')
xstr = 'xgb_100_monthv'

df_train = pd.read_csv('../datas/all3.csv')  # 0.6 filter
acl = [ci for ci in df_train.columns if 'timestamp_online' in ci]
acl = acl + [ci for ci in df_train.columns if 'timestamp' in ci]
alist = ['iptype_cnt_rate2','idip_cnt_scan0rate2','idip_cnt_rate2','devicetype_cnt_scan00','idtype_ip0','iptype_cnt_1rate2','ip_cnt_scan10','iddevice_cnt_00','iddevice_cnt_11','ipdevice_cnt_12','idcity_device1','idtype_device1','ipdevice_city1','iptype_cnt_00','idcity_type1','id_cnt_scan11','ip_cnt_scan01','id_cnt_scan12','idip_cnt1','ipdevice_cnt_scan01','idip_cnt_11','idcity_cnt_scan12','idip_city2','device_from2','device_from0','ip_cnt_sec01','iddevice_cnt_rate0','idip_cnt_1rate1','idcity_from2','idcity_from0','iptype_cnt_scan00','idip_cnt_02','iddevice_type2','iddevice_type1','iptype_cnt_1rate1','idcity_type2','idip_cnt_scan00','iddevice_cnt_scan0rate2','iddevice_cnt_02','idip_cnt_scan02','idcity_cnt_sec00','idcity_cnt_sec01','ipdevice_cnt_02','ip_cnt_scan0rate2','ip_cnt_scan0rate1','iddevice_from2','idip_device1','idip_from2','idip_from0','ipdevice_diff21','ipdevice_diff22','iptype_cnt0','iptype_cnt1','idip_cnt_1rate2','iptype_cnt_scan0rate0','ipdevice_diff30','iddevice_ip2','idtype_ip1','devicetype_cnt_scan11','iddevice_cnt_10','ipdevice_cnt_11','id_from0','id_from1','ipdevice_city0','ip_cnt_12','ipdevice_time_mean1','idtype_cnt_scan02','ipdevice_diff11','iddevice_cnt1','device_ip1','idcity_device2','idip_cnt0','ipdevice_cnt_scan02','devicetype_from1','idip_cnt_12','devicetype_cnt_01','idtype_cnt_scan01','idip_cnt_1rate0','iddevice_cnt_rate2','idcity_from1','devicetype_ip0','iptype_cnt_scan02','id_from2','ip_device2','ip_city0','ipdevice_cnt2','iddevice_cnt_01','idip_cnt_scan01','iddevice_diff31','iddevice_from0','ip_cnt_02','ip_cnt_00','idtype_cnt_scan0rate1','idip_from1','ipdevice_diff31','iddevice_ip0','device_cnt_sec02','ipdevice_cnt_rate2','ip_cnt_scan11','idip_cnt_scan12','idcity_cnt_11','id_device1','idtype_cnt_scan00','idcity_cnt_scan00','device_cnt_scan02','device_cnt_scan00','ip_from1','id_cnt_scan0rate1','idcity_ip1','iptype_cnt_rate1','id_cnt_sec00','idtype_cnt_00','id_ip1','idtype_cnt1','ip_cnt_1rate2','iddevice_cnt_scan00','id_cnt_scan01','devicetype_ip1','ip_type0','idcity_cnt_rate1','ipdevice_cnt0','ipdevice_cnt_00','ipdevice_diff20','id_cnt_scan0rate2','iddevice_cnt_1rate0','iddevice_cnt_1rate1','iddevice_cnt_sec00','idtype_from2','ip_cnt2','iptype_diff32','idtype_cnt_sec02','ip_cnt_11','ipdevice_diff10','iddevice_cnt2','id_cnt_scan10','idcity_cnt_scan0rate1','devicetype_from2','ip_cnt_scan12','idip_city1','id_cnt_sec02','idtype_cnt0','iddevice_cnt_rate1','iddevice_cnt_scan02','ipdevice_diff32','ipdevice_time_mean2','devicetype_cnt1','idtype_cnt_10','iptype_device1','idcity_cnt_01','iptype_cnt_rate0','iddevice_ip1','iddevice_cnt_12','iptype_diff30','ip_cnt_10','idcity_cnt_scan02','ipdevice_diff12','idtype_cnt_rate2','device_cnt_rate1','idip_diff30','iptype_time_mean2','idip_diff20','ip_cnt_scan00','idcity_ip0','iptype_cnt_scan12','idip_cnt_10','ip_cnt_1rate1','idip_diff31','idip_diff32','devicetype_cnt0','ip_time_mean2','ipdevice_time_max2','devicetype_city0','ip_cnt_01','id_cnt_scan00','id_cnt_rate1','devicetype_cnt_1rate0','iddevice_cnt_1rate2','ip_time_max1','idcity_cnt_12','id_device2','idip_diff12','idcity_cnt_scan01','ip_cnt_rate1','idtype_cnt_rate0','idtype_cnt_rate1','iptype_cnt_10','ip_id2','devicetype_cnt_00','idtype_cnt_1rate2','idcity_cnt_1rate0','device_cnt_scan0rate0','idtype_cnt_11','devicetype_cnt_1rate1','idtype_ip2','devicetype_cnt_scan12','device_type2','device_ip0','ip_cnt_rate2','ipdevice_id1','iptype_time_mean1','idcity_cnt_scan0rate2','iptype_time_min2','idtype_cnt_01','device_cnt_scan10','iptype_time_max2','device_cnt_00','devicetype_diff30','id_cnt_rate2','idcity_cnt_rate2','ip_diff12','ip_cnt1','idtype_cnt_12','ip_device1','idip_time_mean2','idcity_cnt_10','id_type0','id_type1','idtype_device2','id_cnt_scan0rate0','idip_diff22','idcity_cnt_scan0rate0','idcity_cnt_scan10','id_ip2','ipdevice_time_min2','ip_cnt0','devicetype_city1','idip_time_max1','idtype_from0','iddevice_diff12','iptype_diff31','device_cnt_scan01','idcity_ip2','iptype_diff20','idcity_cnt_scan11','devicetype_cnt_scan0rate2','id_ip0','ipdevice_time_max1']
acl = acl + alist
print acl
df_train = df_train.drop(['trade_stamp']+acl,axis=1)


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
#df_train = df_train[df_train['time']>='2015-05-01 00:00:00']

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
    'nthread':8,                # 最大线程数
    'gamma':0.01,               # 节点分裂所需的最小损失函数下降值,默认 0
    'max_depth':8,              # 树的最大深度，默认 6
    'lambda':5,                 # 权重的L2正则化项
    'subsample':0.8,            # 样本抽样
    'colsample_bytree':0.8,     # 列抽样
    #'min_child_weight':1,      # 决定最小叶子节点样本权重和,当它的值较大时，可以避免模型学习到局部的特殊样本,如果这个值过高，会导致欠拟合
    'eta':0.01,                 # 学习率，通过减少每一步的权重，可以提高模型的鲁棒性 0.01-0.2
    #'scale_pos_weight':,       # 类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
    'seed':123,
}

watchlist = [(dtrain,'train'),(dval,'valid')]
bst = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

# 预测验证集
ytest_pred = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
xtest['y'] = ytest_pred
xtest = xtest.sort_values('y',ascending=False).reset_index(drop=True)
xtest.to_csv('../datas/{0}_ytest_pred'.format(xstr),index=None)

# 预测验证集
yval_pred = bst.predict(dval,ntree_limit=bst.best_ntree_limit)
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../datas/{0}_yval_pred'.format(xstr),index=None)

n = 0
while dfval.loc[n]['is_risk'] == 1:
    n += 1
print "top n: ",n

# 预训练证集
ytrain_pred = bst.predict(dtrain,ntree_limit=bst.best_ntree_limit)
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../datas/{0}_ytrain_pred'.format(xstr),index=None)

fpr, tpr, _ = roc_curve(y_train, ytrain_pred)

roc_auc = auc(fpr, tpr)
print 'train:',roc_auc

fpr, tpr, _ = roc_curve(y_val, yval_pred)

roc_auc = auc(fpr, tpr)
print 'valid:',roc_auc


importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

print importance

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df = df.sort_values('fscore',ascending=False).reset_index(drop=True)

df = df.loc[:30].sort_values('fscore',ascending=True)

df.plot(kind='barh', x='feature', y='fscore', legend=False)
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()


