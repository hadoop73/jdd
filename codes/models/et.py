# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# TODO 调参
df_train = pd.read_csv('../../datas/login_trade_train')
df_train.fillna(0,inplace=True)

# 构造测试集
df_val = df_train[df_train['time']>='2015-06-01 00:00:00']
# 构造训练集
df_train = df_train[df_train['time']<'2015-06-01 00:00:00'][df_train['time']>='2015-04-10 00:00:00']

df_train = df_train.drop('time',axis=1)
df_val = df_val.drop('time',axis=1)

#test = pd.read_csv('../datas/baseline_feas_test')
test = pd.read_csv('../../datas/login_trade_test')
test.fillna(0,inplace=True)
test = test.drop('time',axis=1)

y_train = df_train['is_risk'].values
y_val = df_val['is_risk'].values

dftrain = df_train[['is_risk','rowkey']]
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
dfval = df_val[['is_risk','rowkey']]
df_val = df_val.drop(['is_risk','rowkey'], axis=1)

X_train = df_train.values
X_val = df_val.values

dtest = test[['is_risk','rowkey']]
X_test = test.drop(['is_risk','rowkey'], axis=1).values

from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=100, max_depth=10,
          random_state=0)
clf.fit(X_train,y_train)

# 预测验证集
ytest_pred = clf.predict_proba(X_test)[:,1]
dtest['y'] = ytest_pred
dtest = dtest.sort_values('y',ascending=False).reset_index(drop=True)
dtest.to_csv('../../datas/et_ytest_pred',index=None)

# 预测验证集
yval_pred = clf.predict_proba(X_val)[:,1]
dfval['y'] = yval_pred
dfval = dfval.sort_values('y',ascending=False).reset_index(drop=True)
dfval.to_csv('../../datas/et_yval_pred',index=None)

# 预训练证集
ytrain_pred = clf.predict_proba(X_train)[:,1]
dftrain['y'] = ytrain_pred
dftrain = dftrain.sort_values('y',ascending=False).reset_index(drop=True)
dftrain.to_csv('../../datas/et_ytrain_pred',index=None)

fpr, tpr, _ = roc_curve(y_train, ytrain_pred)

roc_auc = auc(fpr, tpr)
print 'train:',roc_auc

fpr, tpr, _ = roc_curve(y_val, yval_pred)

roc_auc = auc(fpr, tpr)
print 'valid:',roc_auc

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()




