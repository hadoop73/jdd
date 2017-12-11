# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df_train = pd.read_csv('../../datas/login_trade_train')
df_train.fillna(0,inplace=True)

df_train = df_train.drop('time',axis=1)

test = pd.read_csv('../../datas/login_trade_test')
test.fillna(0,inplace=True)

test = test.drop('time',axis=1)
dtest = test[['is_risk','rowkey']]
d_test = test.drop(['is_risk','rowkey'], axis=1).values

y_train = df_train['is_risk'].values
#y_test = df_test['is_risk'].values
df_train = df_train.drop(['is_risk','rowkey'], axis=1)
X_train = df_train.values

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100,learning_rate=.1,
         max_depth=10, random_state=0)
clf.fit(X_train,y_train)
fpr, tpr, _ = roc_curve(y_train, clf.predict_proba(X_train)[:,1])

roc_auc = auc(fpr, tpr)
print roc_auc

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()