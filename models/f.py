# coding:utf-8

import pandas as pd


lgb80 = pd.read_csv('../datas/lgb60_month__yval_pred')


lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)

print "-"*20,'xgb100'

print "-"*20,'adboost80'

#print "-"*20,'lgb80'
#print lgb80.head(50)

#print "-"*20,'lgb100'
#print lgb100.head(50)
d = lgb80
xs = d[d['is_risk']==1].shape[0]
n = 0
d = d[['is_risk','rowkey','y']]
def f(x):
    global n
    n += x
    return n
d['cnt'] = d['is_risk'].map(f)
d['recall'] = 1.0*d['cnt']/xs
d['precision'] = 1.0*d['cnt']/(1+d.index)
d['f'] =(1+0.1*0.1)*d['precision']*d['recall']/(0.1*0.1*d['precision']+d['recall'])
print "+"*20,'方法一'
print d[:30]
