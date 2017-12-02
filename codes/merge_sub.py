# coding:utf-8

import pandas as pd

xgb100 = pd.read_csv('../datas/xgb_100_month_ytest_pred')
adboost80 = pd.read_csv('../datas/adboost_80_all_ytest_pred')
lgb80 = pd.read_csv('../datas/lgb80_month_ytest_pred')



xgb100 = xgb100.sort_values('y',ascending=False).reset_index(drop=True)
adboost80 = adboost80.sort_values('y',ascending=False).reset_index(drop=True)
lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)

print "-"*20,'xgb100'

print "-"*20,'adboost80'



print "-"*20,'lgb8061'

xr = range(xgb100.shape[0])
xr = xr[::-1]

xgb100['range1'] = xr
adboost80['range2'] = xr
lgb80['range3'] = xr


# 融合方法一
bs = set(xgb100.loc[:14]['rowkey'].values.tolist() + \
    adboost80.loc[:13]['rowkey'].values.tolist() + \
    lgb80.loc[:12]['rowkey'].values.tolist())

print len(bs)
xgb100['a'] = xgb100['rowkey'].map(lambda x:1 if x in bs else 0)
xgb100['a'] = xgb100.index.map(lambda x:1 if x<30 else 0)
d = xgb100.sort_values(['a','y'],ascending=False).reset_index(drop=True)

print d.head(50)
d = d.sort_values('rowkey')
d[['rowkey','a']].to_csv('../datas/aa.csv',index=None,header=None)





