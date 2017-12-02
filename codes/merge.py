# coding:utf-8

import pandas as pd

xgb100 = pd.read_csv('../datas/xgb_100_monthv_yval_pred')
adboost80 = pd.read_csv('../datas/adboost_80_all_yval_pred')
lgb80 = pd.read_csv('../datas/lgb60_month__yval_pred')
lgb100 = pd.read_csv('../datas/lgb100_yval_pred')
lgb8061 = pd.read_csv('../datas/lgb8061_yval_pred')


xgb100 = xgb100.sort_values('y',ascending=False).reset_index(drop=True)
adboost80 = adboost80.sort_values('y',ascending=False).reset_index(drop=True)
lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)
lgb100 = lgb100.sort_values('y',ascending=False).reset_index(drop=True)
lgb8061 = lgb8061.sort_values('y',ascending=False).reset_index(drop=True)

print "-"*20,'xgb100'

print "-"*20,'adboost80'

#print "-"*20,'lgb80'
#print lgb80.head(50)

#print "-"*20,'lgb100'
#print lgb100.head(50)

print "-"*20,'lgb8061'

xr = range(xgb100.shape[0])
xr = xr[::-1]

xgb100['range1'] = xr
adboost80['range2'] = xr
lgb80['range3'] = xr
lgb100['range4'] = xr
lgb8061['range5'] = xr

# 融合方法一
bs = []
i = 0
while lgb80.loc[i]['is_risk']==1:
    bs.append(lgb80.loc[i]['rowkey'])
    i += 1

xi = 0
while xgb100.loc[xi]['is_risk']==1:
    #bs.append(xgb100.loc[xi]['rowkey'])
    xi += 1

bs = set(bs)

dd = lgb80
print len(bs)
dd['a'] = dd['rowkey'].map(lambda x:1 if x in bs else 0)
d = dd.sort_values(['a','y'],ascending=False).reset_index(drop=True)

xs = d[d['is_risk']==1].shape[0]
n = 0
d = d[['is_risk','rowkey','y','a']]
def f(x):
    global n
    n += x
    return n
d['cnt'] = d['is_risk'].map(f)
d['recall'] = 1.0*d['cnt']/xs
d['precision'] = 1.0*d['cnt']/(1+d.index)
d['f'] =(1+0.1*0.1)*d['precision']*d['recall']/(0.1*0.1*d['precision']+d['recall'])
print "+"*20,'方法一'
print d[:50]

d = xgb100.merge(adboost80[['rowkey','range2']],on='rowkey')
d = d.merge(lgb80[['rowkey','range3']],on='rowkey')
d = d.merge(lgb100[['rowkey','range4']],on='rowkey')
d = d.merge(lgb8061[['rowkey','range5']],on='rowkey')

# 融合方法二
d['s'] = 1.0*(0.8*d['range1']+0.*d['range2']+0.2*d['range3'])/(xgb100.shape[0])

d = d.sort_values('s',ascending=False).reset_index(drop=True)

d = d[['is_risk','rowkey','y','s']]

#d.to_csv('../datas/3a.csv',index=None)
xs = d[d['is_risk']==1].shape[0]
n = 0
def f(x):
    global n
    n += x
    return n
d['cnt'] = d['is_risk'].map(f)
d['recall'] = 1.0*d['cnt']/xs
d['precision'] = 1.0*d['cnt']/(1+d.index)
d['f'] =(1+0.1*0.1)*d['precision']*d['recall']/(0.1*0.1*d['precision']+d['recall'])
print "+"*20,'方法二'
print d.head(30)


# 融合方法三

d1 = xgb100
d2 = adboost80[['is_risk','rowkey','y']]
d2.columns = ['is_risk','rowkey','y2']

d3 = lgb80[['is_risk','rowkey','y']]
d3.columns = ['is_risk','rowkey','y3']

d = d1.merge(d2[['rowkey','y2']],on='rowkey')
d = d.merge(d3[['rowkey','y3']],on='rowkey')

d['yymin'] = d[['y','y2','y3']].min(axis=1)
d['yymax'] = d[['y','y2','y3']].max(axis=1)
d['yy'] = d[['y','y3']].sum(axis=1)# - d['yymax']
d = d.sort_values('yy',ascending=False).reset_index(drop=True)
xgb100['a'] = xgb100['rowkey'].map(lambda x:1 if x in bs else 0)

n = 0
def f(x):
    global n
    n += x
    return n
d['cnt'] = d['is_risk'].map(f)
d['recall'] = 1.0*d['cnt']/xs
d['precision'] = 1.0*d['cnt']/(1+d.index)
d['f'] =(1+0.1*0.1)*d['precision']*d['recall']/(0.1*0.1*d['precision']+d['recall'])
d = d[['is_risk','rowkey','y','a','cnt','recall','precision','yy','f']]

print '+'*20,'方法三'
print d.head(30)

