# coding:utf-8

import pandas as pd

lgb80= pd.read_csv('../datas/lgb60_month_a_yval_pred')
#lgb100 = pd.read_csv('../datas/lgb60_month2_yval_pred')
xgb100 = pd.read_csv('../datas/lgb60_month_6_yval_pred')


xgb100 = xgb100.sort_values('y',ascending=False).reset_index(drop=True)
lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)

print "-"*20,'xgb100'
print "-"*20,'adboost80'
print "-"*20,'lgb8061'

# 融合方法一
bs = []
i = 0
yi = 0
while lgb80.loc[i]['is_risk']==1:
    bs.append(lgb80.loc[i]['rowkey'])

    yi = lgb80.loc[i]['y']
    i += 1


xi = 0
yyi = 0
bi = 0
xbs = []
while xgb100.loc[xi]['is_risk']==1:
    xbs.append(xgb100.loc[xi]['rowkey'])
    if xgb100.loc[xi]['rowkey'] in bs:
        bi += 1
    yyi = xgb100.loc[xi]['y']
    xi += 1
    if xi%10 ==0:
        print 'xi:',xi,xi - bi


print 'xi:',xi,xi - bi,yyi,yi

#yi = lgb80.loc[i-5]['y']
#yyi = xgb100.loc[xi-5]['y']
print yyi,yi

bs += xbs
bs = set(bs)
print 'merge valid',len(bs)
lgb80['a'] = lgb80['rowkey'].map(lambda x: 1 if x in bs else 0)
dd = lgb80

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

lgb80 = pd.read_csv('../datas/lgb60_month_a_ytest_pred')
print 'test--------',lgb80.shape
xgb100 = pd.read_csv('../datas/lgb60_month_6_ytest_pred')

xgb100 = xgb100.sort_values('y',ascending=False).reset_index(drop=True)
lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)

bs =  lgb80[lgb80['y']>=yi]['rowkey'].values.tolist()
#print bs
xbs = xgb100[xgb100['y']>=yyi]['rowkey'].values.tolist()
print len(bs),len(xbs)
xxi = 0
for xbi in range(len(xbs)):
    if xbs[xbi] in bs:
        xxi += 1
    if xbi%10 == 0:
        print('xbi:',xbi,xbi + 1 - xxi)

print('all:',xbi+1,xbi + 1 - xxi)

bs += xbs
#bs = bs + xgb100.loc[:34]['rowkey'].values.tolist()

print len(bs)
bs = set(bs)
print 'merge test',len(bs)

# drop = [175932,15259,24499,29741,79383,79372,79369,79450,79447,79376,165543,5052,5040,4907,4883,79450,79447,79376,127567,127564,110429,110425,149176,149091,154255,116137]
#
# bs = [i  for i in bs if i not in drop]
#
# insert = [13934,13932,14569,14561,85955,85952,5028,5034,5162,5165,112897,110429,106152,106147,93223,116307,116308]
#
#
# bs += insert
#
# bs = set(bs)

lgb80['a'] = lgb80['rowkey'].map(lambda x: 1 if x in bs else 0)


#print lgb80.sort_values(['a','y'],ascending=False).head(60)

d = lgb80[['rowkey','a']].sort_values('rowkey')


print d[d['a']==1].shape
print d.head()
print d.shape
#d.to_csv('../datas/abc.csv',index=None,header=None)
