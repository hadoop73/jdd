# coding:utf-8

import pandas as pd

lgb80 = pd.read_csv('../datas/lgb60_month_a_yval_pred')
lgb_month = pd.read_csv('../datas/lgb60_month_6_yval_pred')

lgb80 = lgb80.sort_values('y',ascending=False).reset_index(drop=True)

print "-"*20,'xgb100'
print "-"*20,'adboost80'
print "-"*20,'lgb8061'

# 融合方法一
bs = []
i = 0
yi = 0.788189252031     # lgb60_more_yval_pred
y2 = 0.79485472459     # lgb60_month_b_id_yval_pred

bs_test = []
lgb80_test = pd.read_csv('../datas/lgb60_month_a_ytest_pred')
lgb_month_test = pd.read_csv('../datas/lgb60_month_6_ytest_pred')
while lgb80_test.loc[i]['y']>=yi:
    bs_test.append(lgb80_test.loc[i]['rowkey'])
    i += 1

print len(bs_test)
i = 0
while lgb_month_test.loc[i]['y']>=y2:
    bs_test.append(lgb_month_test.loc[i]['rowkey'])
    i += 1

print len(bs_test)
bs_test = set(bs_test)
print 'merge test: ',len(bs_test)


lgb80_test['p'] = lgb80_test['rowkey'].map(lambda x: 1 if x in bs_test else 0)

#print lgb80.sort_values(['a','y'],ascending=False).head(60)

d = lgb80_test[['rowkey','p']].sort_values('rowkey')

print d[d['p']==1].shape
print d.head()

d.to_csv('../datas/ans_test',index=None)


print '\n'
print '-'*30
print '\n'

i = 0
n0 = 0
while lgb80.loc[i]['y']>=yi:
    bs.append(lgb80.loc[i]['rowkey'])
    if lgb80.loc[i]['is_risk'] == 0:
        n0 += 1
    i += 1

print 'one:', len(bs)
print '000:',n0


i = 0
n0 = 0
while lgb_month.loc[i]['y']>=y2:
    if lgb_month.loc[i]['rowkey'] not in bs and lgb_month.loc[i]['is_risk'] == 0:
        n0 += 1
    bs.append(lgb_month.loc[i]['rowkey'])
    i += 1

bs = set(bs)
print 'two:',len(bs)
print '000:',n0


lgb80['p'] = lgb80['rowkey'].map(lambda x: 1 if x in bs else 0)

#print lgb80.sort_values(['a','y'],ascending=False).head(60)

d = lgb80[['rowkey','p']].sort_values('rowkey')

print d[d['p']==1].shape
print d.head()

d.to_csv('../datas/ans',index=None)
