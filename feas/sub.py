# coding:utf-8

import pandas as pd

r = pd.read_csv('../datas/lgb60_month__ytest_pred')


r['is_risk'] = r.index.map(lambda x: 1 if x < 30 else 0)

print r.head(50)


r = r.sort_values('rowkey')
r[['rowkey','is_risk']].to_csv('../datas/aaa.csv',index=None,header=None)

print r.head()
