# coding:utf-8

import pandas as pd

ans = pd.read_csv('../datas/improve')
ans = ans[['rowkey','is_risk','id','y1']]

pans = pd.read_csv('../datas/lgb60_mkans')
pans = pans[['rowkey','y1']]
pans.rename(columns={'y1':'y2'},inplace=True)

ans = ans.merge(pans,on='rowkey',how='left')

ans['sum'] = 0.2*ans['y1'] + 0.8*ans['y2']

ans = ans.sort_values('sum',ascending=False).reset_index(drop=True)

print ans.head(50)

print ans.loc[50:100]

#print ans[ans['is_risk']==0].shape
