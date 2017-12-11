# coding:utf-8

import pandas as pd

ans = pd.read_csv('../datas/ans')
ans = ans[ans['p']==1]


lgb80= pd.read_csv('../datas/lgb60_more_yval_pred')
lgb80.rename(columns={'y':'y1'},inplace=True)
ans = ans.merge(lgb80[['rowkey','y1','is_risk','id']],on='rowkey',how='left')


lgb100 = pd.read_csv('../datas/lgb60_month_b_id_yval_pred')
lgb100.rename(columns={'y':'y2'},inplace=True)
ans = ans.merge(lgb100[['rowkey','y2']],on='rowkey',how='left')

xgb100 = pd.read_csv('../datas/lgb60_month_1_yval_pred')
xgb100.rename(columns={'y':'y3'},inplace=True)
ans = ans.merge(xgb100[['rowkey','y3']],on='rowkey',how='left')

m2 = pd.read_csv('../datas/lgb60_month2_yval_pred')
m2.rename(columns={'y':'y4'},inplace=True)
ans = ans.merge(m2[['rowkey','y4']],on='rowkey',how='left')



ans['sum'] = (ans['y1'] )

ans = ans[['rowkey','id','is_risk','sum']]
ans = ans.sort_values('sum',ascending=False).reset_index(drop=True)
print ans.head(50)

print ans.loc[50:100]
