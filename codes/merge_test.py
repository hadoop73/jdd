# coding:utf-8

import pandas as pd

xgb100 = pd.read_csv('../datas/xgb_40_filter_ytest_pred')
adboost80 = pd.read_csv('../datas/adboost_80_filter_ytest_pred')
lgb80 = pd.read_csv('../datas/lgb80_filter_ytest_pred')


bs = set(xgb100[xgb100['y']>=0.761551856995]['rowkey'].values.tolist() + \
    adboost80[adboost80['y']>=0.511255620841]['rowkey'].values.tolist() + \
    lgb80[lgb80['y']>=0.805510113839]['rowkey'].values.tolist())

print len(bs)



