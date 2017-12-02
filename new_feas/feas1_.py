# coding:utf-8

import pandas as pd

df_train = pd.read_csv('../datas/feas_new')  # 0.6 filter



# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-03-01 00:00:00']  # 只需要从 3-7 开始分析

t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade,t_trade_test])
del t_trade_test


df_train = df_train.merge(t_trade[['rowkey','time']],on='rowkey')


print df_train.head()
df_train.to_csv('../datas/feas_new_time',index=None)




