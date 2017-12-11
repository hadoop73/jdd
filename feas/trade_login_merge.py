# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool

trade3 = pd.read_csv('../datas/baseline_month_trade3')
trade4 = pd.read_csv('../datas/baseline_month_trade4')
trade5 = pd.read_csv('../datas/baseline_month_trade5')
trade6 = pd.read_csv('../datas/baseline_month_trade6')
trade7 = pd.read_csv('../datas/baseline_month_trade7')

trade = pd.concat([trade4,trade5,trade6,trade7])

trade_login3 = pd.read_csv('../datas/trade_login31')
trade_login4 = pd.read_csv('../datas/trade_login41')
trade_login5 = pd.read_csv('../datas/trade_login51')
trade_login6 = pd.read_csv('../datas/trade_login61')
trade_login7 = pd.read_csv('../datas/trade_login71')

trade_login = pd.concat([trade_login4,trade_login5,trade_login6,trade_login7])

trcols = trade.columns
print trcols
tcols = trade_login.columns
print tcols

cs = [ci for ci in trcols if ci in tcols]
cs = [ci for ci in cs if ci not in ['rowkey','time']]

print cs
trade_login = trade_login.drop(cs,axis=1)
#trade_login = trade_login.drop('time',axis=1)

data = trade.merge(trade_login,on='rowkey')

print data.shape
#data.to_csv('../datas/all1.csv',index=None)  # 读取 trade_login31 后缀带 1 的文件
data.to_csv('../datas/all3.csv',index=None)   # 读取 trade_login3 后缀没有带 1 的文件
