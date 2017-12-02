# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


# 数据合并
login_train = pd.read_csv('../datas/baseline_feas_train1')
login_train = login_train.drop('is_risk',axis=1)
trade_train = pd.read_csv('../datas/trade_baseline_3_train')
data_train = pd.merge(login_train,trade_train,left_on='rowkey',right_on='rowkey')
data_train.to_csv('../datas/login_trade_train',index=None)

login_test = pd.read_csv('../datas/baseline_feas_test')
login_test = login_test.drop('is_risk',axis=1)
trade_test = pd.read_csv('../datas/trade_baseline_3_test')
data_test = pd.merge(login_test,trade_test,left_on='rowkey',right_on='rowkey')
data_test.to_csv('../datas/login_trade_test',index=None)



