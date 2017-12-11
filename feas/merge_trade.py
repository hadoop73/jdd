# coding:utf-8

import pandas as pd
import numpy as np
import os,sys



t_login = pd.read_csv('../datas/t_login.csv')
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)

# TODO 7 月份的数据之前已经做好了的



d4 = pd.read_csv('../datas/baseline_month_trade4')
d5 = pd.read_csv('../datas/baseline_month_trade5')
d6 = pd.read_csv('../datas/baseline_month_trade6')

print d4.head()

print d4.shape,d5.shape
d = pd.concat([d4,d5])

print d.shape

d.to_csv('../datas/baseline_login_train',index=None)


d6.to_csv('../datas/baseline_login_val',index=None)


d = pd.concat([d,d6])
d.to_csv('../datas/baseline_login_train_all',index=None)

# 从 baseline_feas.ipynb 生成
test = pd.read_csv('../datas/baseline_feas')
test.to_csv('../datas/baseline_login_test',index=None)

