# coding:utf-8

import pandas as pd
import numpy as np
import os,sys



t_login = pd.read_csv('../datas/t_login.csv')
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)

# TODO 7 月份的数据之前已经做好了的
for i in [2]:#[3,4,5,6]: # 4,5,6
    login_month = pd.read_csv('../datas/baseline_month{0}'.format(i))       # login 的前一个月统计特征
    login_month = login_month.set_index('idx')
    d = t_login[t_login['time'] >= '2015-0{0}-01 00:00:00'.format(i - 1)]
    d = d[d['time'] < '2015-0{0}-01 00:00:00'.format(i + 1)]                # 筛选前一个月和当前月
    d = d.sort_values('timestamp').reset_index(drop=True)
    d = d[d['time'] >= '2015-0{0}-01 00:00:00'.format(i)]                   # 取当月的 login 原数据
    d = d.merge(login_month,left_index=True,right_index=True)               # 通过 idx 和 index 来拼接
    d.to_csv('../datas/baseline_login{0}'.format(i),index=None)




