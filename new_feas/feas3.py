# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


# TODO 统计每个月的每条交易最近一次，两次登陆的 ip,device 等环境信息

# TODO 合并 login 的 train,test 数据
t_login = pd.read_csv('../datas/t_login.csv')
t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = t_login[t_login['time']>='2015-03-01 00:00:00']  # 只考虑 3-7 月的登陆信息，也就是 trade 只往前看一个月
t_login = pd.concat([t_login,t_login_test])
del t_login_test

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login = t_login.sort_values('timestamp') \
                .reset_index(drop=True)

# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-03-01 00:00:00']  # 只需要从 3-7 开始分析

t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade,t_trade_test])
del t_trade_test

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())
t_trade = t_trade.sort_values('trade_stamp') \
                .reset_index(drop=True)



def baseline_new(idx):
    # TODO 每一条交易记录
    pass



# 如果最近登陆统计存在

dtt = t_trade[t_trade['time']>='2015-04-01 00:00:00']    # 只用 4,5,6,7 做特征进行 train,valiade,test
t_trade_list = dtt.index.tolist()
del dtt

last_f = '../datas/feas_new_3'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    df = pool.map(baseline_new,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(df)
    data.to_csv(last_f,index=None)




