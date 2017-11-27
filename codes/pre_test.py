# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login_test.csv')
t_trade = pd.read_csv('../datas/t_trade_test.csv')



def ftrade(trade):
    rowkey, id, timestamp = trade
    data = t_login[t_login['result'] == 1]
    # 由于登陆存在登陆用时所以需要根据时间戳进行判断
    d = data[data['id']==id][data['timestamp_online']<timestamp]
    res = {}
    res['rowkey'] = rowkey
    res['p_count'] = d.shape[0]
    data = t_login[t_login['id']==id][t_login['timestamp_online']<timestamp]
    res['all_count'] = data.shape[0]
    print res
    return res

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp']]).tolist()
pool = Pool(8)
d = pool.map(ftrade,t_trade_list)
pool.close()
pool.join()
df = pd.DataFrame(d)
print(df.head())
df.to_csv('../datas/pre_test.csv',index=None)


