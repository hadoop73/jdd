# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool



t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-05-01 00:00:00']


t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())

t_trade = t_trade.sort_values('trade_stamp') \
                .reset_index(drop=True)

# t_login['time'] = t_login['time'].astype(pd.Timestamp)


"""
log_id
timelong
device
log_from
ip
city
result
timestamp
type
id
is_scan
is_sec
time
"""


def baseline_3(idx):
    # TODO 交易自身的特征,之前的交易次数,时间差，时段分布
    # TODO 交叉特征

    id = t_trade.loc[idx]['id']
    time = t_trade.loc[idx]['time']
    timestamp = t_trade.loc[idx]['trade_stamp']
    res = {}
    res['time'] = time
    res['trade_stamp'] = t_trade.loc[idx]['trade_stamp']
    res['rowkey'] = t_trade.loc[idx]['rowkey']
    res['id'] = id
    res['is_risk'] = t_trade.loc[idx]['is_risk']

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    res['hour_v'] = res['hour']/6                                               # 时间段
    t = t.date()
    res['weekday'] = t.weekday()                                                # 周信息

    d = t_trade[t_trade['id']==id][t_trade['trade_stamp']<timestamp]
    res['trade_cnt'] = d.shape[0]                                               # 交易次数
    res['trade_weekday_cnt'] = d[d['weekday']==res['weekday']].shape[0]         # 同一个周几交易次数
    res['trade_weekday_cnt_rate'] = 1.0 * res['trade_weekday_cnt'] / (1 + res['trade_cnt'])
    res['hour_v_cnt'] = d[d['hour_v'] == res['hour_v']].shape[0]                # 同一个时间段交易次数
    res['hour_v_cnt_rate'] = 1.0 * res['hour_v_cnt'] / (1 + res['trade_cnt'])

    print res
    return res

dtt = t_trade[t_trade['time']>='2015-06-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt

# 如果最近登陆统计存在
#last_f = '../datas/baseline_1part_test'
#last_f = '../datas/trade_baseline_3_train'

last_f = '../datas/baseline_month_trade6'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    d = pool.map(baseline_3,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)
