# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login.csv')
t_login = t_login[t_login['time']>='2015-03-01 00:00:00']

t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = pd.concat([t_login,t_login_test])

t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-03-01 00:00:00']
t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade,t_trade_test])

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())
t_trade = t_trade.sort_values('time') \
                .reset_index(drop=True)

# t_login['time'] = t_login['time'].astype(pd.Timestamp)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']

t_login = t_login.sort_values('time') \
                .reset_index(drop=True)

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


# TODO 最近 3 次登陆的 ip,log_from,device 不重复次数,除去最近短时间多次登陆情况
# TODO 最近一次登陆的 log_from

def baseline_1(idx):
    # 2天内登陆与2天以为的登陆时间存在 gap ,当天 1h 内多次登陆记录，log_from 变动，存在登陆中断
    # 两天内无其他登陆，只有这些登陆，可以剔除掉 log_from=2,type=1  对登陆的记录，再统计记录数
    # 存在 log_from=1,type=3 的多次登陆，可能由于系统或者其他原因，log_from=2,type=1 多次登陆并不应该算是特征了
    # 过去两天内没有交易记录

    res = {}
    ida = t_trade.loc[idx]
    res['rowkey'] = ida['rowkey']

    t = pd.Timestamp(ida['time'])
    days2str = str(t - pd.Timedelta(days=2))

    d = t_login[(t_login['id'] == ida['id']) & (t_login['result'] == 1)&
                         (t_login['time'] >= days2str)&
                         (t_login['time'] < ida['time'])]

    d = d.sort_values('time',ascending=False)
    d['time_diff'] = d['timestamp'].diff(-1)
    d['time_diff'] = d['time_diff'].shift()
    d = d[~(d['time_diff']<=20)].reset_index(drop=True)

    try:
        res['log_from_now'] = d.loc[0]['log_from']
    except:
        res['log_from_now'] = None

    try:
        res['log_from_now1'] = d.loc[1]['log_from']
    except:
        res['log_from_now1'] = None

    res['log_from_u_n'] = d.loc[:1]['log_from'].unique().size
    res['device_u_n'] = d.loc[:1]['device'].unique().size
    res['ip_u_n'] = d.loc[:1]['ip'].unique().size
    res['city_u_n'] = d.loc[:1]['city'].unique().size
    print res
    return res

#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_trade[t_trade['time']>='2015-03-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt
"""
for i in t_trade_list[:10]:
    baseline_1(i)
"""
# 如果最近登陆统计存在
last_f = '../datas/feas_a'
if os.path.exists(last_f) and False:
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    d = pool.map(baseline_1,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)
