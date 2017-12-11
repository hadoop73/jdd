# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login.csv')

t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = pd.concat([t_login,t_login_test])

t_trade = pd.read_csv('../datas/t_trade.csv')
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


def baseline_1(idx):
    # 2天内登陆与2天以为的登陆时间存在 gap ,当天 1h 内多次登陆记录，log_from 变动，存在登陆中断
    # 两天内无其他登陆，只有这些登陆，可以剔除掉 log_from=2,type=1  对登陆的记录，再统计记录数
    # 存在 log_from=1,type=3 的多次登陆，可能由于系统或者其他原因，log_from=2,type=1 多次登陆并不应该算是特征了
    # 过去两天内没有交易记录

    res = {}
    ida = t_trade.loc[idx]
    res['rowkey'] = ida['rowkey']
    res['is_risk'] = ida['is_risk']
    res['id'] = ida['id']
    res['time'] = ida['time']
    t = pd.Timestamp(res['time'])
    days2str = str(t - pd.Timedelta(days=2))

    d = t_login[(t_login['id'] == res['id']) &
                         (t_login['time'] >= days2str)&
                         (t_login['time'] < res['time'])]

    d = d.sort_values('time') \
        .reset_index(drop=True)

    dpre2 = t_login[(t_login['id'] == res['id']) &
                (t_login['time'] < days2str)]

    dpre2 = dpre2.sort_values('time',ascending=False) \
        .reset_index(drop=True)

    td = d.sort_values('time',ascending=False).reset_index(drop=True)
    try:
        tm = td.loc[0]['time']
    except:
        tm = str(t - pd.Timedelta(hours=1))
    res['hour2_cnt'] = t_trade[(t_trade['time'] < tm)&(t_trade['time']>=days2str)].shape[0]
    res['all_cnt'] = t_trade[(t_trade['time'] > tm) & (t_trade['time'] < res['time'])].shape[0]


    try:
        res['days2_time_diff'] = d.loc[0]['timestamp'] - dpre2.loc[0]['timestamp']
    except:
        res['days2_time_diff'] = 1e6

    res['log_from_size_u'] = d['log_from'].unique().size
    res['is_scan_size_u'] = d['is_scan'].unique().size
    res['ip_size_u'] = d['ip'].unique().size

    res['login2daycnt'] = d.shape[0]
    res['login_ip_rate'] = 1.0*res['login2daycnt']/(1 + res['ip_size_u'] )

    res['login2daycnt0'] = d.shape[0] == 0 and 1 or 0

    d['login_time_diff'] = d['timestamp'].diff()
    d['login_time_diff'] = d['login_time_diff'].fillna(1e6)

    d60 = d[d['login_time_diff']>60]
    res['login2day_valid_cnt'] = d60.shape[0]

    d2 = d[d['log_from'] == 1].sort_values('time')
    d2['login_time_diff'] = d2['timestamp'].diff()
    d2['login_time_diff'] = d2['login_time_diff'].fillna(1e6)
    res['login2day_log_from1'] = d2[d2['login_time_diff'] > 60].shape[0]

    d2 = d[d['log_from'] == 2].sort_values('time')
    d2['login_time_diff'] = d2['timestamp'].diff()
    d2['login_time_diff'] = d2['login_time_diff'].fillna(1e6)
    res['login2day_log_from2'] = d2[d2['login_time_diff'] > 60].shape[0]

    d2 = d[d['log_from'] != 1].sort_values('time')
    d2['login_time_diff'] = d2['timestamp'].diff()
    d2['login_time_diff'] = d2['login_time_diff'].fillna(1e6)
    res['login2day_log_from0'] = d2[d2['login_time_diff'] > 60].shape[0]

    print res
    return res

#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_trade[t_trade['time']>='2015-03-01 00:00:00']
dtt = dtt[dtt['time']<'2015-04-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt
"""
for i in t_trade_list[:10]:
    baseline_1(i)
"""
# 如果最近登陆统计存在
last_f = '../datas/feasb1_month3'
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
