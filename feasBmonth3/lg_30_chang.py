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

# TODO 重复的 device，ip 数量
# TODO ip，device，type，result 变化的次数，时间间隔
# TODO 登陆时间差小于 20 的次数，result !=1 的次数，以及时间差
# TODO 连续 3 天登陆，相同 log_from 之间的时间差，前 1，2，3 天的登陆记录数
# TODO feas3month1.py trade_login_time_diff1 需要考虑 log_from=2 多次重复登陆情况
# TODO 当天内的 timediff,log_from=2,diff<=20 个数
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

    res['result_not_1'] = d[d['result']!=1].shape[0]
    res['days3cnt'] = 0
    for i in [1,2,3]:
        start_str = str(t - pd.Timedelta(days=i))
        end_str = str(t - pd.Timedelta(days=i+1))
        res['day{0}cnt'] = t_login[(t_login['id'] == res['id']) &
                    (t_login['time'] <= start_str) &
                    (t_login['time'] > end_str)].shape[0]
        res['days3cnt'] += res['day{0}cnt']


    d['login_time_diff'] = d['timestamp'].diff()
    res['login_time_diff_lt20'] = d[d['login_time_diff']<=20].shape[0]
    d['login_time_diff'] = d['login_time_diff'].shift(-1)

    d = d[~(d['login_time_diff']<=20)]
    res['login_cnt20'] = d.shape[0]

    res['duplicate_ip'] = res['login_cnt20'] - d['ip'].unique().size
    res['duplicate_device'] = res['login_cnt20'] - d['device'].unique().size

    login_data1 = d[(d['result'] == 1)] \
        .sort_values('time', ascending=False) \
        .reset_index(drop=False)

    login_data1['diff'] = login_data1['timestamp'].diff(-1)
    res['login_time_diff_mean'] = np.mean(login_data1['diff'])
    res['login_time_diff_max'] = np.max(login_data1['diff'])
    res['login_time_diff_min'] = np.min(login_data1['diff'])

    d1nn = login_data1.shape[0]
    try:
        login = login_data1.loc[0]  # 最近一条登陆信息,可能没有最近一条登陆信息
        login_time, lg_time = login['timestamp'], login['time']

        res['trade_login_time_diff1'] = login_time - login_data1.loc[1]['timestamp']
        res['trade_login_time_diff2'] = login_time - login_data1.loc[2]['timestamp']
        res['login_all_diff'] = login_time - login_data1.loc[d1nn-1]['timestamp']

    except:
        res['trade_login_time_diff1'] = None
        res['trade_login_time_diff2'] = None
        res['login_all_diff'] = None


    # 同一个 log_from 的登陆时间差
    for i in [0,1,2]:
        try:
            log_from,tstamp = login_data1.loc[d1nn-1-i]['log_from'],login_data1.loc[d1nn-1-i]['timestamp']
            d1 = login_data1[(login_data1['log_from']==log_from)&(login_data1['timestamp']<tstamp)].reset_index(drop=True)
            d1n = d1.shape[0]
            try:
                res['time_diff_{0}_0'.format(i)] = d1.loc[0]['timestamp'] - d1.loc[1]['timestamp']
            except:
                res['time_diff_{0}_0'.format(i)] = None

            try:
                res['time_diff_{0}_1'.format(i)] = d1.loc[1]['timestamp'] - d1.loc[2]['timestamp']
            except:
                res['time_diff_{0}_1'.format(i)] = None
        except:
            res['time_diff_{0}_0'.format(i)] = None
            res['time_diff_{0}_1'.format(i)] = None

    print res
    return res

#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_trade[t_trade['time']>='2015-03-01 00:00:00']
#dtt = dtt[dtt['time']<'2015-04-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt
"""
for i in t_trade_list[:10]:
    baseline_1(i)
"""
# 如果最近登陆统计存在
last_f = '../datas/feasb3_all'
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
