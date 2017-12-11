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

    login = t_login[(t_login['id'] == res['id']) &
                         (t_login['time'] >= days2str)&
                         (t_login['time'] < res['time'])]


    res['timelong_max'] = login['timelong'].max()
    res['idcnt'] = login.shape[0]
    login1 = login[login['result']==1]
    res['fail_cnt'] = res['idcnt'] - login1.shape[0]
    login1 = login1.sort_values('time').reset_index(drop=True)

    login1['time_diff'] = login1['timestamp'].diff()
    login1 = login1[login1['time_diff'] > 20].reset_index(drop=True)
    d1nn = login1.shape[0]
    try:
        res['timelong0_max'] = login1.loc[d1nn-1]['timelong']
        res['time_all'] = login1.loc[d1nn-1]['timestamp'] - login1.loc[0]['timestamp']
    except:
        res['timelong0_max'] = None
        res['time_all'] = None

    for i in [0,1]:
        try:
            res['time_1_diff{0}'.format(i)] = login1.loc[d1nn-i]['timestamp'] - login1.loc[d1nn-1-i]['timestamp']
        except:
            res['time_1_diff{0}'.format(i)] = None

    for ci in ['ip', 'device']:
        try:
            res[ci+'cnt'] = login1[ci].unique().size
            res[ci + 'rate'] = 1.0*d1nn/(1+res[ci+'cnt'])
        except:
            res[ci + 'cnt'] = None
            res[ci + 'rate'] = None

    try:
        res['ip_device_rate'] = res['iprate']*res['devicerate']
    except:
        res['ip_device_rate'] = None

    for ci in ['ip','device']:
            for i in [0,1,2]:
                try:
                    dci,log_from = login1.loc[d1nn-1-i][ci],login1.loc[d1nn-1-i]['log_from']
                    d1 = login1[(login1[ci]==dci)&(login1['log_from']==log_from)]
                    d1['time_diff'] = d1['timestamp'].diff()
                    d1 = d1[d1['time_diff']>20]
                    res[ci + '_log_from_time_diff_min{0}'.format(i)] = d1['time_diff'].min()
                    res[ci + '_log_from_time_diff_max{0}'.format(i)] = d1['time_diff'].max()
                    res[ci + '_log_from_time_diff_mean{0}'.format(i)] = d1['time_diff'].mean()
                except:
                    res[ci + '_log_from_time_diff_min{0}'.format(i)] = None
                    res[ci + '_log_from_time_diff_max{0}'.format(i)] = None
                    res[ci + '_log_from_time_diff_mean{0}'.format(i)] = None

    print res
    return res

#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_trade[t_trade['time']>='2015-04-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt
"""
for i in t_trade_list[:10]:
    baseline_1(i)
"""
# 如果最近登陆统计存在
last_f = '../datas/feasb2'
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
