# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login_test.csv')
t_trade = pd.read_csv('../datas/t_trade_test.csv')
t_trade['is_risk'] = -1

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)

t_login = t_login.sort_values('time').reset_index(drop=True)
# t_login['time'] = t_login['time'].astype(pd.Timestamp)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)


def baseline(trade):
    rowkey, id, timestamp, is_risk, time = trade
    res = {}
    res['rowkey'] = rowkey
    res['is_risk'] = is_risk
    res['time'] = time

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    t = t.date()
    res['weekday'] = t.weekday()

    d_trade = t_trade[t_trade['id'] == id][t_trade['trade_stamp'] < timestamp]
    res['trade_cnt'] = d_trade.shape[0]

    # TODO 案例分析正样本的登陆结果
    data = t_login[t_login['result'] == 1]
    # 由于登陆存在登陆用时所以需要根据时间戳进行判断
    d = data[data['id']==id][data['timestamp_online']<timestamp] \
        .sort_values('time', ascending=False) \
        .reset_index()

    # 最近一次的登陆所用时间
    for i in [0,1,2]:
        try:
            res['timelong_{0}'.format(i + 1)] = d.loc[i]['timelong']
            res['time_diff_{0}'.format(i + 1)] = timestamp - d.loc[i]['timestamp_online']
            res['log_from_{0}'.format(i + 1)] = d.loc[i]['log_from']
            res['city_{0}'.format(i + 1)] = d.loc[i]['city']
            res['result_{0}'.format(i + 1)] = d.loc[i]['result']
            res['type_{0}'.format(i + 1)] = d.loc[i]['type']
            res['is_scan_{0}'.format(i + 1)] = d.loc[i]['is_scan']
            res['is_sec_{0}'.format(i + 1)] = d.loc[i]['is_sec']
            res['time_{0}_hour'.format(i + 1)] = pd.Timestamp(d.loc[i]['time']).hour
        except:
            res['timelong_{0}'.format(i + 1)] = None
            res['time_diff_{0}'.format(i + 1)] = None
            res['log_from_{0}'.format(i + 1)] = None
            res['city_{0}'.format(i + 1)] = None
            res['result_{0}'.format(i + 1)] = None
            res['type_{0}'.format(i + 1)] = None
            res['is_scan_{0}'.format(i + 1)] = None
            res['is_sec_{0}'.format(i + 1)] = None
            res['time_{0}_hour'.format(i + 1)] = None

    try:
        res['ip_eq'] = (d.loc[0]['ip'] == d.loc[1]['ip']) and 1 or 0
    except:
        res['ip_eq'] = None

    try:
        res['device_eq'] = (d.loc[0]['device'] == d.loc[1]['device']) and 1 or 0
    except:
        res['device_eq'] = None

    for ci in ['ip','device','city','is_sec','is_scan']:
        try:
            res['last_3{0}_cnt'.format(ci)] = d.loc[:3][ci].unique().size
        except:
            res['last_3{0}_cnt'.format(ci)] = None

    print res
    return  res


t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()


# 如果最近登陆统计存在
last_f = '../datas/baseline_test'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(12)
    d = pool.map(baseline,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)
