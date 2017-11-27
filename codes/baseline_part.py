# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login.csv')
t_login = t_login[t_login['time']>='2015-03-01 00:00:00']

t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-04-01 00:00:00']

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())

t_login = t_login.sort_values('time').reset_index(drop=True)
# t_login['time'] = t_login['time'].astype(pd.Timestamp)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)

t_login = t_login.sort_values('timestamp') \
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


def baseline_3(trade):
    # TODO 交易自身的特征,之前的交易次数,时间差，时段分布
    # TODO 交叉特征

    rowkey, id, timestamp, is_risk, time = trade
    res = {}
    res['rowkey'] = rowkey
    res['id'] = id
    res['is_risk'] = is_risk

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    res['hour_v'] = res['hour']/6                                               # 时间段
    t = t.date()
    res['weekday'] = t.weekday()                                                # 周信息

    d = t_trade[t_trade['trade_stamp']<timestamp]
    res['trade_cnt'] = d.shape[0]                                               # 交易次数
    res['trade_weekday_cnt'] = d[d['weekday']==res['weekday']].shape[0]         # 同一个周几交易次数
    res['trade_weekday_cnt_rate'] = 1.0 * res['trade_weekday_cnt'] / (1 + res['trade_cnt'])
    res['hour_v_cnt'] = d[d['hour_v'] == res['hour_v']].shape[0]                # 同一个时间段交易次数
    res['hour_v_cnt_rate'] = 1.0 * res['hour_v_cnt'] / (1 + res['trade_cnt'])

    print res
    return res


def baseline_2(idx):
    # TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
    # TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
    # TODO 交易自身的特征,之前的交易次数
    # TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数

    res = {}
    d = t_login.loc[:idx]
    idx += 1
    res['idx'] = idx
    timestamp = t_login.loc[idx]['timestamp']
    # TODO ip 在之前时间内，登陆的次数，登陆的用户数，登陆的时间长度，登陆的城市数，登陆的成功次数
    # TODO id 在之前的时间内，登陆的 ip 次数，device 次数，登陆时间长度，城市数，登陆成功次数
    # TODO ip,id,device,type,city 以及这些的组合，
    # TODO 特征之间的交叉特征，
    for ci in ['id']:
        idata = t_login.loc[idx][ci]
        dci = d[d[ci] == idata]
        dci = dci.sort_values('timestamp', ascending=False) \
                .reset_index(drop=True)
        for i in [0, 1, 2]:
            try:
                res[ci + '_diff{0}'.format(i + 1)] = timestamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
            except:
                res[ci + '_diff{0}'.format(i + 1)] = None
        res[ci + '_time_max'] = dci['timelong'].max()                   # id 之前登陆的 timelong 最大
        res[ci + '_time_min'] = dci['timelong'].min()                   # id 之前登陆的 timelong 最小
        res[ci + '_time_mean'] = dci['timelong'].mean()                 # id 之前登陆的 timelong 均值
        res[ci + '_cnt'] = dci.shape[0]                                 # id 之前登陆的次数
        res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]           # id 之前登陆失败的次数
        res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])

        res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]           # id 之前登陆成功的次数
        res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

        res[ci + '_ips'] = dci['ip'].unique().size                      # id 之前登陆的 ip 数
        res[ci + '_devices'] = dci['device'].unique().size              # id 之前登陆的 device 数
        res[ci + '_devices'] = dci['city'].unique().size                # id 之前登陆的 city 数
        res[ci + '_type'] = dci['type'].unique().size                   # id 之前登陆的 type 数
        res[ci + '_from'] = dci['log_from'].unique().size               # id 之前登陆的 log_from 数
        res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]     # id 之前安全插件登陆次数
        res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]    # id 之前非安全插件登陆次数
        res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]     # id 之前扫码登陆次数
        res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]    # id 之前扫码登陆次数
        res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])

    print res
    return res


def baseline_1(idx):
    # TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
    # TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
    # TODO 交易自身的特征,之前的交易次数
    # TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数

    res = {}
    d = t_login.loc[:idx-1]
    res['idx'] = idx
    timestamp = t_login.loc[idx]['timestamp']
    # TODO ip 在之前时间内，登陆的次数，登陆的用户数，登陆的时间长度，登陆的城市数，登陆的成功次数
    # TODO id 在之前的时间内，登陆的 ip 次数，device 次数，登陆时间长度，城市数，登陆成功次数
    # TODO ip,id,device,type,city 以及这些的组合，


    for ci in ['ip','device','id']:
        idata = t_login.loc[idx][ci]
        dci = d[d[ci] == idata]
        dci = dci.sort_values('timestamp', ascending=False) \
                .reset_index(drop=True)
        for i in [0, 1, 2]:
            try:
                res[ci + '_diff{0}'.format(i + 1)] = timestamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
            except:
                res[ci + '_diff{0}'.format(i + 1)] = None
        res[ci + '_time_max'] = dci['timelong'].max()                   # ip 之前登陆的 timelong 最大
        res[ci + '_time_min'] = dci['timelong'].min()                   # ip 之前登陆的 timelong 最小
        res[ci + '_time_mean'] = dci['timelong'].mean()                 # ip 之前登陆的 timelong 均值
        res[ci + '_cnt'] = dci.shape[0]                                 # ip 之前登陆的次数
        res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]           # ip 之前登陆失败的次数
        res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]           # ip 之前登陆成功的次数
        res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])
        res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

        for ii in ['id','ip','device','type','city']:
            if ii != ci:
                res[ci + "_" + ii] = dci[ii].unique().size              # ip 之前登陆的 id 数
        res[ci + '_from'] = dci['log_from'].unique().size               # ip 之前登陆的 log_from 数
        res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]     # ip 之前安全插件登陆次数
        res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]    # ip 之前非安全插件登陆次数
        res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]     # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]    # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])

    for ci in [('id', 'ip'),('id', 'device'),('id', 'type'),
               ('ip', 'device'),('ip', 'type'),('device', 'type'),('id','city')]:
        idata,jdata = t_login.loc[idx][ci[0]],t_login.loc[idx][ci[1]]
        dci = d[((d[ci[0]] == idata)&(d[ci[1]] == jdata))]
        ci = ci[0] + ci[1]
        dci = dci.sort_values('timestamp', ascending=False) \
                .reset_index(drop=True)
        for i in [0, 1, 2]:
            try:
                res[ci + '_diff{0}'.format(i + 1)] = timestamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
            except:
                res[ci + '_diff{0}'.format(i + 1)] = None
        res[ci + '_time_max'] = dci['timelong'].max()                   # ip 之前登陆的 timelong 最大
        res[ci + '_time_min'] = dci['timelong'].min()                   # ip 之前登陆的 timelong 最小
        res[ci + '_time_mean'] = dci['timelong'].mean()                 # ip 之前登陆的 timelong 均值
        res[ci + '_cnt'] = dci.shape[0]                                 # ip 之前登陆的次数
        res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]           # ip 之前登陆失败的次数
        res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]           # ip 之前登陆成功的次数
        res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])
        res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

        for ii in ['id','ip','device','type','city']:
            if ii not in ci:
                res[ci + "_" + ii] = dci[ii].unique().size              # ip 之前登陆的 id 数
        res[ci + '_from'] = dci['log_from'].unique().size               # ip 之前登陆的 log_from 数
        res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]     # ip 之前安全插件登陆次数
        res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]    # ip 之前非安全插件登陆次数
        res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]     # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]    # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])
    del d,dci
    print res
    return res

def baseline(trade):
    rowkey, id, timestamp, is_risk, time = trade
    res = {}
    res['rowkey'] = rowkey
    res['is_risk'] = is_risk

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


#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_login[t_login['time']>='2015-04-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt,t_trade

# 如果最近登陆统计存在
last_f = '../datas/baseline_1part456'
if os.path.exists(last_f):
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
