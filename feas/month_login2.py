# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login.csv')
t_login = t_login[t_login['time']>='2015-01-01 00:00:00']
t_login = t_login[t_login['time']<'2015-03-01 00:00:00']


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



#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_login[t_login['time']>='2015-02-01 00:00:00']

t_trade_list = dtt.index.tolist()
del dtt

# 如果最近登陆统计存在
last_f = '../datas/baseline_month2'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(12)
    d = pool.map(baseline_1,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)
