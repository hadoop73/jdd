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
    # TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
    # TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
    # TODO 交易自身的特征,之前的交易次数
    # TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数

    res = {}
    ida = t_trade.loc[idx]
    res['rowkey'] = ida['rowkey']
    res['is_risk'] = ida['is_risk']
    res['id'] = ida['id']
    res['time'] = ida['time']
    trade_stamp = ida['trade_stamp']
    t = pd.Timestamp(res['time'])
    days30str = str(t - pd.Timedelta(days=60))

    d = t_login[(t_login['timestamp'] < trade_stamp) &
                         (t_login['time'] >= days30str)]

    # TODO ip 在之前时间内，登陆的次数，登陆的用户数，登陆的时间长度，登陆的城市数，登陆的成功次数
    # TODO id 在之前的时间内，登陆的 ip 次数，device 次数，登陆时间长度，城市数，登陆成功次数
    # TODO ip,id,device,type,city 以及这些的组合，

    login_data = d[(d['id']==res['id'])&(d['result'] == 1)] \
        .sort_values('time', ascending=False) \
        .reset_index(drop=True)

    for ci in ['ip','device','id']:
        try:
            idata = login_data.loc[0][ci]
            dci = d[d[ci] == idata]
            dci = dci.sort_values('time', ascending=False) \
                    .reset_index(drop=True)
            for i in [0, 1, 2]:
                try:
                    res[ci + '_diff{0}'.format(i + 1)] = trade_stamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
                except:
                    res[ci + '_diff{0}'.format(i + 1)] = None
            res[ci + '_time_max'] = dci['timelong'].max()                   # ip 之前登陆的 timelong 最大
            # TODO 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
            res[ci + '_cnt'] = dci.shape[0]                                 # ip 之前登陆的次数
            res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]           # ip 之前登陆失败的次数
            res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]           # ip 之前登陆成功的次数
            res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])
            res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

            for ii in ['id','ip','device','type','city']:
                # TODO  ip,city 重复
                if ii != ci:
                    res[ci + "_" + ii] = dci[ii].unique().size              # ip 之前登陆的 id 数
            res[ci + '_from'] = dci['log_from'].unique().size               # ip 之前登陆的 log_from 数
            res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]     # ip 之前安全插件登陆次数
            res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]    # ip 之前非安全插件登陆次数
            res[ci + '_cnt_sec0rate'] = 1.0 * res[ci + '_cnt_sec0'] / (1 + res[ci + '_cnt'])

            res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]     # ip 之前扫码登陆次数
            res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]    # ip 之前扫码登陆次数
            res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])
        except:
            for i in [0, 1, 2]:
                    res[ci + '_diff{0}'.format(i + 1)] = None
            res[ci + '_time_max'] = None                                    # ip 之前登陆的 timelong 最大
            # TODO 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
            res[ci + '_cnt'] = None                                # ip 之前登陆的次数
            res[ci + '_cnt_0'] = None          # ip 之前登陆失败的次数
            res[ci + '_cnt_1'] = None           # ip 之前登陆成功的次数
            res[ci + '_cnt_rate'] = None
            res[ci + '_cnt_1rate'] = None

            for ii in ['id','ip','device','type','city']:
                # TODO  ip,city 重复
                if ii != ci:
                    res[ci + "_" + ii] = None                                  # ip 之前登陆的 id 数
            res[ci + '_from'] = None
            res[ci + '_cnt_sec1'] = None
            res[ci + '_cnt_sec0'] = None
            res[ci + '_cnt_sec0rate'] = None
            res[ci + '_cnt_scan1'] = None
            res[ci + '_cnt_scan0'] = None
            res[ci + '_cnt_scan0rate'] = None
    try:
        for ci in [('id', 'ip'),('id', 'device'),('id', 'type'),
                   ('ip', 'device'),('ip', 'type'),('device', 'type'),('id','city')]:
            idata,jdata = login_data.loc[0][ci[0]],login_data.loc[0][ci[1]]
            dci = d[((d[ci[0]] == idata)&(d[ci[1]] == jdata))]
            ci = ci[0] + ci[1]
            dci = dci.sort_values('time', ascending=False) \
                    .reset_index(drop=True)
            for i in [0, 1, 2]:
                try:
                    res[ci + '_diff{0}'.format(i + 1)] = trade_stamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
                except:
                    res[ci + '_diff{0}'.format(i + 1)] = None
            res[ci + '_time_max'] = dci['timelong'].max()                   # ip 之前登陆的 timelong 最大
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
    except:
        for ci in [('id', 'ip'), ('id', 'device'), ('id', 'type'),
                   ('ip', 'device'), ('ip', 'type'), ('device', 'type'), ('id', 'city')]:
            ci = ci[0] + ci[1]
            for i in [0, 1, 2]:
                res[ci + '_diff{0}'.format(i + 1)] = None
            res[ci + '_time_max'] = None
            res[ci + '_cnt'] = None
            res[ci + '_cnt_0'] = None
            res[ci + '_cnt_1'] = None
            res[ci + '_cnt_rate'] = None
            res[ci + '_cnt_1rate'] = None

            for ii in ['id', 'ip', 'device', 'type', 'city']:
                if ii not in ci:
                    res[ci + "_" + ii] = None
            res[ci + '_from'] = None
            res[ci + '_cnt_sec1'] = None
            res[ci + '_cnt_sec0'] = None
            res[ci + '_cnt_scan1'] = None
            res[ci + '_cnt_scan0'] = None
            res[ci + '_cnt_scan0rate'] = None
    print res
    return res

#t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()
dtt = t_trade[(t_trade['time']>='2015-03-01 00:00:00')&(t_trade['time']<'2015-04-01 00:00:00')]

t_trade_list = dtt.index.tolist()
del dtt
"""
for i in t_trade_list[:10]:
    baseline_1(i)
"""
# 如果最近登陆统计存在
last_f = '../datas/feas_trade_login_month3'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(4)
    d = pool.map(baseline_1,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)
