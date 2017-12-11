# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


t_login = pd.read_csv('../datas/t_login.csv')
t_login = t_login[t_login['time']>='2015-01-26 00:00:00']

t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = pd.concat([t_login,t_login_test])

t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-01-26 00:00:00']
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

    d = t_login[(t_login['time'] < res['time']) &
                         (t_login['time'] >= days30str)]

    # TODO ip 在之前时间内，登陆的次数，登陆的用户数，登陆的时间长度，登陆的城市数，登陆的成功次数
    # TODO id 在之前的时间内，登陆的 ip 次数，device 次数，登陆时间长度，城市数，登陆成功次数
    # TODO ip,id,device,type,city 以及这些的组合，

    login_data = d[(d['id'] == res['id'])] \
        .sort_values('time', ascending=True)

    login_data['login_time_diff'] = login_data['timestamp'].diff()

    login_data = login_data \
        .sort_values('time', ascending=False) \
        .reset_index(drop=False)


    res['month2_cnt'] = login_data.shape[0]

    login_data1 = login_data[(login_data['result'] == 1)] \
        .sort_values('time', ascending=False) \
        .reset_index(drop=False)

    tis = [60, 2 * 60, 30 * 60, 60 * 60, 24 * 60 * 60, 2 * 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60,
           60 * 24 * 60 * 60]
    sis = [0, 60, 2 * 60, 30 * 60, 60 * 60, 24 * 60 * 60, 2 * 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60]
    for si, ti in zip(sis, tis):
        try:
            login_time = login_data1.loc[0]['timestamp']

            login_tmp = login_data[
                (login_data['timestamp'] < login_time - si) & (login_data['timestamp'] >= login_time - ti)]
            res['timelong_{0}_max'.format(ti)] = login_tmp['timelong'].max()  # ip 之前登陆的 timelong 最大
            res['login_{0}_cnt'.format(ti)] = login_tmp.shape[0]

            res['login_time_diff_{0}_max'.format(ti)] = login_tmp['login_time_diff'].max()
            res['login_time_diff_{0}_min'.format(ti)] = login_tmp['login_time_diff'].min()
            res['login_time_diff_{0}_mean'.format(ti)] = login_tmp['login_time_diff'].mean()

            di = 60
            res['login_{0}_cnt{1}'.format(ti, di)] = login_tmp[login_tmp['login_time_diff'] <= di].shape[0]
            di = 180
            res['login_{0}_cnt{1}'.format(ti, di)] = \
                login_tmp[(login_tmp['login_time_diff'] <= di) & (login_tmp['login_time_diff'] > 60)].shape[0]
            di = 300
            res['login_{0}_cnt{1}'.format(ti, di)] = \
                login_tmp[(login_tmp['login_time_diff'] <= di) & (login_tmp['login_time_diff'] > 180)].shape[0]
            di = 900
            res['login_{0}_cnt{1}'.format(ti, di)] = \
                login_tmp[(login_tmp['login_time_diff'] <= di) & (login_tmp['login_time_diff'] > 300)].shape[0]
            di = 900
            res['login_{0}_cnt{1}'.format(ti, 901)] = \
                login_tmp[(login_tmp['login_time_diff'] > di)].shape[0]

            res['ip_{0}_cnt'.format(ti)] = login_tmp['ip'].unique().size
            res['type_{0}_cnt'.format(ti)] = login_tmp['type'].unique().size
            res['log_from_{0}_cnt'.format(ti)] = login_tmp['log_from'].unique().size
            res['result_{0}_cnt'.format(ti)] = login_tmp['result'].unique().size

            res['iptype_{0}_rate'.format(ti)] = 1.0*res['type_{0}_cnt'.format(ti)]/(1+res['ip_{0}_cnt'.format(ti)])
            res['iplog_from_{0}_rate'.format(ti)] = 1.0*res['log_from_{0}_cnt'.format(ti)]/(1+res['ip_{0}_cnt'.format(ti)])
            res['ipresult_{0}_rate'.format(ti)] = 1.0*res['result_{0}_cnt'.format(ti)]/(1+res['ip_{0}_cnt'.format(ti)])
            res['iplogin_{0}_rate'.format(ti)] = 1.0*res['login_{0}_cnt'.format(ti)]/(1+res['ip_{0}_cnt'.format(ti)])

        except:
            res['timelong_{0}_max'.format(ti)] = None  # ip 之前登陆的 timelong 最大
            res['login_{0}_cnt'.format(ti)] = 0


            res['login_time_diff_{0}_max'.format(ti)] = None
            res['login_time_diff_{0}_min'.format(ti)] = None
            res['login_time_diff_{0}_mean'.format(ti)] = None

            di = 60
            res['login_{0}_cnt{1}'.format(ti, di)] = 0
            di = 180
            res['login_{0}_cnt{1}'.format(ti, di)] = 0
            di = 300
            res['login_{0}_cnt{1}'.format(ti, di)] = 0
            di = 900
            res['login_{0}_cnt{1}'.format(ti, di)] = 0
            res['login_{0}_cnt{1}'.format(ti, 901)] = 0

            res['ip_{0}_cnt'.format(ti)] = 0
            res['type_{0}_cnt'.format(ti)] = 0
            res['log_from_{0}_cnt'.format(ti)] = 0
            res['result_{0}_cnt'.format(ti)] = 0

            res['iptype_{0}_rate'.format(ti)] = 0
            res['iplog_from_{0}_rate'.format(ti)] = 0
            res['ipresult_{0}_rate'.format(ti)] = 0
            res['iplogin_{0}_rate'.format(ti)] = 0

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
            res[ci + '_{0}_max'.format(ti)] = dci.loc[:2]['timelong'].max()  # ip 之前登陆的 timelong 最大
            res[ci + '_{0}_cnt'.format(ti)] = dci.shape[0]
            for ii in ['id','ip','device','type','city']:
                # TODO  ip,city 重复
                if ii != ci:
                    res[ci + "_" + ii+"{0}".format(ti)] = dci[ii].unique().size              # ip 之前登陆的 id 数
        except:
            for i in [1, 2]:
                res[ci + '_diff{0}'.format(i + 1)] = None
            res[ci + '_{0}_max'.format(ti)] = None                                                 # ip 之前登陆的 timelong 最大
            res[ci + '_{0}_cnt'.format(ti)] = 0
            for ii in ['id', 'ip', 'device', 'type', 'city']:
                # TODO  ip,city 重复
                if ii != ci:
                    res[ci + "_" + ii + "{0}".format(ti)] = 0  # ip 之前登陆的 id 数

    for ci in [('id', 'ip'), ('id', 'device'), ('id', 'type'),
                                   ('ip', 'device'), ('ip', 'type'), ('id', 'city')]:
        cis = ci[0] + ci[1]
        try:
            idx = login_data1.loc[0]['index']
            idata, jdata = login_data.loc[idx][ci[0]], login_data.loc[idx][ci[1]]

            dci = d[((d[ci[0]] == idata) & (d[ci[1]] == jdata))]

            dci = dci.sort_values('time', ascending=False) \
                .reset_index(drop=True)
            for i in [1, 2]:
                try:
                    res[cis + '_diff{0}'.format(i + 1)] = trade_stamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
                except:
                    res[cis + '_diff{0}'.format(i + 1)] = None
            res[cis + '_{0}_max'.format(ti)] = dci.loc[:2]['timelong'].max()  # ip 之前登陆的 timelong 最大
            res[cis + '_{0}_cnt'.format(ti)] = dci.shape[0]
            for ii in ['id', 'ip', 'device', 'type', 'city']:
                # TODO  ip,city 重复
                if ii != cis:
                    res[cis + "_" + ii + "{0}".format(ti)] = dci[ii].unique().size  # ip 之前登陆的 id 数
        except:
            for i in [1, 2]:
                res[cis + '_diff{0}'.format(i + 1)] = None
            res[cis + '_{0}_max'.format(ti)] = None  # ip 之前登陆的 timelong 最大
            res[cis + '_{0}_cnt'.format(ti)] = 0
            for ii in ['id', 'ip', 'device', 'type', 'city']:
                # TODO  ip,city 重复
                if ii not in cis:
                    res[cis + "_" + ii + "{0}".format(ti)] = 0  # ip 之前登陆的 id 数

    # 最近一次登陆，之前一个小时登陆记录，1 min,2 min,5 min,10 min,30 min,1h,1day(交易次数)
    traded = t_trade[(t_trade['id']==res['id'])&(t_trade['time'] < res['time']) & (t_trade['time'] >= days30str)]  # 筛一个月内的数据
    res['trade_30day_cnt'] = traded.shape[0]
    traded = traded.sort_values('time', ascending=True)

    traded['trade_time_diff'] = traded['trade_stamp'].diff()

    tis = [60, 2 * 60, 30 * 60, 60 * 60, 24 * 60 * 60, 2 * 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60,
           60 * 24 * 60 * 60]
    sis = [0, 60, 2 * 60, 30 * 60, 60 * 60, 24 * 60 * 60, 2 * 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60]
    for si, ti in zip(sis, tis):
        trade_data = traded[(traded['trade_stamp'] >= trade_stamp - ti) & (traded['trade_stamp'] < trade_stamp - si)]
        res['trade_{0}_hour'.format(ti)] = trade_data['hour'].mean()
        res['trade_{0}_cnt'.format(ti)] = trade_data.shape[0]
        res['logintrade_{0}_rate'.format(ti)] = 1.0*res['trade_{0}_cnt'.format(ti)]/(1+res['login_{0}_cnt'.format(ti)])
        res['iptrade_{0}_rate'.format(ti)] = 1.0*res['trade_{0}_cnt'.format(ti)]/(1+res['ip_{0}_cnt'.format(ti)])

        res['trade_time_diff_{0}_max'.format(ti)] = traded['trade_time_diff'].max()
        res['trade_time_diff_{0}_min'.format(ti)] = traded['trade_time_diff'].min()
        res['trade_time_diff_{0}_mean'.format(ti)] = traded['trade_time_diff'].mean()

        di = 60
        res['trade_{0}_cnt{1}'.format(ti, di)] = traded[traded['trade_time_diff'] <= di].shape[0]
        di = 180
        res['trade_{0}_cnt{1}'.format(ti, di)] = \
            traded[(traded['trade_time_diff'] <= di) & (traded['trade_time_diff'] > 60)].shape[0]
        di = 300
        res['trade_{0}_cnt{1}'.format(ti, di)] = \
            traded[(traded['trade_time_diff'] <= di) & (traded['trade_time_diff'] > 180)].shape[0]
        di = 900
        res['trade_{0}_cnt{1}'.format(ti, di)] = \
            traded[(traded['trade_time_diff'] <= di) & (traded['trade_time_diff'] > 300)].shape[0]
        di = 900
        res['trade_{0}_cnt{1}'.format(ti, 901)] = \
            traded[(traded['trade_time_diff'] > di)].shape[0]

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
last_f = '../datas/feas_month_3_7'
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
    print(data.shape)
    data.to_csv(last_f,index=None)
