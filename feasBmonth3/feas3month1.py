# coding:utf-8

import pandas as pd
import numpy as np
import os, sys
from multiprocessing import Pool

#######################################################################################
# TODO id,time,hour,生成登陆的时间差
# TODO 最近一次的登陆时长
# TODO 2,5,7,15,30 天前， device,ip,device# TODOip，登陆过同一个 id 的次数
# TODO 交易时间戳与最近一次登陆时间戳的差值
# TODO 最近一次登陆成功的时间戳与最近第二次登陆成功的时间戳，最近第三次时间戳的差值
# TODO 最近第二次登陆的 result，type 类型，以及与最近一次成功登陆的时间戳差值
# TODO 最近一次登陆产生多少次的交易次数
# TODO 过去 60s/2min/5min/10min/30min/1h/2day
#### TODO 登陆时间差小于 60, 120, 180 的次数
#### TODO 统计其中连续 result=31，result=1 数据出现的次数，以及它们之间的时间差值
#### TODO city 的变动次数，变动率
# TODO  最近一次登陆，之前一个小时登陆记录，1 min,2 min,5 min,10 min,30 min,1h,1day(交易次数)
# TODO 2,5,7,15,30 天前交易记录数
#######################################################################################
# TODO 合并 login 的 train,test 数据
t_login = pd.read_csv('../datas/t_login.csv')
t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = t_login[t_login['time'] >= '2015-01-21 00:00:00']  # 只考虑 3-7 月的登陆信息，也就是 trade 只往前看一个月
t_login = pd.concat([t_login, t_login_test])
del t_login_test

# TODO log_from 异常处理
t_login['log_from'] = t_login['log_from'].map(lambda x: 3 if x not in [1,2,8,10,11,16,18,21] else x)
# TODO result 异常处理
t_login['result'] = t_login['result'].map(lambda x: 5 if x not in [-4,-2,-1,1,6,22,31] else x)
# TODO timelong 异常处理
t_login = t_login[t_login['timelong']>0]

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login = t_login.sort_values('time') \
    .reset_index(drop=True)

# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time'] >= '2015-01-21 00:00:00']  # 只需要从 3-7 开始分析

t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade, t_trade_test])
del t_trade_test

t_trade['trade_stamp'] = t_trade['time'].map(lambda x: pd.to_datetime(x).value // 10 ** 9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x: pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x: x / 6)
t_trade['weekday'] = t_trade['time'].map(lambda x: pd.Timestamp(x).date().weekday())
t_trade = t_trade.sort_values('trade_stamp') \
    .reset_index(drop=True)


def baseline_new(idx):
    idata = t_trade.loc[idx]
    res = {}
    res['rowkey'] = idata['rowkey']
    res['id'] = idata['id']
    res['time'] = idata['time']                                             # 交易时间，用于模型训练切分数据
    res['is_risk'] = idata['is_risk']

    t = pd.Timestamp(res['time'])
    days30str = str(t - pd.Timedelta(days=60))
    res['hour'] = t.hour                                                    # 交易的时段，比较重要的特征

    timestamp, id = idata['trade_stamp'], idata['id']

    login_data = t_login[(t_login['id'] == id) &
                         (t_login['timestamp'] < timestamp) &
                         (t_login['time'] >= days30str)] \
        .sort_values('time') \
        .reset_index(drop=True)                                             # 筛选数据用于做 login 的特征
    login_data['login_time_diff'] = login_data['timestamp'].diff()          # 统计登陆时间差

    login_data1 = login_data[(login_data['result'] == 1)] \
        .sort_values('time', ascending=False) \
        .reset_index(drop=False)                                            # 只关注登陆成功的记录

    try:
        login = login_data1.loc[0]                                          # 最近一条登陆信息,可能没有最近一条登陆信息
        res['timelong'] = login['timelong']                                 # 登陆时长
        login_time, lg_time = login['timestamp'], login['time']
        lg_time = pd.Timestamp(lg_time)
        ix, ip = login['index'], login['ip']

        ti = 1
        lg_timestr = str(lg_time - pd.Timedelta(days=ti))
        lg_stimestr = str(lg_time - pd.Timedelta(days=ti + 2))
        login_tmp = login_data[(login_data['time'] < lg_timestr) & (login_data['time'] >= lg_stimestr)]  # 2/5/7/15 天前的登陆记录
        res['ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip)].shape[0]
        res['device_id_{0}'.format(ti)] = login_tmp[(login_tmp['device'] == login['device'])].shape[0]
        res['device_ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip) &
                                                       (login_tmp['device'] == login['device'])].shape[0]

        ti = 7
        lg_timestr = str(lg_time - pd.Timedelta(days=ti))
        lg_stimestr = str(lg_time - pd.Timedelta(days=ti + 7))
        login_tmp = login_data[(login_data['time'] < lg_timestr) & (login_data['time'] >= lg_stimestr)]  # 2/5/7/15 天前的登陆记录
        res['ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip)].shape[0]
        res['device_id_{0}'.format(ti)] = login_tmp[(login_tmp['device'] == login['device'])].shape[0]
        res['device_ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip) &
                                                       (login_tmp['device'] == login['device'])].shape[0]
        ti = 15
        lg_timestr = str(lg_time - pd.Timedelta(days=ti))
        login_tmp = login_data[(login_data['time'] < lg_timestr)]                     # 2/5/7/15 天前的登陆记录
        res['ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip)].shape[0]
        res['device_id_{0}'.format(ti)] = login_tmp[(login_tmp['device'] == login['device'])].shape[0]
        res['device_ip_id_{0}'.format(ti)] = login_tmp[(login_tmp['ip'] == ip) &
                                                       (login_tmp['device'] == login['device'])].shape[0]
        # 登陆时间差
        res['trade_login_time_diff'] = timestamp - login_time                         # 登陆与交易的时间差
        try:
            res['trade_login_time_diff1'] = login_time - login_data1.loc[1]['timestamp']
        except:
            res['trade_login_time_diff1'] = None

        try:
            res['trade_login_time_diff2'] = login_time - login_data1.loc[2]['timestamp']
        except:
            res['trade_login_time_diff2'] = None

        try:
            ixd = login_data.loc[ix - 1]
            if ixd['result'] == 31 and ixd['type'] == 3:
                res['result31_1'] = 1  # 是否有连续两次 result=31,result=1
            else:
                res['result31_1'] = 0

            if ixd['result'] == 1 and ixd['type'] == 3:
                res['result1_3'] = 1
            else:
                res['result1_3'] = 0

            if ixd['result'] == 1 and ixd['type'] == 1:
                res['result1_1'] = 1
            else:
                res['result1_1'] = 0

            res['login2_time_diff'] = login_time - ixd['timestamp']  # 统计两次登陆的时间差

        except:
            res['result31_1'] = None  # 是否有连续两次 result=31,result=1
            res['result1_3'] = None
            res['result1_1'] = None

            res['login2_time_diff'] = None

        # TODO login_time_diff < 100 的次数，ip,id/id 最近 30min，result=31,result=1 的次数
        # 最近一次登陆产生了多少次交易记录
        trade = t_trade[(t_trade['trade_stamp'] > login_time) & (t_trade['trade_stamp'] < timestamp)]
        res['trade_login_hour_cnt'] = trade.shape[0]

        # 过去一段时间内是否存在短时间登陆情况，可以统计 login_time_diff < 60,120,180 的次数
        # 最近 30s/5min/30min/1h/2day
        tis = [60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60, 48 * 60 * 60]
        sis = [0, 60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60]
        for si,ti in zip(sis,tis):
            login_tmp = login_data[(login_data['timestamp'] < login_time - si) &(login_data['timestamp'] >= login_time - ti) ]

            di = 60
            res['login_{0}_cnt{1}'.format(ti, di)] = login_tmp[login_tmp['login_time_diff'] <= di].shape[0]
            di = 120
            res['login_{0}_cnt{1}'.format(ti, di)] = login_tmp[(login_tmp['login_time_diff'] <= di)&(login_tmp['login_time_diff'] > di - 60)].shape[0]
            di = 180
            res['login_{0}_cnt{1}'.format(ti, di)] = login_tmp[login_tmp['login_time_diff'] > di - 60].shape[0]

            res['city_cnt{0}'.format(ti)] = login_tmp['city'].unique().size
            res['city_cnt{0}rate'.format(ti)] = 1 - 1.0 * res['city_cnt{0}'.format(ti)] / (1 + login_tmp.shape[0])

            for i in [1,2,3,8,10,11,16,18,21]:
                res['log_from_{0}_{1}'.format(ti,i)] = login_tmp[login_tmp['log_from']==i].shape[0]

            for i in [-4,-2,-1,5,1,6,22,31]:
                res['result_{0}_{1}'.format(ti,i)] = login_tmp[login_tmp['result']==i].shape[0]

            for i in [1,2,3]:
                res['type_{0}_{1}'.format(ti,i)] = login_tmp[login_tmp['type']==i].shape[0]

        login_tmp = login_data[(login_data['timestamp'] >= login_time - 48 * 60 * 60)]
        r31 = login_tmp[(login_tmp['result'] == 31) & (login_tmp['type'] == 3)].index.tolist()
        n, tsum = 0, []
        ti = 0
        try:
            for i in r31:
                if login_tmp.loc[i + 1]['result'] == 1 and login_tmp.loc[i + 1]['type'] == 3:  # 统计连续 result=31，result=1 的个数
                    n += 1
                    tsum.append(login_tmp.loc[i + 1]['timestamp'] - login_tmp.loc[i]['timestamp'])
            res['result_31_and_1cnt{0}'.format(ti)] = n
            res['result_timestamp_min{0}'.format(ti)] = min(tsum)
            res['result_timestamp_max{0}'.format(ti)] = max(tsum)
            res['result_timestamp_mean{0}'.format(ti)] = np.mean(tsum)
        except:
            res['result_31_and_1cnt{0}'.format(ti)] = None
            res['result_timestamp_min{0}'.format(ti)] = None  # 统计这些时间间隔的 min,max,mean
            res['result_timestamp_max{0}'.format(ti)] = None
            res['result_timestamp_mean{0}'.format(ti)] = None
    except:
        res['timelong'] = None
        ti = 1
        res['ip_id_{0}'.format(ti)] = None
        res['device_id_{0}'.format(ti)] = None
        res['device_ip_id_{0}'.format(ti)] = None
        ti = 7
        res['ip_id_{0}'.format(ti)] = None
        res['device_id_{0}'.format(ti)] = None
        res['device_ip_id_{0}'.format(ti)] = None
        ti = 15
        res['ip_id_{0}'.format(ti)] = None
        res['device_id_{0}'.format(ti)] = None
        res['device_ip_id_{0}'.format(ti)] = None
        # 登陆时间差
        res['trade_login_time_diff'] = None  # 登陆与交易的时间差
        res['trade_login_time_diff1'] = None
        res['trade_login_time_diff2'] = None
        res['result31_1'] = None
        res['result1_3'] = None
        res['result1_1'] = None
        res['login2_time_diff'] = None
        res['trade_login_hour_cnt'] = None
        # 过去一段时间内是否存在短时间登陆情况，可以统计 login_time_diff < 60,120,180 的次数
        # 最近 30s/5min/30min/1h/2day
        tis = [60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60, 48 * 60 * 60]
        sis = [0, 60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60]
        for si, ti in zip(sis, tis):
            di = 60
            res['login_{0}_cnt{1}'.format(ti, di)] = None
            di = 120
            res['login_{0}_cnt{1}'.format(ti, di)] = None
            di = 180
            res['login_{0}_cnt{1}'.format(ti, di)] = None
            res['city_cnt{0}'.format(ti)] = None
            res['city_cnt{0}rate'.format(ti)] = None
            for i in [1, 2, 3, 8, 10, 11, 16, 18, 21]:
                res['log_from_{0}_{1}'.format(ti, i)] = None
            for i in [-4, -2, -1, 5, 1, 6, 22, 31]:
                res['result_{0}_{1}'.format(ti, i)] = None
            for i in [1, 2, 3]:
                res['type_{0}_{1}'.format(ti, i)] = None
        ti = 0
        res['result_31_and_1cnt{0}'.format(ti)] = None
        res['result_timestamp_min{0}'.format(ti)] = None  # 统计这些时间间隔的 min,max,mean
        res['result_timestamp_max{0}'.format(ti)] = None
        res['result_timestamp_mean{0}'.format(ti)] = None

    # TODO
    # 最近一次登陆，之前一个小时登陆记录，1 min,2 min,5 min,10 min,30 min,1h,1day(交易次数)
    traded = t_trade[
        (t_trade['id'] == res['id']) & (t_trade['time'] < res['time']) & (t_trade['time'] >= days30str)]  # 筛一个月内的数据

    tis = [60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60, 24 * 60 * 60]
    sis = [0, 60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60]
    for si,ti in zip(sis,tis):
        trade_data = traded[(traded['trade_stamp'] >= timestamp - ti)&(traded['trade_stamp'] < timestamp - si)]
        res['trade_{0}_cnt'.format(ti)] = trade_data.shape[0]

    tis = [1, 2, 7, 15, 30,60]
    sis = [0, 1, 2, 7, 15,30]
    for si,ti in zip(sis,tis):
        td_timestr = str(t - pd.Timedelta(days=ti))
        sd_timestr = str(t - pd.Timedelta(days=si))
        trade_tmp = traded[(traded['time'] < sd_timestr)&(traded['time'] >= td_timestr)]  # 2/5/7/15 天前的交易记录
        res['trade_{0}_cnt_pre'.format(ti)] = trade_tmp.shape[0]

    # TODO 过去 2-3 内，是否存在，以内同一 ip 登陆 15s，20s 以内，且有 result=31，result=1，并记录次数，time diff
    # TODO 该 id 30min，1h 交易次数，交易 hour
    print(res)
    return res


# 如果最近登陆统计存在

# 只用 4,5,6,7 做特征进行 train,valiade,test
dtt = t_trade[t_trade['time']>='2015-03-01 00:00:00']
dtt = dtt[dtt['time']<'2015-04-01 00:00:00']
t_trade_list = dtt.index.tolist()
'''
baseline_new(t_trade_list[1000])
'''
last_f = '../datas/feas_login_new31_month3'
if os.path.exists(last_f) and False:
    data = pd.read_csv(last_f)
else:
    import time

    start_time = time.time()
    pool = Pool(8)
    df = pool.map(baseline_new, t_trade_list)
    pool.close()
    pool.join()
    print('time : ', 1.0 * (time.time() - start_time) / 60)
    data = pd.DataFrame(df)
    print data.shape
    data.to_csv(last_f, index=None)
