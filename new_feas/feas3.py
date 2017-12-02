# coding:utf-8

import pandas as pd
import numpy as np
import os, sys
from multiprocessing import Pool

# TODO 统计每个月的每条交易最近一次，两次登陆的 ip,device 等环境信息

# TODO 合并 login 的 train,test 数据
t_login = pd.read_csv('../datas/t_login.csv')
t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = t_login[t_login['time'] >= '2015-03-01 00:00:00']  # 只考虑 3-7 月的登陆信息，也就是 trade 只往前看一个月
t_login = pd.concat([t_login, t_login_test])
del t_login_test

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login = t_login.sort_values('timestamp') \
    .reset_index(drop=True)

# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time'] >= '2015-04-01 00:00:00']  # 只需要从 3-7 开始分析

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
    # TODO 每一条交易记录之前的登陆记录是否存在同一个 `ip` 一次 `result=31` 和 `type=3` 的记录 (id=175546)
    # TODO 连续两条登陆记录一条，同一个 `ip`，有一次 `result=31` 和 `type=3` 而且时间间隔非常短，与交易的日期也很近
    idata = t_trade.loc[idx]
    res = {}
    res['rowkey'] = idata['rowkey']
    res['time'] = idata['time']
    timestamp, id = idata['trade_stamp'], idata['id']

    login_data = t_login[(t_login['id'] == id) &
                         (t_login['timestamp_online'] < timestamp)]
    cntmax = 0
    for ti in [5 * 50, 30 * 60, 60 * 60, 24 * 60 * 60, 7 * 24 * 60 * 60, 15 * 24 * 60 * 60]:
        di = login_data[(login_data['timestamp_online'] >= timestamp - ti)]
        di = di[di['result']==31]
        res['ip_cnt{0}'.format(ti)] = di['ip'].unique().size
        res['result_31_cnt{0}'.format(ti)] = di.shape[0]                       # 统计 result=31 的次数
        res['ip_rate{0}'.format(ti)] = 1.0*res['result_31_cnt{0}'.format(ti)]/(1+res['ip_cnt{0}'.format(ti)])  # 统计 result=31 的次数
        if res['result_31_cnt{0}'.format(ti)] > 0:
            n = 0
            for ix in di.index.tolist():
                if di.loc[ix]['ip'] == di.loc[ix+1]['ip'] and di.loc[ix+1]['result']==1:
                     n += 1
            res['result_2_cnt{0}'.format(ti)] = n                             # 统计 result=31 后登陆成功的次数
        else:
            res['result_2_cnt{0}'.format(ti)] = 0
        cntmax = max(cntmax,res['result_2_cnt{0}'.format(ti)])
    res['result_2_max'] = cntmax                                              # 统计 result=31 后登陆成功的次数 max
    # TODO 短时间内登陆很多次 (id=80306)

    print(res)
    return res


# 如果最近登陆统计存在

# 只用 4,5,6,7 做特征进行 train,valiade,test
t_trade_list = t_trade.index.tolist()

last_f = '../datas/feas_new_p'
if os.path.exists(last_f):
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
    data.to_csv(last_f, index=None)
