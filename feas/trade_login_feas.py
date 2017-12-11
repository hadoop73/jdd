# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool


# TODO 筛选 4 - 7 月，根据 trade 记录使用 login 构造特征，以前 5min,30min,1h,3day,7day,15day 的登陆特征
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-04-01 00:00:00']
# 7 月份交易数据
t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade,t_trade_test])

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())

t_trade = t_trade.sort_values('trade_stamp') \
                .reset_index(drop=True)


# TODO 筛选 4 - 7 月份 login 数据

t_login = pd.read_csv('../datas/t_login.csv')
t_login_test = pd.read_csv('../datas/t_login_test.csv')

t_login = t_login[t_login['time']>='2015-03-13 00:00:00']
t_login = pd.concat([t_login,t_login_test])

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)
t_login['hour'] = t_login['time'].map(lambda x:pd.Timestamp(x).hour)
t_login['weekday'] = t_login['time'].map(lambda x:pd.Timestamp(x).date().weekday())

t_login = t_login.sort_values('timestamp') \
                .reset_index(drop=True)


# TODO 根据 trade 数据，从 login 中提取特征,以前 5min,30min,1h,1day,3day,7day,15day 的登陆特征
def baseline(trade):
    rowkey, id, timestamp, is_risk, time = trade
    res = {}
    res['rowkey'] = rowkey
    res['time'] = time
    res['is_risk'] = is_risk

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    t = t.date()
    res['weekday'] = t.weekday()

    tlogin = t_login[t_login['id'] == id][t_login['timestamp_online'] < timestamp]
    res['login_cnt_all'] = tlogin.shape[0]
    # TODO  从 5min,30min,1h,1day,3day,7day,15day 都转化为秒，用时间戳来筛选
    ts = [300, 1500, 216000, 5184000, 15552000, 36288000, 77760000]
    for ti in ts:
        dtmp = tlogin[tlogin['timestamp'] >= timestamp - ti]
        res['login_cnt_{0}'.format(ti)] = dtmp.shape[0]                                 # 登陆的次数
        res['login_faile_cnt_{0}'.format(ti)] = dtmp[dtmp['result'] == -1].shape[0]     # 登陆失败的次数

        for ci in ['ip', 'device', 'city', 'is_sec','weekday',
                   'is_scan','type','result','log_from','hour']:
            try:
                res['login_{0}_cnt_{1}'.format(ci, ti)] = dtmp[ci].unique().size
            except:
                res['login_{0}_cnt_{1}'.format(ci, ti)] = None

    print res
    return  res

t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()


# 如果最近登陆统计存在
last_f = '../datas/trade_login_all'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    d = pool.map(baseline,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)







