# coding:utf-8

import pandas as pd
import numpy as np
import os,datetime
from multiprocessing import Pool

# TODO 合并 train 和 test 数据
# TODO 最近登陆与交易的时间差
# TODO 考虑上一个月或者之前同样登陆环境的风险程度，也就是同样的环境登陆次数
# TODO 是否存在短时间内同一个 id，多个 device 或者 ip 登陆，短时间内一个 ip 多次登陆，判断最近两次的 ip 是否相等
# TODO ip-id,device-id 是否有登陆成功但是没有交易行为，这很有可能没有风险，因为被黑账户肯定第一次就会进行交易的，统计登陆次数/交易次数
# TODO 风险记录会交易多次，牟利更多,统计当天的交易次数，30 min，1h 之内的交易次数


# TODO 合并 login 的 train,test 数据
t_login = pd.read_csv('../datas/t_login.csv')
t_login_test = pd.read_csv('../datas/t_login_test.csv')
t_login = t_login[t_login['time']>='2015-03-01 00:00:00']  # 只考虑 3-7 月的登陆信息，也就是 trade 只往前看一个月
t_login = pd.concat([t_login,t_login_test])
del t_login_test

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)
t_login = t_login.sort_values('timestamp') \
                .reset_index(drop=True)

# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-03-01 00:00:00']  # 只需要从 3-7 开始分析

t_trade_test = pd.read_csv('../datas/t_trade_test.csv')
t_trade_test['is_risk'] = -1
t_trade = pd.concat([t_trade,t_trade_test])
del t_trade_test

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())
t_trade = t_trade.sort_values('trade_stamp') \
                .reset_index(drop=True)



def baseline_new(idx):

    idata =  t_trade.loc[idx]
    res = {}
    res['rowkey'] = idata['rowkey']
    time = idata['time']

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    res['hour_v'] = res['hour'] / 6                 # 时间段
    res['weekday'] = t.dayofweek                    # 周信息

    month = t.month
    start_date = '2015-0{0}-01 00:00:00'.format(month - 1)          # 从上一个月开始统计

    timestamp,id = idata['trade_stamp'],idata['id']                 # 筛选 trade 之前的登陆信息，从最近的 login 进行特征提取

    # TODO 筛选 trade 数据并构造特征,统计之前一个月内的交易次数
    # month_date = '2015-0{0}-01 00:00:00'.format(month)          # 从上一个月开始统计
    trade_data = t_trade[(t_trade['id'] == id) &
                         (t_trade['time'] >= start_date) &
                         (t_trade['time'] < time)]


    # TODO 交易时间与登陆时间之间的差值
    login_data = t_login[(t_login['id'] == id) &
                         (t_login['time'] >= start_date) &
                         (t_login['time'] < time)]
    login_data = login_data[login_data['result']==1]
    login_data = login_data.sort_values('time',ascending=False) \
                        .reset_index(drop=True)     # 对登陆成功的数据进行倒序排序

    res['login_13_cnt'] = login_data.loc[:2].shape[0]
    res['log_from_12_cnt'] = login_data.loc[:1]['log_from'].unique().size
    res['ip_12_cnt'] = login_data.loc[:1]['ip'].unique().size
    res['timelong_13_max'] = login_data.loc[:2]['timelong'].max()
    try:
        res['login_12_time_diff'] = login_data.loc[0]['timestamp'] - login_data.loc[1]['timestamp']
    except:
        res['login_12_time_diff'] = None

    try:
        res['login_23_time_diff'] = login_data.loc[1]['timestamp'] - login_data.loc[2]['timestamp']
    except:
        res['login_23_time_diff'] = None

    del trade_data,login_data
    print res
    return res



# 如果最近登陆统计存在

dtt = t_trade[t_trade['time']>='2015-04-01 00:00:00']    # 只用 4,5,6,7 做特征进行 train,valiade,test
t_trade_list = dtt.index.tolist()
del dtt

last_f = '../datas/feas_new_more'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    df = pool.map(baseline_new,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(df)
    data.to_csv(last_f,index=None)













