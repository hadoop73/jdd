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
t_login = t_login[t_login['time']>='2015-01-21 00:00:00']  # 只考虑 3-7 月的登陆信息，也就是 trade 只往前看一个月
t_login = pd.concat([t_login,t_login_test])
del t_login_test

t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)
t_login = t_login.sort_values('timestamp') \
                .reset_index(drop=True)

# TODO 合并 trade 的 train，test 数据
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-01-21 00:00:00']  # 只需要从 3-7 开始分析

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
    res['is_risk'] = idata['is_risk']
    time = idata['time']

    t = pd.Timestamp(time)
    res['hour'] = t.hour
    res['hour_v'] = res['hour'] / 6                 # 时间段
    res['weekday'] = t.dayofweek                    # 周信息

    month = t.month
    start_date = str(t - pd.Timedelta(days=60))
    #start_date = '2015-0{0}-01 00:00:00'.format(month - 1)          # 从上一个月开始统计

    timestamp,id = idata['trade_stamp'],idata['id']                 # 筛选 trade 之前的登陆信息，从最近的 login 进行特征提取

    # TODO 筛选 trade 数据并构造特征,统计之前一个月内的交易次数
    # month_date = '2015-0{0}-01 00:00:00'.format(month)          # 从上一个月开始统计
    trade_data = t_trade[(t_trade['id'] == id) &
                         (t_trade['time'] >= start_date) &
                         (t_trade['trade_stamp'] < timestamp)]

    res['trade_cnt'] = trade_data.shape[0]                                      # 交易次数
    res['trade_weekday_cnt'] = trade_data[trade_data['weekday']==res['weekday']].shape[0]         # 同一个周几交易次数
    res['trade_weekday_cnt_rate'] = 1.0 * res['trade_weekday_cnt'] / (1 + res['trade_cnt'])
    res['hour_v_cnt'] = trade_data[trade_data['hour_v'] == res['hour_v']].shape[0]                # 同一个时间段交易次数
    res['hour_v_cnt_rate'] = 1.0 * res['hour_v_cnt'] / (1 + res['trade_cnt'])

    for ti in [5*60,30*60,60*60]:
        tmp = trade_data[(trade_data['trade_stamp'] >= timestamp - ti)]
        res['time_cnt{0}'.format(ti)] = tmp.shape[0]

    # TODO 交易时间与登陆时间之间的差值
    login_data = t_login[(t_login['id'] == id) &
                         (t_login['time'] >= start_date) &
                         (t_login['timestamp_online'] < timestamp)]
    login_data = login_data[login_data['result']==1]
    login_data = login_data.sort_values('timestamp_online',ascending=False) \
                        .reset_index(drop=True)     # 对登陆成功的数据进行倒序排序

    for i in [0, 1]:
        try:
            res['time_diff{0}'.format(i + 1)] = timestamp - login_data.loc[i]['timestamp_online']           # 距离上一次的登陆时间间隔
            # TODO 登陆 ip 与交易次数的关系，登陆很多次没有购买肯定就是没有风险的
            login_ip = login_data.loc[i]['ip']
            res['ip_cnt{0}'.format(i + 1)] = login_data[login_data['ip']==login_ip].shape[0]                # 之前 ip 登陆的次数
            login_device = login_data.loc[i]['device']
            res['device_cnt{0}'.format(i + 1)] = login_data[login_data['device']==login_device].shape[0]    # 之前 ip 登陆的次数
            res['ip_device_cnt{0}'.format(i + 1)] = login_data[(login_data['device']==login_device)&
                                                               (login_data['ip']==login_ip)].shape[0]       # 之前 ip,device 登陆的次数
        except:
            res['time_diff{0}'.format(i + 1)] = None
            res['ip_cnt{0}'.format(i + 1)] = 0
            res['device_cnt{0}'.format(i + 1)] = 0
            res['ip_device_cnt{0}'.format(i + 1)] = 0
        res['ip_cnt_trade_rate{0}'.format(i + 1)] = 1.0*res['ip_cnt{0}'.format(i + 1)]/(1+res['trade_cnt'])
        res['device_cnt_trade_rate{0}'.format(i + 1)] = 1.0*res['device_cnt{0}'.format(i + 1)]/(1+res['trade_cnt'])
        res['ip_device_cnt_trade_rate{0}'.format(i + 1)] = 1.0*res['ip_device_cnt{0}'.format(i + 1)]/(1+res['trade_cnt'])


    try:
        res['time_diff12'] = res['time_diff1'] - res['time_diff2']
        res['ip12'] = login_data.loc[0]['ip'] == login_data.loc[1]['ip'] and 1 or 0                 # 最近登陆两次的 ip 是否相等
        res['device12'] = login_data.loc[0]['device'] == login_data.loc[1]['device'] and 1 or 0     # 最近登陆两次的 device 是否相等
    except:
        res['time_diff12'] = None
        res['ip12'] = None
        res['device12'] = None

    # TODO 改 max 为 sum
    res['ip_device12_sum'] = res['ip12'] + res['device12']
    del trade_data,login_data
    print res
    return res



# 如果最近登陆统计存在

dtt = t_trade[t_trade['time']>='2015-04-01 00:00:00']    # 只用 4,5,6,7 做特征进行 train,valiade,test
t_trade_list = dtt.index.tolist()
del dtt

last_f = '../datas/feas_new2'
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













