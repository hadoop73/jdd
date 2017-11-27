# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool

t_login = pd.read_csv('../datas/t_login.csv')
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)

t_login['time'] = t_login['time'].astype(pd.Timestamp)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)

#  给定一个登录时间和 id ，从 t_login 中构造特征

# 交易id，交易时间；根据交易时间和 id  在 login 和 trade 表中筛选数据并构建特征

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

# 最近登陆统计
def etl_Login_last(trade):
    rowkey, id, timestamp = trade
    # 把 time 和 id 转化为 list 方便进行多进程处理
    # t_trade[['time','id']].values.tolist()
    # time:2015-01-01 03:02:01.0  id:18138
    # 给定 id 和 交易时间，筛选登录信息
    # ( 当天 1 个小时内的最近登录信息，最近的登录信息，
    # 最近的 3 次登录信息，最近 3 次登录的 ip 信息，log_from)
    # 最近 3 次的 timelong 信息，最近 3 次的 city 信息，
    # 最近 1 次的交易信息，device 信息，是否使用安全控件，是否扫描

    # 由于登陆时间比登陆时间戳大 28800.0，且存在登陆结果 result 不等于 1 的情况，所以需要过滤
    # TODO 案例分析正样本的登陆结果
    data = t_login[t_login['result'] == 1]
    # 由于登陆存在登陆用时所以需要根据时间戳进行判断
    d = data[data['id']==id][data['timestamp_online']<timestamp]\
                .sort_values('time',ascending=False)\
                .reset_index()
    res = {}
    res['rowkey'] = rowkey
    # 之前时间内，ip,device 登陆次数
    ip_dict = dict(d['ip'].value_counts())
    device_dict = dict(d['device'].value_counts())

    # 考虑 ip,device 的登陆失败次数
    dd = t_login[t_login['id'] == id][t_login['timestamp_online'] < timestamp] \
        .sort_values('time', ascending=False) \
        .reset_index()
    dd['result'] = dd['result'].map(lambda x: x == 1 and 1 or -1)
    ip_dtt = dict(dd[dd['result'] == -1]['ip'].value_counts())
    device_dtt = dict(dd[dd['result'] == -1]['device'].value_counts())

    # 最近一次的登陆所用时间
    res['timelong_1st'] = d.loc[0]['timelong']
    res['time_diff_1st'] = timestamp - d.loc[0]['timestamp_online']
    res['ip_1st_count'] = ip_dict[d.loc[0]['ip']]
    res['ip_1st_result_count'] = d.loc[0]['ip'] in ip_dtt and ip_dtt[d.loc[0]['ip']] or 0
    res['device_1st_count'] = device_dict[d.loc[0]['device']]
    res['device_1st_result_count'] = d.loc[0]['device'] in device_dtt and device_dtt[d.loc[0]['device']] or 0

    res['log_from_1st'] = d.loc[0]['log_from']
    res['city_1st'] = d.loc[0]['city']
    res['result_1st'] = d.loc[0]['result']
    res['type_1st'] = d.loc[0]['type']
    res['is_scan_1st'] = d.loc[0]['is_scan']
    res['is_sec_1st'] = d.loc[0]['is_sec']
    res['time_1st_hour'] = d.loc[0]['time'].hours

    # 最近两次登陆所用时间
    res['timelong_2st'] = d.loc[1]['timelong']
    res['time_diff_2st'] = timestamp - d.loc[1]['timestamp_online']
    res['ip_2st_count'] = ip_dict[d.loc[1]['ip']]
    res['ip_2st_result_count'] = d.loc[1]['ip'] in ip_dtt and ip_dtt[d.loc[1]['ip']] or 0
    res['device_2st_count'] = device_dict[d.loc[1]['device']]
    res['device_2st_result_count'] = d.loc[1]['device'] in device_dtt and device_dtt[d.loc[1]['device']] or 0


    res['log_from_2st'] = d.loc[1]['log_from']
    res['city_2st'] = d.loc[1]['city']
    res['result_2st'] = d.loc[1]['result']
    res['type_2st'] = d.loc[1]['type']
    res['is_scan_2st'] = d.loc[1]['is_scan']
    res['is_sec_2st'] = d.loc[1]['is_sec']
    res['time_2st_hour'] = d.loc[1]['time'].hours

    # 最近三次登陆
    res['timelong_3st'] = d.loc[2]['timelong']
    res['time_diff_3st'] = timestamp - d.loc[2]['timestamp_online']
    res['ip_3st_count'] = ip_dict[d.loc[2]['ip']]
    res['ip_3st_result_count'] = d.loc[2]['ip'] in ip_dtt and ip_dtt[d.loc[2]['ip']] or 0
    res['device_3st_count'] = device_dict[d.loc[2]['device']]
    res['device_3st_result_count'] = d.loc[2]['device'] in device_dtt and device_dtt[d.loc[2]['device']] or 0

    res['log_from_3st'] = d.loc[2]['log_from']
    res['city_3st'] = d.loc[2]['city']
    res['result_3st'] = d.loc[2]['result']
    res['type_3st'] = d.loc[2]['type']
    res['is_scan_3st'] = d.loc[2]['is_scan']
    res['is_sec_3st'] = d.loc[2]['is_sec']
    res['time_3st_hour'] = d.loc[2]['time'].hours

    for ci in ['ip','device','city','is_sec','is_scan']:
        res['last_3{0}_cnt'.format(ci)] = d.loc[:3][ci].unique().size

    # 交易的时间小时
    # res['trad_hour'] = time.date().hours
    # res['trad_weekday'] = time.weekday()
    # # 最近登录时间
    # res['last_interval'] = (pd.Timestamp(time) - pd.Timestamp(d.loc[0]['time'])).total_seconds()
    print res
    return  res


def etl_Login_past(trade):
    rowkey, id, timestamp, is_risk,time = trade
    res = {}
    res['rowkey'] = rowkey
    res['is_risk'] = is_risk
    dd = t_login[t_login['timestamp_online'] < timestamp]
    #ip_dtt = dict(dd[dd['result'] == -1]['ip'].value_counts())
    #device_dtt = dict(dd[dd['result'] == -1]['device'].value_counts())
    # TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
    # TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
    # TODO 交易自身的特征,之前的交易次数
    # TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数
    t = pd.Timestamp(time)
    res['hour'] = t.hour
    t = t.date()
    res['weekday'] = t.weekday()

    d_trade = t_trade[t_trade['id']==id][t_trade['trade_stamp']<timestamp]
    res['trade_cnt'] = d_trade.shape[0]

    d = t_login[t_login['id'] == id][t_login['timestamp_online'] < timestamp]
    for ci in ['ip','device']:
        dt = dd[[ci, 'id', 'type']].groupby([ci, 'id'], as_index=False)['type'].agg({'{0}_cnt'.format(ci):np.size})
        dt = dt[[ci,'{0}_cnt'.format(ci)]].groupby(ci,as_index=False).count()  # 统计 ip 登陆过多少个 id
        for ti in [120,300]:
            dtmp = d[d['timestamp_online'] >= timestamp - ti ]
            # 交易次数
            res['trade_cnt_{0}'.format(ti)] = d_trade[d_trade['trade_stamp'] >= timestamp - ti ].shape[0]
            res['login_cnt_{0}'.format(ti)] = dtmp.shape[0]  # 登陆次数
            ipu = dtmp[ci].unique()
            res['{0}_id_count_{1}'.format(ci, ti)] = ipu.size # 统计有多少个 ip 登陆了
            tmp = dt[ci].map(lambda x:x in ipu)
            res['{0}_id_max_{1}'.format(ci,ti)] = dt[tmp]['{0}_cnt'.format(ci)].max()
            res['{0}_id_min_{1}'.format(ci,ti)] = dt[tmp]['{0}_cnt'.format(ci)].min()
            res['{0}_id_mean_{1}'.format(ci,ti)] = dt[tmp]['{0}_cnt'.format(ci)].mean()
            res['{0}_id_sum_{1}'.format(ci, ti)] = dt[tmp]['{0}_cnt'.format(ci)].sum()
    print(res)
    return res

def etl_Login_minutes(trade):
    rowkey, id, timestamp,is_risk = trade
    # 考虑交易时间前 30s，1 分钟，2分钟，5分钟的登陆交易情况
    # 交易前 1 分钟登陆的 ip 数，交易数，device 数
    # 可以通过 ip，device 进行统计
    # TODO 一段时间内，登陆次数，交易次数，登陆时长，登陆结果
    res = {}
    res['rowkey'] = rowkey
    res['is_risk'] = is_risk
    data = t_login[t_login['result'] == 1]
    d = data[data['id'] == id][data['timestamp_online'] < timestamp] \
        .sort_values('time', ascending=False) \
        .reset_index()

    dd = t_login[t_login['id'] == id][t_login['timestamp_online'] < timestamp] \
        .sort_values('time', ascending=False) \
        .reset_index()
    dd['result'] = dd['result'].map(lambda x: x == 1 and 1 or -1)
    #ip_dtt = dict(dd[dd['result'] == -1]['ip'].value_counts())
    #device_dtt = dict(dd[dd['result'] == -1]['device'].value_counts())

    for ti in [30,60,120,300]:
        dtmp = d[d['timestamp_online'] >= timestamp - ti ]
        dd30 = dd[dd['timestamp_online'] >= timestamp - ti]
        res['login_{0}_fail'.format(ti)] = dd30[dd30['result'] == -1].shape[0]
        res['login_{0}_cnt'.format(ti)] = dtmp.shape[0]
        res['login_{0}_timelong_max'.format(ti)] = dtmp['timelong'].max()
        res['login_{0}_timelong_mean'.format(ti)] = dtmp['timelong'].mean()
        res['login_{0}_timelong_min'.format(ti)] = dtmp['timelong'].min()
        res['login_{0}_timelong_std'.format(ti)] = dtmp['timelong'].std()
        for ci in ['ip', 'device', 'city', 'is_sec', 'is_scan']:
            res['login_{0}_{1}_cnt'.format(ti, ci)] = dtmp[ci].unique().size
    print(res)
    return res

t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()


# TODO 题目意图：存在一些登陆记录，然后某个时刻出现了交易行为，判断其是否有风险
# 根据某一条登陆信息，判断其有多大的风险


# 如果最近登陆统计存在
last_f = '../datas/etl_Login_past'
if os.path.exists(last_f):
        data = pd.read_csv(last_f)
else:
    import time
    start_time = time.time()
    pool = Pool(8)
    d = pool.map(etl_Login_past,t_trade_list)
    pool.close()
    pool.join()
    print 'time : ', 1.0*(time.time() - start_time)/60
    data = pd.DataFrame(d)
    #print(data.head(100))
    data.to_csv(last_f,index=None)









