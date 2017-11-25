# coding:utf-8

import pandas as pd
import numpy as np
import os,sys

t_login = pd.read_csv('../datas/t_login.csv')
t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)

t_login['time'] = t_login['time'].astype(pd.Timestamp)
t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']

#  给定一个登录时间和 id ，从 t_login 中构造特征

# 交易id，交易时间；根据交易时间和 id  在 login 和 trade 表中筛选数据并构建特征
id = 18138
time = '2015-01-01 03:02:31.0'

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
def etl_Login(id,timestamp,feasDir=''):
    if os._exists(feasDir) :
        pass
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

    for ci in ['ip','device','city']:
        res['last_3{0}_cnt'.format(ci)] = d.loc[:3][ci].unique().size


    # 考虑交易时间前 30s，1 分钟，2分钟，5分钟的登陆交易情况
    # 交易前 1 分钟登陆的 ip 数，交易数，device 数
    # 可以通过 ip，device 进行统计

    # 交易的时间小时
    res['trad_hour'] = time.date().hours
    res['trad_weekday'] = time.weekday()
    # 最近登录时间
    res['last_interval'] = (pd.Timestamp(time) - pd.Timestamp(d.loc[0]['time'])).total_seconds()

    return  res








