# coding:utf-8

import pandas as pd
import numpy as np

t_login = pd.read_csv('../datas/t_login.csv')
t_trade = pd.read_csv('../datas/t_trade.csv')
t_login['time'] = t_login['time'].astype(pd.Timestamp)

#  给定一个登录时间和 id ，从 t_login 中构造特征

# 交易id，交易时间；根据交易时间和 id  在 login 和 trade 表中筛选数据并构建特征
id = 18138
time = '2015-01-01 03:02:31.0'

def etl_Login(id,time):
    # 把 time 和 id 转化为 list 方便进行多进程处理
    # t_trade[['time','id']].values.tolist()
    # time:2015-01-01 03:02:01.0  id:18138
    # 给定 id 和 交易时间，筛选登录信息
    # ( 当天 1 个小时内的最近登录信息，最近的登录信息，
    # 最近的 3 次登录信息，最近 3 次登录的 ip 信息，log_from)
    # 最近 3 次的 timelong 信息，最近 3 次的 city 信息，
    # 最近 1 次的交易信息，device 信息，是否使用安全控件，是否扫描
    d = t_login[t_login['id']==id][t_login['time']<time]\
                .sort_values('time',ascending=False)\
                .reset_index()
    res = {}
    # 交易的时间小时
    res['trad_hour'] = time.date().hours
    res['trad_weekday'] = time.weekday()
    # 最近登录时间
    res['last_interval'] = (pd.Timestamp(time) - pd.Timestamp(d.loc[0]['time'])).total_seconds()
    # 最近 3 次登录的 ip 次数统计，设备次数统计
    for ci in ['ip','device','city']:
        res['last_3{0}_cnt'.format(ci)] = d.loc[0:3][ci].unique().size

    # 最近一次是否有安全控件
    res['login_is_sec'] = d.loc[0]['is_sec']

    return  res








