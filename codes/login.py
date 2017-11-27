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

# TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
# TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
# TODO 交易自身的特征,之前的交易次数
# TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数






