# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
import multiprocessing
from multiprocessing import Pool
import multiprocessing.pool

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

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def baseline_1(da):
    # TODO 统计这段时间内 ip，device 的登陆失败记录,最长登陆时间记录，用户变动 ip,device 的频次，登陆的频次
    # TODO 两次登陆的时间差，更换 ip,device 的时间差，一段时间内 ip 登陆次数的多少
    # TODO 交易自身的特征,之前的交易次数
    # TODO 对登陆表的特征做的更丰富，再和交易表进行拼接，统计 ip,device 登陆次数，失败次数，扫码次数，安全控件次数
    idx,data = da
    res = {}
    d = data.loc[:idx - 1]
    res['idx'] = idx
    timestamp = data.loc[idx]['timestamp']
    # TODO ip 在之前时间内，登陆的次数，登陆的用户数，登陆的时间长度，登陆的城市数，登陆的成功次数
    # TODO id 在之前的时间内，登陆的 ip 次数，device 次数，登陆时间长度，城市数，登陆成功次数
    # TODO ip,id,device,type,city 以及这些的组合，

    for ci in ['ip', 'device', 'id']:
        idata = data.loc[idx][ci]
        dci = d[d[ci] == idata]
        dci = dci.sort_values('timestamp', ascending=False) \
            .reset_index(drop=True)
        for i in [0, 1, 2]:
            try:
                res[ci + '_diff{0}'.format(i + 1)] = timestamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
            except:
                res[ci + '_diff{0}'.format(i + 1)] = None
        res[ci + '_time_max'] = dci['timelong'].max()  # ip 之前登陆的 timelong 最大
        res[ci + '_time_min'] = dci['timelong'].min()  # ip 之前登陆的 timelong 最小
        res[ci + '_time_mean'] = dci['timelong'].mean()  # ip 之前登陆的 timelong 均值
        # TODO new 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
        res[ci + '_time_std'] = dci['timelong'].std()

        # TODO new 关注最近三次
        try:
            res[ci + '_time_max2'] = dci.loc[:2]['timelong'].max()  # ip 之前登陆的 timelong 最大
            res[ci + '_time_min2'] = dci.loc[:2]['timelong'].min()  # ip 之前登陆的 timelong 最小
            res[ci + '_time_mean2'] = dci.loc[:2]['timelong'].mean()  # ip 之前登陆的 timelong 均值
            # TODO new 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
            res[ci + '_time_std2'] = dci.loc[:2]['timelong'].std()
        except:
            res[ci + '_time_max2'] = None
            res[ci + '_time_min2'] = None
            res[ci + '_time_mean2'] = None
            res[ci + '_time_std2'] = None

        res[ci + '_cnt'] = dci.shape[0]  # ip 之前登陆的次数
        res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]  # ip 之前登陆失败的次数
        res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]  # ip 之前登陆成功的次数
        res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])
        res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

        for ii in ['id', 'ip', 'device', 'type', 'city']:
            # TODO  ip,city 重复
            if ii != ci:
                res[ci + "_" + ii] = dci[ii].unique().size  # ip 之前登陆的 id 数

        # res[ci + '_city'] = dci['city'].unique().size  # ip 之前登陆的 log_from 数
        res[ci + '_from'] = dci['log_from'].unique().size  # ip 之前登陆的 log_from 数
        res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]  # ip 之前安全插件登陆次数
        res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]  # ip 之前非安全插件登陆次数
        res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]  # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]  # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])

    for ci in [('id', 'ip'), ('id', 'device'), ('id', 'type'),
               ('ip', 'device'), ('ip', 'type'), ('device', 'type'), ('id', 'city')]:
        idata, jdata = data.loc[idx][ci[0]], data.loc[idx][ci[1]]
        dci = d[((d[ci[0]] == idata) & (d[ci[1]] == jdata))]
        ci = ci[0] + ci[1]
        dci = dci.sort_values('timestamp', ascending=False) \
            .reset_index(drop=True)
        for i in [0, 1, 2]:
            try:
                res[ci + '_diff{0}'.format(i + 1)] = timestamp - dci.loc[i]['timestamp']  # 距离上一次的登陆时间间隔
            except:
                res[ci + '_diff{0}'.format(i + 1)] = None
        res[ci + '_time_max'] = dci['timelong'].max()  # ip 之前登陆的 timelong 最大
        res[ci + '_time_min'] = dci['timelong'].min()  # ip 之前登陆的 timelong 最小
        res[ci + '_time_mean'] = dci['timelong'].mean()  # ip 之前登陆的 timelong 均值
        # TODO new 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
        res[ci + '_time_std'] = dci['timelong'].std()
        # TODO new 关注最近三次
        try:
            res[ci + '_time_max2'] = dci.loc[:2]['timelong'].max()  # ip 之前登陆的 timelong 最大
            res[ci + '_time_min2'] = dci.loc[:2]['timelong'].min()  # ip 之前登陆的 timelong 最小
            res[ci + '_time_mean2'] = dci.loc[:2]['timelong'].mean()  # ip 之前登陆的 timelong 均值
            # TODO new 统计稳定性 res[ci + '_time_std'] = dci['timelong'].std()
            res[ci + '_time_std2'] = dci.loc[:2]['timelong'].std()
        except:
            res[ci + '_time_max2'] = None
            res[ci + '_time_min2'] = None
            res[ci + '_time_mean2'] = None
            res[ci + '_time_std2'] = None

        res[ci + '_cnt'] = dci.shape[0]  # ip 之前登陆的次数
        res[ci + '_cnt_0'] = dci[dci['result'] != 1].shape[0]  # ip 之前登陆失败的次数
        res[ci + '_cnt_1'] = dci[dci['result'] == 1].shape[0]  # ip 之前登陆成功的次数
        res[ci + '_cnt_rate'] = 1.0 * res[ci + '_cnt_0'] / (1 + res[ci + '_cnt'])
        res[ci + '_cnt_1rate'] = 1.0 * res[ci + '_cnt_1'] / (1 + res[ci + '_cnt'])

        for ii in ['id', 'ip', 'device', 'type', 'city']:
            if ii not in ci:
                res[ci + "_" + ii] = dci[ii].unique().size  # ip 之前登陆的 id 数
        res[ci + '_from'] = dci['log_from'].unique().size  # ip 之前登陆的 log_from 数
        res[ci + '_cnt_sec1'] = dci[dci['is_sec'] == True].shape[0]  # ip 之前安全插件登陆次数
        res[ci + '_cnt_sec0'] = dci[dci['is_sec'] == False].shape[0]  # ip 之前非安全插件登陆次数
        res[ci + '_cnt_scan1'] = dci[dci['is_scan'] == True].shape[0]  # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0'] = dci[dci['is_scan'] == False].shape[0]  # ip 之前扫码登陆次数
        res[ci + '_cnt_scan0rate'] = 1.0 * res[ci + '_cnt_scan0'] / (1 + res[ci + '_cnt'])
    del d, dci
    print(res)
    return res



def fun(month):
    last_f = '../datas/baseline_month_new{0}'.format(month)
    if os.path.exists(last_f):
        #data = pd.read_csv(last_f)
        pass
    else:
        import time
        start_time = time.time()

        data = t_login[t_login['time'] >= '2015-0{0}-01 00:00:00'.format(month - 1)]
        data = data[data['time'] < '2015-0{0}-01 00:00:00'.format(month + 1)]
        data = data.sort_values('timestamp').reset_index(drop=True)

        dtt = data[data['time'] >= '2015-0{0}-01 00:00:00'.format(month)]
        t_trade_list = dtt.index.tolist()
        ds = [data]*len(t_trade_list)
        arg = zip(t_trade_list,ds)

        pool = Pool(8)
        d = pool.map(baseline_1, arg)
        pool.close()
        pool.join()
        print('time : ', 1.0 * (time.time() - start_time) / 60)
        data = pd.DataFrame(d)
        data.to_csv(last_f, index=None)

if __name__ == '__main__':
    t_login = pd.read_csv('../datas/t_login.csv')
    t_login_test = pd.read_csv('../datas/t_login_test.csv')

    t_login = pd.concat([t_login,t_login_test])

    t_login['timestamp_online'] = t_login['timestamp'] + t_login['timelong']
    t_login['result'] = t_login['result'].map(lambda x: x == 1 and 1 or -1)

    months = [2,3,4,5,6,7]
    pool = MyPool(3)
    pool.map(fun,months)
    pool.close()
    pool.join()








