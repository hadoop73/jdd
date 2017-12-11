# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool

# login 的按月统计信息
test_data = pd.read_csv('../datas/baseline_login3')
test_data1 = pd.read_csv('../datas/baseline_login4')

test_data = pd.concat([test_data,test_data1])
del test_data1

columns = test_data.columns
cols = ["log_id","timelong","device","log_from","ip","city","result","timestamp","type","id","is_scan","is_sec","time"]
cs = [ci for ci in columns if ci not in cols]

t_trade = pd.read_csv('../datas/t_trade.csv')
t_trade = t_trade[t_trade['time']>='2015-04-01 00:00:00']
t_trade = t_trade[t_trade['time']<'2015-05-01 00:00:00']

t_trade['trade_stamp'] = t_trade['time'].map(lambda x:pd.to_datetime(x).value//10**9 - 28800.0)
t_trade['hour'] = t_trade['time'].map(lambda x:pd.Timestamp(x).hour)
t_trade['hour_v'] = t_trade['hour'].map(lambda x:x/6)
t_trade['weekday'] = t_trade['time'].map(lambda x:pd.Timestamp(x).date().weekday())


def baseline(trade):
    rowkey, id, timestamp, is_risk, time = trade
    res = {}
    res['rowkey'] = rowkey
    res['is_risk'] = is_risk

    d = test_data[((test_data['id']==id)&(test_data['timestamp']< timestamp))]
    d = d.sort_values('timestamp',ascending=False) \
                .reset_index(drop=True)

    for i in [0,1,2]:
        css = [ci + "{0}".format(i) for ci in cs]
        names = dict(zip(cs,css))
        try:
            dd = d.loc[i][cs].rename(names)
        except:
            nones = [None] * len(css)
            dd = zip(css,nones)
        res.update(dict(dd))

    print res
    return res


t_trade_list = np.array(t_trade[['rowkey','id','trade_stamp','is_risk','time']]).tolist()


# 如果最近登陆统计存在
last_f = '../datas/trade_login4'
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






