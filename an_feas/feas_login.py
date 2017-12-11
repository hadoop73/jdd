# coding:utf-8

import pandas as pd
import numpy as np
import os,sys
from multiprocessing import Pool

data= pd.read_csv('../datas/improve')
data = data.sort_values('y1',ascending=False).reset_index(drop=True)


s = '''
\nt5 = login[(login['time']<'{0}')]
t5 = t5[(t5['time']>'{1}')]
# {2}
t5[t5['id']=={3}].sort_values('time').reset_index(drop=True)'''


i = 0
while data.loc[i]['is_risk']==1:
    t,id = data.loc[i]['time'],data.loc[i]['id']
    t2 = str(pd.Timestamp(t) - pd.Timedelta(days=5))
    tp2 = str(pd.Timestamp(t) + pd.Timedelta(days=1))
    ts = s.format(tp2,t2,t,id)
    print ts
    i += 1


print '## top n positive sample'


s = '''
\n# ndata
t5 = login[(login['time']<'{0}')]
t5 = t5[(t5['time']>'{1}')]
# {2}
t5[t5['id']=={3}].sort_values('time').reset_index(drop=True)'''
bi = 0
while bi < 10:
    if data.loc[i]['is_risk']==0:
        t,id = data.loc[i]['time'],data.loc[i]['id']
        t2 = str(pd.Timestamp(t) - pd.Timedelta(days=5))
        tp2 = str(pd.Timestamp(t) + pd.Timedelta(days=1))
        ts = s.format(tp2,t2,t,id)
        print ts
        bi += 1
    i += 1

print '-'*20

s = '''
\n# trade
p0 = trade[(trade['id']=={0})&(trade['time']>="{1}")&(trade['time']<"{2}")].sort_values('time',ascending=False).reset_index(drop=True)
# 30 min 以内多次交易，2 min，1min，5min，10min，1h，1day
p0.head(50)'''
i = 0
while data.loc[i]['is_risk']==1:
    t,id = data.loc[i]['time'],data.loc[i]['id']
    t2 = str(pd.Timestamp(t) - pd.Timedelta(days=5))
    tp2 = str(pd.Timestamp(t) + pd.Timedelta(days=1))
    ts = s.format(id,t2,tp2)
    print ts
    i += 1


print '## top n positive sample'


s = '''
\n# ndata trade
p0 = trade[(trade['id']=={0})&(trade['time']>="{1}")&(trade['time']<"{2}")].sort_values('time',ascending=False).reset_index(drop=True)
# 30 min 以内多次交易，2 min，1min，5min，10min，1h，1day
p0.head(50)'''
bi = 0
while bi < 10:
    if data.loc[i]['is_risk']==0:
        t,id = data.loc[i]['time'],data.loc[i]['id']
        t2 = str(pd.Timestamp(t) - pd.Timedelta(days=30))
        tp2 = str(pd.Timestamp(t) + pd.Timedelta(days=1))
        ts = s.format(id, t2, tp2)
        print ts
        bi += 1
    i += 1
