# coding:utf-8

import pandas as pd

sub1 = pd.read_csv('../datas/abc.csv',header=None)
sub1.columns = ['rowkey','p']

print sub1.head()
print sub1.shape

sub2 = pd.read_csv('../datas/sub.csv',header=None)
sub2.columns = ['rowkey','p']

a1 = sub1[sub1['p']==1]['rowkey'].values.tolist()
print len(a1)


a2 = sub2[sub2['p']==1]['rowkey'].values.tolist()
r = [i for i in a2 if i in a1]
print len(a2)
print 'common: ',len(a2) - len(r)

sa = set(a1+a2)
print len(sa)

sub1['p'] = sub1['rowkey'].map(lambda x:1 if x in sa else 0)

d = sub1

print d[d['p']==1].shape
print d.head()
print d.shape

d.to_csv('../datas/abc_sub.csv',index=None,header=None)
