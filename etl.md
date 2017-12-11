

# 数据分析

**交易次数统计**

![](./trade_cnt.png)

**正样本分布**

![正样本分布](./psample.png)

**登陆分布**

![past_data.ipynb](./login_cnt.png)

正负样本比例为：负样本 129076，正样本 3643。

正样本/负样本=0.02822368217174378。


交易表中：没有记录的交易用户数去重后 8447 个。其中正样本 152 个。
负样本 8310 个，其和不相等，所以存在用户既是正样本，又是负样本。

`all_count` 为 0（登录表中没有登陆记录）一共 18530 条，其中负样本 18066 条，正样本 464 条。

`p_count` 为 0（登录表中没有登陆成功记录）一共 19991 条，其中负样本 19493 条，正样本 498 条。



`trade_test` 中 7000 条没有登陆记录，7199 条没有登陆成功记录

|记录数|样本条数|正样本条数|负样本条数|
| :-----:|:-----:|:-----:|:---:
|0| (18530, 6)| (464, 6)| (18066, 6)|
|1| (14722, 6)| (243, 6)| (14479, 6)|
|2| (10170, 6)| (220, 6)| (9950, 6)|


- `baseline_part` 产生训练数据的 `feas` 保存在 `baseline_1part456`,
    - 再通过 `baseline_feas_train.ipynb` 生成 `login` 文件 `baseline_feas_train`
    - 再通过 `baseline_feas` 生成特征 `baseline_feas_train1`

- `baseline_part_test` 产生测试数据的 `feas` 保存在 `baseline_1part_test`，
    - 再通过 `baseline_feas.ipynb` 生成可以预测的特征文件 `baseline_feas`，
    - 再通过 `baseline_feas_test` 生成特征 `baseline_feas_test`


- `baseline_part_test` 产生 `test` `trade` 的特征信息，保存在 `trade_baseline_3_test`

- `baseline_part_test` 产生 `train` `trade` 的特征信息，保存在 `trade_baseline_3_train`

- `trade_login.py` 合并 `login` 和 `trade` 数据生成 `login_trade` 数据

## 重新生成 `login`,`trade` 数据的特征

**(效果不好)** 这里的数据是统计一个月的区间，还可以减少区间大小，`5 min,1h,1day,3day,7day` 进行统计

- `feas/trade_login_feas` 4-7 月产生 `trade` 之前一个时间区间内 `login` 的特征保存在 `trade_login_all`

 **训练数据和验证数据集一起生成，在模型训练时候才进行拆分**，不过数据的生成过程需要指定时间窗口。

- `feas/month_login[4/5/6/7].py` 分别生成 4-7 月的登陆特征

- `feas/merge_login.py` 把原 `login` 信息拼接单个月生成的特征 `datas/baseline_login_train`,
    `datas/baseline_login_val`，`datas/baseline_login_test`

- `feas/month_trade[4/5/6/7].py` 分别生成 4-7 月的交易特征

- `feas/merge_trade.py` 把原 `login` 信息拼接单个月生成的特征 `datas/baseline_trade_train`,
    `datas/baseline_trae_val`，`datas/baseline_trade_test`

- 在 `trade_feas_with_login[4-7].py` 为每一条 `trade` 数据拼接 `login` 的原数据。

- 在 `trade_login_merge.py` 合并 `login` 和 `trade` 生成训练预测数据 `all3.csv`

- [ ] 需要拼原 `login` 信息，在根据 `rowkey` 拼 `trade` 数据

## 新数据
- `basline` 产生的 `basline` 数据

- `new_feas/feas3.py` 产生登陆，交易前的一段时间内的统计，生成特征保存在 `feas_login_more`

- `new_feas/feas3.py` 产生登陆，交易前的一段时间内的统计，生成特征保存在 `feas_login_new`

- `new_feas/feas_month.py` 产生登陆，交易前的 30 天时间内的统计，生成特征保存在 `feas_month`

- `new_feas/feas_month2.py` 产生登陆，交易前的 60 天时间内的统计，生成特征保存在 `feas_month2`

- `new_feas/feas3month.py` 对 `new_feas/feas3.py` 的时间区间限制为 2 months 生成文件 `feas_login_new3`

- `new_feas/feas3month3.py` 使用 3 `months` 生成特征 `feas_login_new3_3`，比 `new_feas/feas3month.py` 多 3 月份数据

- `new_feas/fes_trade_login.py` 使用 3 `months`(多了3月份) 生成 `ip`,`device`,`id` 历史统计特征 `feas_trade_login_month3`

- `new_feas/feas1month3.py` 对 `feas1.py` 进行调整，往前看 60 天，生成文件 `feas_new_month3`

- `new_feas/feas_month2_all.py` 对 `new_feas/feas_month2.py` 进行重构，

- `new_feas/feas_all.py` 对 `all3.csv` 进行重构，生成文件 `all_3_7`


- `new_feas/feas3month1.py` 对 `feas3month1.py` 进行了重构，生成文件 `feas_login_new31` topn 效果不好

- `feas1more.py` 对 `feas1.py` 进行了重构生成 `feas_new_more`，效果不太好 

- `new_feas/feas_month2.py` 代码重构 `feas_month_3_7`

- `feasB/feasB1.py` 主要用于精细刻画登陆信息，保存文件于 `feasb1`

- `feasB/feasB2.py` 为了提升 `b/lgb_24.py` 生成文件 `feasb2`


- `feasBmonth3/feas_a.py` 分析了 `feasBmonth3/lgb30_22_42.py` 结果生成 `feas_a`



- 分析案例记录在 `案例分析.md` 中，在 `new_feas/feas1.py` 中重新构造特征。
构造的特征保存在 `feas_new`，与 `all3.csv` 合并在 `models/lgb30.py` 中训练有提升到 30。
 
 - `time_diff1`,`time_diff1`,`time_cnt3600`,`time_cnt300`,`time_cnt1800` 特征重要性提升。

- `new_feas/feas2.py` 新增一列特征，保存在 `feas_new_sum` 中。

`lgb` 预测结果能够到达 `0.912157`,top 40 有几个负样本需要调整，代码保存在 `mid/lgb912_40`,生成文件为 `lgb60_month_1_s`


#  模型预测
对于同一批数据:
- 训练数据为 [2015-04-01,2015-06-01)
- 预测数据为 [2015-06-01,2015-07-01)
- 验证数据为 [2015-07-01,2015-08-01)

|模型|训练数据|预测数据|树颗数|topN效果 | 参数
|:---:|:---:|:---:|:---:|:---:|:---:
|lgb|0.905-0.999|0.747-0.907|80|很好|num_leaves:61,rate=0.1|
|lgb|0.86-0.984|0.719-0.89|80|好|同下|
|lgb|0.874-0.997|0.716-0.905|100|好|num_leaves:31,rate=0.1|
|adboost|0.9855|0.9095|100|最好
|et|0.97|0.77|100
|xgb|0.785-|0.669-|80
|xgb|0.85-0.999|0.72-0.912|100|最好
|rf|097|0.805|100

**对原训练数据进行筛选，lgb 的效率提升**

|模型|训练数据|树颗数|topN效果 | 参数
|:---:|:---:|:---:|:---:|:---:
|lgb|0.6筛选列|80|13|num_leaves:61,rate=0.1|
|lgb|0.86-0.984|80|22(2/34)|同下|
|lgb|0.874-0.997|100|好|num_leaves:31,rate=0.1|
|adboost|0.9855|100|最好
|et|0.97|100
|xgb|0.785-|80
|xgb|0.788-0.96|100|最好
|rf|097|100

# 提交记录

|序号|1的个数|线上得分|线下|过程
|:---:|:---:|:---:|:---:|:---
|1|84|0.784|0.804|merge_test.ipynb 结合xgb和adboost 结果，可能准确率下降了，提升不高


# 模型融合

`merge.py` 融合方法中，方法一的结果更优。

- `xgb_40_filter_yval_pred`
- `adboost_80_filter_yval_pred`
- `lgb80_filter_yval_pred`

方法一融合 0.902681，topN 27 


# 分析案例

- 分析案例记录在 `案例分析.md` 中，在 `new_feas/feas1.py` 中重新构造特征。
构造的特征保存在 `feas_new`，与 `all3.csv` 合并在 `models/lgb30.py` 中训练有提升到 30。
 
 - `time_diff1`,`time_diff1`,`time_cnt3600`,`time_cnt300`,`time_cnt1800` 特征重要性提升。

- `new_feas/feas2.py` 新增一列特征，保存在 `feas_new_sum` 中。

`lgb` 预测结果能够到达 `0.912157`,top 40 有几个负样本需要调整，代码保存在 `mid/lgb912_40`,生成文件为 `lgb60_month_1_s`



## 合并

- `feas/sub_merge_n.py` 选 `b/lgb_30.py` 产生的 top77 个结果 `lgb60_more_yval_pred` 和 `b/lgb_24.py` 产生的 top58 结果 `lgb60_month_b_id_yval_pred` 进行合并，产生一个数据集，再用其他模型对结果进行打分。

- 合并的结果保存在 `ans` 中，总共 100 个 `rowkey` 其中负样本 27 个，正样本 73 个。

- `b/data_filter.py` 评估筛选的 100 个结果

- `b/lgb_ans.py` 用于对 `b/data_filter.py` 中结果进行最有排序

## 减少特征

删除一些不需要的特征同样能起到提升效果。删除重要性为 0 的数据，数据减少了有时候也会影响结果

