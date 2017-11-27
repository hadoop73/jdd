

# 数据分析

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


- `baseline_part_test` 产生 `trade` 的特征信息，保存在 `trade_baseline_3_test`

