

## 结果提交

|TopN|线上结果|线下merge(1,2,3)|xgb|lgb|adboost|说明|
|:---:|:---:|:---:|:---:|:---:|:---:|:---
|30|0.7436|0.88/0.82/0.82|14/0.98/0.87|12/0.99/0.89|13/0.94/0.86|all.csv,xgb 100,lgb 80,adboost 100|
|30|0.80438|0.881963 (lgb单模型)| |0.881963 / 0.906120(28)| |结果文件：lgb60_month__ytest_pred，文件保存：models/lgb_train.py
|54|0.885162|0.906120| | 单模型大于等于 top 28 概率的所有 rowkey |
|120|0.724574| | | 两个 lgb 方法一融合
|89|0.756191| | | 两个 lgb 融合效果低于 0.885162 |
|61|0.892277	|0.912376| | 单个 lgb30 模型
|77|0.887859|0.927314| | 模型融合(10+30)| |lgb_feas_month.py + lgb30.py
|117|0.909314|0.952322| | 两个模型(30+30)|lgb_feas_month30_23.py + lgb30.py
|103|0.898348| | | 两个模型(30+30) 模型取倒数第五个概率筛选提交数据
|91|0.905276| | | 两个模型(30+30)
|82|0.918385|0.925150| | top26+top30 融合(`models/lgb_24_918.py`)
|100|0.858939|0.936666| | top100(3个模型融合，sub_merge3.py，最后一个是 top51)
|108|0.909376|0.944117|
|93|0.856859 | 0.938288 |top42 | lgb60_month_1_yval_pred 和 lgb60_mk_yval_pred 融合


