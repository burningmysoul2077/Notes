# Tianchi-安泰杯跨境电商智能算法大赛(高质量baseline)

## 竞赛题目

>  AliExpress 是中国最大出口B2C电商平台，2010 年平台成立至今已过 8 年，高速发展，日趋成熟。覆盖全球 230 个国家和地区，支持世界 18 种语言站点，22 个行业囊括日常消费类目；目前的主要交易市场为俄、美、西、巴、法等国。

> 对于 AliExpress 来说，目前某些国家的用户群体比较成熟。这些成熟国家的用户沉淀了大量的该国用户的行为数据。被挖掘利用后形成我们的推荐算法，用来更好的服务于该国用户。

> 但是还有一些待成熟国家的用户在 AliExpress 上的行为比较稀疏，对于这些国家用户的推荐算法如果单纯不加区分的使用全网用户的行为数据，可能会忽略这些国家用户的一些独特的心智；而如果只使用这些国家的用户的行为数据，由于数据过于稀疏，不具备统计意义，会难以训练出正确的模型。于是怎样利用已成熟国家的稠密用户数据和待成熟国家的稀疏用户数据训练出对于待成熟国家用户的正确模型对于我们更好的服务待成熟国家用户具有非常重要的意义。

> 本次比赛给出若干日内来自成熟国家的部分用户的行为数据，以及来自待成熟国家的A部分用户的行为数据，以及待成熟国家的B部分用户的行为数据去除每个用户的最后一条购买数据，让参赛人预测B部分用户的最后一条行为数据。

## 竞赛数据

---

# Baseline 分析探索

## Baseline 1

-  来源： https://github.com/RainFung/Tianchi-AntaiCup-International-E-commerce-Artificial-Intelligence-Challenge/tree/master
-  代码

### 代码片段：导入所需的包

```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import gc
%matplotlib inline
# 禁用科学计数法
pd.set_option('display.float_format',lambda x : '%.2f' % x)
```

-  在 Jupyter Notebook 中可以使用代码禁用 Numpy 和 Pandas 的科学计数法。

```python
#禁用科学计数法
np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=150)  
pd.set_option('display.float_format',lambda x : '%.2f' % x)
```

### 代码片段：数据预处理

```python
# 在训练和测试集中分别增加一列标识 is_train
# 然后合并训练集和测试集
df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])

# 提取订单生成日期的年月日时
# `Series.dt.date`
df['create_order_time'] = pd.to_datetime(df['create_order_time'])
df['date'] = df['create_order_time'].dt.date
df['day'] = df['create_order_time'].dt.day
df['hour'] = df['create_order_time'].dt.hour

# 将数据集df关联到商品数据集item，left join on item_id
df = pd.merge(df, item, how='left', on='item_id')
```
  `pandas.merge(_left_, _right_, _how='inner'_, _on=None_, _left_on=None_, _right_on=None_, _left_index=False_, _right_index=False_, _sort=False_, _suffixes=('_x', '_y')_, _copy=True_, _indicator=False_, _validate=None_)`

-  **left** ：DataFrame
- **right**： DataFrame or named Series，Object to merge with.
- **how**：  {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’

```python
# 计算内存占用
memory = df.memory_usage().sum() / 1024**2 
print('Before memory usage of properties dataframe is :', memory, " MB")

# 转化每列的数据类型为可存储的最小值，目的是减少内存消耗
dtype_dict = {'buyer_admin_id' : 'int32', 
              'item_id' : 'int32', 
              'store_id' : pd.Int32Dtype(),
              'irank' : 'int16',
              'item_price' : pd.Int16Dtype(),
              'cate_id' : pd.Int16Dtype(),
              'is_train' : 'int8',
              'day' : 'int8',
              'hour' : 'int8',
             }

df = df.astype(dtype_dict)
memory = df.memory_usage().sum() / 1024**2 
print('After memory usage of properties dataframe is :', memory, " MB")
del train,test; 
gc.collect()
```

-  `DataFrame.memory_usage(_index=True_, _deep=False_)`  [[source]](https://github.com/pandas-dev/pandas/blob/v1.2.5/pandas/core/frame.py#L2711-L2803)[](https://pandas.pydata.org/pandas-docs/version/1.2/reference/api/pandas.DataFrame.memory_usage.html#pandas.DataFrame.memory_usage "Permalink to this definition")
	-  Return the memory usage of each column in bytes.

-  DataFrame.astype(_dtype_, _copy=True_, _errors='raise'_)[[source]](https://github.com/pandas-dev/pandas/blob/v1.2.5/pandas/core/generic.py#L5724-L5887)[¶](https://pandas.pydata.org/pandas-docs/version/1.2/reference/api/pandas.DataFrame.astype.html#pandas.DataFrame.astype "Permalink to this definition")
	-  Cast a pandas object to a specified dtype `dtype`.
- **dtype**: data type, or dict of column name -> data type
- **copy**: bool, default True,是否返回复本、
- **errors**： {‘raise’, ‘ignore’}, default ‘raise’

- 通过观察下图中的数据集字段取值范围，确定存储类型的最小值

![[Pasted image 20240528215044.png]]

```python
# 保存为hdf5格式文件，加速读取
for col in ['store_id', 'item_price', 'cate_id']:
    df[col] = df[col].fillna(0).astype(np.int32).replace(0, np.nan)
df.to_hdf('./data/train_test.h5', '1.0')
```

-  [HDF5](https://docs.hdfgroup.org/hdf5/v1_14/v1_14_4/_intro_h_d_f5.html),即 Hierarchical Data Format version 5，用于存储和管理大量数据的文件格式
-   An HDF5 file (an object in itself) can be thought of as a container (or group) that holds a variety of heterogeneous data objects (or datasets). The datasets can be images, tables, graphs, and even documents, such as PDF or Excel
-  HDF5 可以看作是 `dataset` 和 `group` 二合一的容器
	- dataset : 数据集，像 numpy 数组一样工作
	- group : 包含了其它 dataset 和 其它 group, 每个 HDF5 文件都有一个 root group

```python
# 在单元格的开头添加%%time ，单元格执行完成后，会输出单元格执行所花费的时间
%%time
```

### 代码片段：数据整体内容分析

```python
# 默认返回前五行数据
df.head()

# Null 空值统计
for pdf in [df, item]:
    for col in pdf.columns:
        print(col, pdf[col].isnull().sum())
# 统计学描述       
df.describe()
item.describe()
```

### 代码片段：数据探查

#### 数据集与测试集

```python
# 分别计算训练和测试数据集的样本总数
train_count = len(df[train])
print('训练集样本量是',train_count)
test_count = len(df[test])
print('测试集样本量是',test_count)
print('样本比例为：', train_count/test_count)
```