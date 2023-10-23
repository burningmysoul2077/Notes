# 1. 赛事信息

## 1.1 大赛官网

- [2023“SEED”第四届江苏大数据开发与应用大赛--新能源赛道](https://www.marsbigdata.com/competition/details?id=40144958741)

## 1.2 赛题描述

-  AI +新能源：电动汽车充电站充电量预测

`在电动汽车充电站运营管理中，准确预测充电站的电量需求对于提高充电站运营服务水平和优化区域电网供给能力非常关键。本次赛题旨在建立站点充电量预测模型，根据充电站的相关信息和历史电量数据，准确预测未来某段时间内充电站的充电量需求。 在赛题数据中，我们提供了电动汽车充电站的场站编号、位置信息、历史电量等基本信息。我们鼓励参赛选手在已有数据的基础上补充或构造额外的特征，以获得更好的预测性能。参赛者需要基于这些数据，利用人工智能相关技术，建立预测模型来预测未来一段时间内的需求电量，帮助管理者提高充电站的运营效益和服务水平，促进电动汽车行业的整体发展。`

## 1.3 赛题任务

-  `根据赛题提供的电动汽车充电站多维度脱敏数据，构造合理特征及算法模型，预估站点未来一周每日的充电量。`

## 1.4 数据描述

- `本赛题提供的数据集包含三张数据表。`
- ` 其中，power_forecast_history.csv 为站点运营数据，power.csv为站点充电量数据，stub_info.csv为站点静态数据，训练集为历史一年的数据，测试集为未来一周的数据。`

<<<<<<< HEAD
集清单与格式说明：

=======
>>>>>>> efe655a24e2780b8c0246f9bb8a8b6b661db9091
![](https://file.public.marsbigdata.com/2023/09/28/62dfR3wroe_f0osL.png)

## 1.5 评估指标

-  评估指标：![](https://file.public.marsbigdata.com/2023/09/28/5dxrlBwhywzdmY5n.png)

![](https://file.public.marsbigdata.com/2023/09/28/w067iymIh8upId22.png)

- ![](https://file.public.marsbigdata.com/2023/09/28/4lzjixV2X0s1RKZK.png)式中为第个数据的真实值，![](https://file.public.marsbigdata.com/2023/09/28/c0Mdu1uG_rYP8umy.png)为第个数据的预测值，n 为样本总数。


### 提交

-  `选手需要根据给定的测试集，预测出目标变量值（站点充电量），并以csv格式（列形式）保存，文件名为result.csv。`

-  初赛提交样例如下：
- 初赛提交样例如下：

![](https://file.public.marsbigdata.com/2023/09/28/HlXuJ03K2Xl9N2mF.png)


<<<<<<< HEAD
=======
![](https://file.public.marsbigdata.com/2023/09/28/HlXuJ03K2Xl9N2mF.png)
>>>>>>> efe655a24e2780b8c0246f9bb8a8b6b661db9091

---

# 2 时间序列学习

-  统一更新到  ![时间序列总结(成长中...)](https://github.com/burningmysoul2077/Notes/blob/main/%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97/%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%80%BB%E7%BB%93(%E6%88%90%E9%95%BF%E4%B8%AD...).md)

---

# 3 赛题学习
<<<<<<< HEAD

## 学习 baseline

- 已跑通 [ 一键运行](https://aistudio.baidu.com/projectdetail/6882171?sUid=2554132&shared=1&ts=1697254726362)

-  pip 安装 lightgbm

```python
!pip install -U lightgbm
# _-U_就是 --upgrade,意思是如果已安装就升级到最新版
```
=======
>>>>>>> efe655a24e2780b8c0246f9bb8a8b6b661db9091
