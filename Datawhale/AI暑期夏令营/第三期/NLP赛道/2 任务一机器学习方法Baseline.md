# 1 Requirements

## Pandas 介绍

>  Pandas 是 Python 语言的一个扩展程序库，用于数据分析

-  _Pandas 是一个开放源码、BSD 许可的库，提供高性能、易于使用的数据结构和数据分析工具。_
-  _Pandas 名字衍生自术语 "panel data"（面板数据）和 "Python data analysis"（Python 数据分析）。_
-  _Pandas 一个强大的分析结构化数据的工具集，基础是 [Numpy](https://www.runoob.com/numpy/numpy-tutorial.html)（提供高性能的矩阵运算）。_
-  _Pandas 可以从各种文件格式比如 CSV、JSON、SQL、Microsoft Excel 导入数据。_
-  _Pandas 可以对各种数据进行运算操作，比如归并、再成形、选择，还有数据清洗和数据加工特征。_
-  Pandas 广泛应用在学术、金融、统计学等各个数据分析领域。

## Scikit-learn 模块介绍

-  Sklearn 是一个机器学习、深度学习中非常常用的 Python 第三方库，内部封装了多种机器学习算法与数据处理算法，提供了包括数据清洗、数据预处理、建模调参、数据验证、数据可视化的全流程功能，是入门机器学习的必备工具。

-  通过使用 sklearn，你可以便捷地完成机器学习的整体流程，尝试使用多种模型完成训练与预测任务，而不需要再手动实现各种机器学习算法。本次基础 Baseline 的各个部分都将使用 sklearn 封装的算法来完成，我们在此处简单介绍 sklearn 的安装、使用与本次用到的相关算法与原理。如想要进一步学习 sklearn 库的用法，可以参见该社区：[https://​scikit​-learn​.org​.cn​/。](https://scikit-learn.org.cn/%E3%80%82)

---

# 2 特征提取

-  特征提取是机器学习任务中的一个重要步骤。我们将 __训练数据的每一个维度称为一个特征__，例如，如果我们想要基于二手车的品牌、价格、行驶里程数三个变量来预测二手车的价格，则品牌、价格、行驶里程数为该任务的三个特征。
- __所谓特征提取，即从训练数据的特征集合中创建新的特征子集的过程__。_提取出来的特征子集特征数一般少于等于原特征数_，但能够更好地表征训练数据的情况，使用提取出的特征子集能够取得更好的预测效果。对于 NLP、CV 任务，我们通常需要将文本、图像特征提取为计算机可以处理的数值向量特征。我们一般可以使用 sklearn 库中的 feature_extraction 包来实现文本与图片的特征提取。

-  __在 NLP 任务中，特征提取一般需要将自然语言文本转化为数值向量表示__，常见的方法包括基于 TF-IDF（词频-逆文档频率）提取或基于 BOW（词袋模型）提取等，两种方法均在 sklearn.feature_extraction 包中有所实现。

## 2.1 基于 TF-IDF

-  TF-IDF (term frequency–inverse document frequency) 是一种用于信息检索与数据挖掘的常用加权技术，
	-   其中，TF 指 term frequence，即词频，指某个词在文章中出现次数与文章总次数的比值；
	-   IDF 指 inverse document frequence，即逆文档频率，指包含某个词的文档数占语料库总文档数的比例。
-  例如，假设语料库为 {"今天 天气 很好", "今天 心情 很 不好", "明天 天气 不好"}，每一个句子为一个文档，则“今天”的 TF 和 IDF 分别为：
	-  $TF(今天∣文档1）= \frac{词在文档一的出现频率}{文档一的总词数} = \frac{1}{3}​$
	- $TF(今天∣文档2）= \frac{词在文档二的出现频率}{文档二的总词数}=\frac{1}{4}$
	- $TF(今天∣文档3）=0$
	-  $IDF(今天）= \log \frac{语料库文档总数}{出现该词的文档数}=\log\frac{2}{3}$​

-  每个词最终的 IF-IDF 即为 TF 值乘以 IDF 值。
-   计算出每个词的 TF-IDF 值后，使用 TF-IDF 计算得到的数值向量替代原文本即可实现基于 TF-IDF 的文本特征提取。

-  我们可以使用 sklearn.feature_extraction.text 中的 TfidfVectorizer 类来简单实现文档基于 TF-IDF 的特征提取：

```python
# 首先导入该类
from sklearn.feature_extraction.text import TfidfVectorizer

# 假设我们已从本地读取数据为 DataFrame 类型，并已经经过基本预处理,data 为已处理的 DataFrame 数据
# 实例化一个 TfidfVectorizer 对象，并使用 fit 方法来拟合数据
vector = TfidfVectorizer().fit(data["text"])

# 拟合之后，调用 transform 方法即可得到提取后的特征数据
train_vector = vector.transform()
```

## 2.2 基于 BOW

-  BOW (Bag of Words) 是一种常用的文本表示方法，其基本思想是假定对于一个文本，忽略其次序和语法、句法，仅仅将其看做是一些词汇的集合，而文本中的每个词汇都是独立的。
-  简单说，就是将每篇文档都看成一个袋子，然后看这个袋子里装的都是些什么词汇，将其分类。
-  具体而言，词袋模型表示一个文本，首先会维护一个词库，词库里维护了每一个词到一个数值向量的映射关系。

-  例如，最简单的映射关系是独热编码，假设词库里一共有四个词，"今天、天气、很、不好"，那么独热编码会将四个词分别编码为：
	-  今天——（1,0,0,0）
	-  天气——（0,1,0,0）
	-  很 ——（0,0,1,0）
	-  不好——（0,0,0,1）

-  而使用词袋模型，就会将上述这句话编码为：
	-  $BOW(Sentence) = Embedding(今天) + Embedding(天气) + Embedding(很) + Embedding(不好) = (1,1,1,1)$

-  我们一般使用 sklearn.feature_extraction.text 中的 CountVectorizer 类来简单实现文档基于频数统计的 BOW 特征提取，其主要方法与 TfidfVectorizer 的主要使用方法一致：

```python
# 首先导入该类
from sklearn.feature_extraction.text import CountVectorizer

# 假设我们已从本地读取数据为 DataFrame 类型，并已经过基本预处理，data 为已处理的 DataFrame 数据
# 实例化一个 CountVectorizer 对象，并使用 fit 方法来拟合数据
vector = CountVectorizer().fit(data["text"])

# 拟合之后，调用 transform 方法即可得到提取后的特征数据
train_vector = vector.transform()
```

## 2.3 Stop-words

-  停用词是 NLP 的一个重要工具，通常被用来提升文本特征的质量，或者降低文本特征的维度。
-  _当使用 TF-IDF 或 BOW 模型来表示文本时，总会遇到一些问题。_
	-  在特定 NLP 任务中，一些词语不能提供有价值的信息作用、可以忽略。这种情况在生活里也非常普遍。
-  以本次学习任务为例，我们希望医学类的词语在特征提取时被突出，对于不是医学类词语的数据就应该考虑让他在特征提取时被淡化，同时一些日常生活中使用频率过高而普遍出现的词语，我们应该选择忽略这些词语，以防对我们的特征提取产生干扰。
-  举个例子，我们依然以讲解 BOW 模型时举得这个例子介绍：
	-  $BOW(Sentence) = Embedding(今天) + Embedding(天气) + Embedding(很) + Embedding(不好) = (1,1,1,1)$
-   当我们需要对这句话进行情感分类时，我们就需要突出它的情感特征，也就是我们希望 "不好" 这个词在经过 BOW 模型编码后的值能够大一点。
-  但是如果我们不使用停用词，那么 “今天天气很好还是不好” 这句话经过 BOW 模型编码后的值就会与上面这句话的编码高度相似，从而严重影响模型判断的结果。

-  那么如何使用停用词解决这个问题呢？
	-  理想一点，我们将除了情感元素的词语全部停用，也就是编码时不考虑，仅保留情感词语，也就是判断句子中 "好" 这个词出现的多还是少，很明显 "好" 这个词出现的多，那情感显然是政现象的

-  本次任务中，日常生活中出现的词语可能都对模型分类很难有太大帮助，比如连词 or again and 等。

### 使用 stop-words 文本文件

```python
# 读取该文件
stops =[i.strip() for i in open(r'stop.txt',encoding='utf-8').readlines()]
```

-  读取文件后在使用 CountVectorizer() 方法时指定 stop_words 参数为 stops 就可以:

```python
vector = CountVectorizer(stop_words=stops).fit(train['text'])
```

---

# 3 划分数据集

-   __在机器学习任务中，我们一般会有三个数据集：训练集、验证集、预测集__。
	-   训练集 train为我们训练模型的拟合数据，是我们前期提供给模型的输入；
	-   验证集 validation 一般是我们自行划分出来验证模型效果以选取最优超参组合的数据集；
	-   测试集 (预测集) test 是最后检验模型效果的数据集。
-   例如在本期竞赛任务中，比赛方提供的 test.csv 就是预测集，我们最终的任务是建立一个模型在预测集上实现较准确的预测。但是预测集一般会限制预测次数，例如在本期比赛中，每人每天仅能提交三次，但是我们知道，机器学习模型一般有很多超参数，为了选取最优的超参组合，我们一般需要多次对模型进行验证，即提供一部分数据让已训练好的模型进行预测，来找到预测准确度最高的模型。

-  因此，我们 _一般会将比赛方提供的训练集也就是 train.csv 进行划分，划分为训练集和验证集_。我们会使用划分出来的训练集进行模型的拟合和训练，而使用划分出来的验证集验证不同参数及不同模型的效果，来找到最优的模型及参数再在比赛方提供的预测集上预测最终结果。

-  __划分数据集的方法有很多，基本原则是同分布采样__。即我们划分出来的验证集和训练集应当是同分布的，以免验证不准确（事实上，最终的预测集也应当和训练集、验证集同分布）。
-  此处我们介绍 _交叉验证_，即对于一个样本总量为 T 的数据集，我们一般随机采样 10%~20%（也就是 0.1T~0.2T 的样本数）作为验证集，而取其他的数据为训练集。如要了解更多的划分方法，可查阅该博客：[https://​blog​.csdn​.net​/hcxddd​/article​/details​/119698879]。
-  我们可以使用 sklearn.model_selection 中的 train_test_split 函数便捷实现数据集的划分：

-  baseline中并没有划分验证集，你也可以自行划分验证集来观察训练中的准确率。

```python
from sklearn.model_selection import train_test_split

# 该函数将会根据给定比例将数据集划分为训练集与验证集
trian_data, eval_data = train_test_split(data, test_size = 0.2)
# 参数 data 为总数据集，可以是 DataFrame 类型
# 参数 test_size 为划分验证集的占比，此处选择 0.2，即划分 20% 样本作为验证集
```

---

# 4 选择机器学习模型

-  我们可以选择多种机器学习模型来拟合训练数据，不同的业务场景、不同的训练数据往往最优的模型也不同。_常见的模型包括线性模型、逻辑回归、决策树、支持向量机、集成模型、神经网络等_。想要深入学习各种机器学习模型的同学，推荐学习《西瓜书》或《统计学习方法》。

-  Sklearn 封装了多种机器学习模型，常见的模型都可以在 sklearn 中找到，sklearn 根据模型的类别组织在不同的包中，此处介绍几个常用包：

	-  sklearn.linear_model：线性模型，如线性回归、逻辑回归、岭回归等
	-  sklearn.tree：树模型，一般为决策树
	-  sklearn.neighbors：最近邻模型，常见如 K 近邻算法
	-  sklearn.svm：支持向量机
	-  sklearn.ensemble：集成模型，如 AdaBoost、GBDT等

## 4.1 Logistic Regression

-  本案例中，我们使用简单但拟合效果较好的逻辑回归模型作为 Baseline 的模型。此处简要介绍其原理。

-  逻辑回归模型，即 Logistic Regression，实则为一个线性分类器，通过 Logistic 函数 (或 Sigmoid 函数)，将数据特征映射到 0～1 区间的一个概率值（样本属于正例的可能性），通过与 0.5 的比对得出数据所属的分类。逻辑回归的数学表达式为：
	-  $f(z) = \frac{1}{1+ e^{-z}}$
	-  $z = w^Tx + w_0$

-  逻辑回归模型简单、可并行化、可解释性强，同时也往往能够取得不错的效果，是较为通用的模型。

-  我们可以使用 sklearn.linear_model.LogisticRegression 来调用已实现的逻辑回归模型：

```python
# 引入模型
model = LogisticRegression()
# 可以在初始化时控制超参的取值，此处使用默认值，具体参数可以查阅官方文档

# 开始训练，这里可以考虑修改默认的 batch_size 与 epoch 来取得更好的效果
# 此处的 train_vector 是已经经过特征提取的训练数据
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测，此处的 test_vector 同样是已经经过特征提取的测试数据
test['label'] = model.predict(test_vector)
```

-  __事实上，sklearn 提供的多种机器学习模型都封装成了类似的类，绝大部分使用方法均和上述一致，即先实例化一个模型对象，再使用 fit 函数拟合训练数据，最后使用 predict 函数预测测试数据即可。__

---

# 5 数据探索

-  __数据探索性分析，是通过了解数据集，了解变量间的相互关系以及变量与预测值之间的关系__，对已有数据在尽量少的先验假设下通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法，从而帮助我们后期更好地进行特征工程和建立模型，是机器学习中十分重要的一步。

-  本次 baseline 实践中，使用 pandas 来读取数据以及数据探索。

## 5.1 使用 pandas 读取数据

### pd.read_csv()

-  在这部分内容里我们利用`pd.read_csv（)` 方法对赛题数据进行读取，`pd.read_csv（)` 参数为需要读取的数据地址，读取后返回一个DataFrame 数据。

```python
import pandas as pd
train = pd.read_csv('./基于论文摘要的文本分类与关键词抽取挑战赛公开数据/train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('./基于论文摘要的文本分类与关键词抽取挑战赛公开数据/testB.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')
```

### DataFrame.apply(len).describe()

-  通过pandas提供的一些方法，我们可以在本地快速查看数据的一些特征

通过 `DataFrame.apply(len).describe()` 方法查看数据长度

```python
print(train['text'].apply(len).describe())
```

```text
count     6000.000000
mean      1620.251500
std        496.956005
min        286.000000
25%       1351.750000
50%       1598.500000
75%       1885.000000
max      10967.000000
Name: text, dtype: float64
```

-  观察输出发现数据长度平均值在 1620 左右

### DataFrame.value_counts()

-  通过 `DataFrame.value_counts()` 方法查看数据数量

```text
print(train["label"].value_counts())
```

```text
label
0    3079
1    2921
Name: count, dtype: int64
```

-  观察输出发现 0-1 标签分布的比较均匀，也就是说我们不必担心数据分布不均而发生过拟合，保证模型的泛化能力。

---

# 6 数据清洗

-  __数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已__。俗话说：garbage in, garbage out。分析完数据后，特征工程前，必不可少的步骤是对数据清洗。

-  数据清洗的作用是利用有关技术如数理统计、数据挖掘或预定义的清理规则将脏数据转化为满足数据质量要求的数据。主要包括缺失值处理、异常值处理、数据分桶、特征归一化/标准化等流程。

-  同时由于表格中存在较多列，我们将这些列的重要内容组合在一起生成一个新的列方便训练

```python
# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fill
```

---

# 7 特征工程

-  __特征工程指的是把原始数据转变为模型训练数据的过程，目的是获取更好的训练数据特征__。特征工程能使得模型的性能得到提升，有时甚至在简单的模型上也能取得不错的效果。

![[Pasted image 20230821161857.png]]

-  这里我们选择使用BOW将文本转换为向量表示

```python
#特征工程
vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])
```

---

# 8 模型训练与验证

-  特征工程也好，数据清洗也罢，都是为最终的模型来服务的，模型的建立和调参决定了最终的结果。模型的选择决定结果的上限， 如何更好的去达到模型上限取决于模型的调参。

-  建模的过程需要我们对常见的线性模型、非线性模型有基础的了解。模型构建完成后，需要掌握一定的模型性能验证的方法和技巧

```python
# 模型训练
model = LogisticRegression()

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])
```

---

# 9 结果输出

-  提交结果需要符合提交样例结果

```python
# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)
test['Keywords'] = test['title'].fillna('')
# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)
```
