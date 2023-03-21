

### 回归VS分类：

- 回归估计一个连续值
- 分类预测一个离散类别


#### 从回归到多类分类：

- 回归：
	- 单连续数值输出
	- 自然区间R
	- 跟真实值的区别作为损失

![[Pasted image 20230320114513.png]]

- 分类：
	- 通常多个输出
	- 输出 $i$ 是预测为第 $i$ 类的置信度

![[Pasted image 20230320114532.png]]

- softmax回归的输出值个数 = 标签中的类别数

##### 均方损失

- 对类别进行一位有效编码
	- $\textbf{y} = [y_{1}, y_{2}, ..., y_{n}]^{T}$
	- $y_{i}=\begin{cases} 1&if \enspace i=y\\ 2&otherwise \end{cases}$

- 使用均方损失训练
- 最大值为预测
	- $\hat{y}=\underset {i}{argmax}\quad o_{i}$

##### 无校验比例

- 对类别进行一位有效编码
- 最大值为预测
	- $\hat{y}=\underset {i}{argmax}\quad o_{i}$
- 需要更置信的识别正确类（大余量），因为我们并不关注数值是多少，更关注置信度
	- $o_y - o_i \geq \Delta(y, i)$
	- 这样可以将正确的类和其他类区分开

##### 校验比例

- 输出匹配概率（非负，和为1）
	- $\hat{\textbf{y}} = softmax(\textbf{o})$
	- $\hat{y_i} = \frac{exp(o_i)}{\sum_{k} exp(o_k)}$
- 真实概率  $\textbf{y}$ 和 $\hat{\textbf{y}}$ 的区别作为损失

#### Softmax和交叉熵损失

- 交叉熵用来衡量两个概率的区别 $H(p,q) = \sum\limits_{i} -p_{i}log(q_i)$
	- 将它作为损失
	- $l(\textbf{y}, \hat{\textbf{y}}) = -\sum\limits_{i}y_{i}log\hat{y_{i}} = -log\hat{y_y}$
		- 因为只有一类 y 为1，其他全部为0，本质就是对真实类别的 y 求log，再求负
		- 不关于对非正确类的预测值，只关心正确类预测值的置信度
- 其梯度是真实概率和预测概率的区别
	- $\partial_{o_{i}} l(y, \hat{y}) = softmax(\textbf{o})_{i} - y_{i}$

### 总结

- Softmax回归是一个多类分类模型
- 使用Softmax操作子得到每个类的预测置信度，和为1的概率
- 使用交叉熵来衡量和预测标号的区别

### 损失函数

#### L2 Loss 

- $l(y, y^{'}) = \frac{1}{2}(y - y^{'})^2$

- 蓝色表示当y=0，y'的函数
- 绿色是它的似然函数
- 橙色是损失函数的梯度，穿过原点的一次函数
![[Pasted image 20230320152401.png]]

> 随着预测值与真实值越来越接近，梯度会随着结果逼近而下降

#### L1 Loss

- $l(y, y^{'}) = \lvert y - y^{'}\rvert$

![[Pasted image 20230320152703.png]]

> 梯度保持不变，但在 0 处不可导、梯度随机
> 当预测值跟真实值比较远时，梯度是常数，好处是稳定，坏处是 0 处不可导，似然函数不平滑

#### Huber's Robust Loss

![[Pasted image 20230320152933.png]]

> 结合L1 Loss 和 L2 Loss的优点
> 当预测值与真实值相差很大时，是一个绝对值误差，在这里在减去 $\frac{1}{2}$ 为的是让曲线连接起来
> 当预测值与真实值比较接近，是均方误差


### 增：信息论基础

- _信息论 information theory_ ：涉及编码、解码、发送以及尽可能简洁地处理信息或数据

#### 熵

- 信息论的核心思想是量化数据中的信息内容
- 在信息论中，该数值被称为分布 𝑃 的 **熵 entropy**
- 可以通过以下方程得到：
	- $𝐻[𝑃] = ∑𝑗−𝑃(𝑗)log𝑃(𝑗)$

- 信息论的基本定理之一指出，为了对从分布𝑝�中随机抽取的数据进行编码， 我们至少需要𝐻[𝑃]�[�]“纳特（nat）”对其进行编码。 “纳特”相当于_比特_（bit），但是对数底为𝑒�而不是2。因此，一个纳特是1log(2)≈1.441log⁡(2)≈1.44比特。

### 信息量[](http://localhost:8888/notebooks/Py_Code/d2l-zh/pytorch/chapter_linear-networks/softmax-regression.ipynb#%E4%BF%A1%E6%81%AF%E9%87%8F)

压缩与预测有什么关系呢？ 想象一下，我们有一个要压缩的数据流。 如果我们很容易预测下一个数据，那么这个数据就很容易压缩。 为什么呢？ 举一个极端的例子，假如数据流中的每个数据完全相同，这会是一个非常无聊的数据流。 由于它们总是相同的，我们总是知道下一个数据是什么。 所以，为了传递数据流的内容，我们不必传输任何信息。也就是说，“下一个数据是xx”这个事件毫无信息量。

但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到"惊异"。 克劳德·香农决定用信息量log1𝑃(𝑗)=−log𝑃(𝑗)log⁡1�(�)=−log⁡�(�)来量化这种惊异程度。 在观察一个事件𝑗�时，并赋予它（主观）概率𝑃(𝑗)�(�)。 当我们赋予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。 在 :eqref:`eq_softmax_reg_entropy`中定义的熵， 是当分配的概率真正匹配数据生成过程时的_信息量的期望_。

### 重新审视交叉熵[](http://localhost:8888/notebooks/Py_Code/d2l-zh/pytorch/chapter_linear-networks/softmax-regression.ipynb#%E9%87%8D%E6%96%B0%E5%AE%A1%E8%A7%86%E4%BA%A4%E5%8F%89%E7%86%B5)

如果把熵𝐻(𝑃)�(�)想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵？ 交叉熵_从_𝑃�_到_𝑄�，记为𝐻(𝑃,𝑄)�(�,�)。 我们可以把交叉熵想象为“主观概率为𝑄�的观察者在看到根据概率𝑃�生成的数据时的预期惊异”。 当𝑃=𝑄�=�时，交叉熵达到最低。 在这种情况下，从𝑃�到𝑄�的交叉熵是𝐻(𝑃,𝑃)=𝐻(𝑃)�(�,�)=�(�)。

简而言之，我们可以从两方面来考虑交叉熵分类目标： （i）最大化观测数据的似然；（ii）最小化传达标签所需的惊异。

### 3.图片分类数据集

#### 3.1 Fashion-MNIST数据集：

- 读取数据集

```python

trans=transforms.ToTensor()

mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True, transform=trans,download=True)

mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False, transform=trans,download=True)

```

- 数据集内图片大小

```python

mnist_train[0][0].shape

torch.Size([1, 28, 28])

```

表示图片为单通道（黑白）的28X28的图片

- 显示数据集图像

```

X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))

show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))

```

<div align="center">

<img src="../imgs/09/09-08.png" alt="image" align="center"width="500"/>

</div>

### 4.从零实现softmax回归

#### softmax:

$$

softmax(X)_{ij}=\frac{exp(X_{ij})}{\sum_{k} exp(X_{ik})}

$$

```python

def softmax(X):

X_exp = torch.exp(X)

partition = X_exp.sum(1, keepdim=True)

return X_exp / partition

```

1. 将图像展平，每个图像看做长度为784的向量，因为数据集有十个类别，所以网络输出维度为10。以此设定参数大小并初始化：

```python

num_inputs = 784

num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)

b = torch.zeros(num_outputs, requires_grad=True)

```

2. 实现softmax回归模型：

```python

def net(X):

return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

```

3. 实现交叉熵损失函数：

```python

def cross_entropy(y_hat, y):

return - torch.log(y_hat[range(len(y_hat)), y])

```

4. 计算正确率：

```python

def accuracy(y_hat, y):

"""计算预测正确的数量"""

if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:

y_hat = y_hat.argmax(axis=1)

cmp = y_hat.type(y.dtype) == y

return float(cmp.type(y.dtype).sum())

```

5. 评估net精度

```python

def evaluate_accuracy(net, data_iter):

"""计算在指定数据集上模型的精度"""

if isinstance(net, torch.nn.Module):

net.eval()

metric = Accumulator(2)

with torch.no_grad():

for X, y in data_iter:

metric.add(accuracy(net(X), y), y.numel())

return metric[0] / metric[1]

```

```python

class Accumulator:

"""在n个变量上累加"""

def __init__(self, n):

self.data = [0.0] * n

def add(self, *args):

self.data = [a + float(b) for a, b in zip(self.data, args)]

def reset(self):

self.data = [0.0] * len(self.data)

def __getitem__(self, idx):

return self.data[idx]

```

6. 定义训练模型：

```python

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):

"""训练模型（定义见第3章）"""

animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],

legend=['train loss', 'train acc', 'test acc'])

for epoch in range(num_epochs):

train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

test_acc = evaluate_accuracy(net, test_iter)

animator.add(epoch + 1, train_metrics + (test_acc,))

train_loss, train_acc = train_metrics

assert train_loss < 0.5, train_loss

assert train_acc <= 1 and train_acc > 0.7, train_acc

assert test_acc <= 1 and test_acc > 0.7, test_acc

```

7. 预测：

```python

def predict_ch3(net, test_iter, n=6):

"""预测标签（定义见第3章）"""

for X, y in test_iter:

break

trues = d2l.get_fashion_mnist_labels(y)

preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))

titles = [true +'\n' + pred for true, pred in zip(trues, preds)]

d2l.show_images(

X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)

```

<div align="center">

<img src="../imgs/09/09-09.png" alt="image" align="center"width="500"/>

</div>

### 5.softmax的简洁实现

> 调用torch内的网络层

```python

import torch

from torch import nn

from d2l import torch as d2l

batch_size=256

train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):

if type(m) == nn.Linear:

nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss=nn.CrossEntropyLoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs=10

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

```

### 6.softmax回归Q&A

**Q1:softlabel训练策略以及为什么有效？**

> softmax用指数很难逼近1，softlabel将正例和负例分别标记为0.9和0.1使结果逼近变得可能，这是一个常用的小技巧。

##### Q2:softmax回归和logistic回归？

> logistic回归为二分类问题，是softmax回归的特例

##### Q3:为什么使用交叉熵，而不用相对熵，互信息熵等其他基于信息量的度量？

> 实际上使用哪一种熵的效果区别不大，所以哪种简单就用哪种

##### Q4:![](http://latex.codecogs.com/gif.latex?\\y*log\hat{y}) 为什么我们只关心正确类，而不关心不正确的类呢？

> 并不是不关心，而是不正确的的类标号为零，所以算式中不体现，如果使用softlabel策略，就会体现出不正确的类。

##### Q5:似然函数曲线是怎么得出来的？有什么参考意义？

> 最小化损失函数也意味着最大化似然函数，似然函数表示统计概率和模型的拟合程度。

##### Q6:在多次迭代之后欧如果测试精度出现上升后再下降是过拟合了吗？可以提前终止吗？

> 很有可能是过拟合，可以继续训练来观察是否持续下降

##### Q7:cnn网络主要学习到的是纹理还是轮廓还是所有内容的综合？

> 目前认为主要学习到的是纹理信息

##### Q8:softmax可解释吗？

> 单纯softmax是可解释的，可以在统计书籍中找到相关的解释。
