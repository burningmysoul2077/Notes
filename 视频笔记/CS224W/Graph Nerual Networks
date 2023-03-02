## Recap
### Node Embeddings
- 图嵌入表示学习，Map nodes to $d$-dimensional embeddings such that similar nodes in the graph are embedded close together
- Similarity $(u, v) \approx {z_v}^Tz_u$
- Encoder: maps each node to a low-dimensional vector. Decoder: dot product between node embeddings
- Shallow encoding: encoder is an *embedding-lookup*
	- DeepWalk, Node2Vec, LINE
	- limitations
		- 每个节点的嵌入向量都需单独训练，节点之间没有参数共享
		- 直推式，无法泛化到新图/新节点
		- 没有利用到节点属性特征

#### Deep Graph Encoders
- 图深度学习 图神经网络
- $ENC(v) =$ multiple layers of non-linear transformations based on graph structure
***

## Basics of Deep Learning

### Machine Learning as Optimization
- Supervised learning: given input $x$, and the goal is to predict label $y$
	- input $x$ can be
		- 向量
		- 文本序列
		- 栅格图片
		- 图
- **The task is formulated as an optimization problem**
	- Objective function: $\mathop{min}\limits_\theta \mathcal{L}(y, f(x))$  
	- $\theta$ : a set of parameters we optimize
		- could contain 1+ scalars, vectors, matrics
	- $\cal L$, loss function
		- 回归问题 L2 loss
		- 分类问题 Cross Entropy
		- common loss functions: L1 loss, huber loss, max margin

#### Loss Function
 - Loss function example:  Cross Entropy (CE) 多分类，交叉熵损失函数
	- $CE(y, f(x)) = -\sum_{i=1}^C(y_ilog{f(x)}_i)$
	- $\mathcal{L} = \sum_{(x, y)\in \mathcal{T}} CE(y, f(x))$
		- training set

#### Gradient Vector
- Gradient vector, the way to optimize the objective function
	- Direction and rate of fastest increase  $\nabla_\theta \mathcal{L} = (\frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2} , \cdots )$，对每一个参数求偏导数，就知道如何优化每一个参数使得损失函数最小化
	- Gradient is the directional derivative in the direction of largest increase 斜率最大的方向
- Gradient Descent
	- 梯度下降 Iterative algorithmL repearedly update weights in the (opposite) direction of gradients until convergence
		- $\theta \leftarrow \theta - \eta\nabla_\theta \cal {L}$
	- 学习率 LR $\eta$, controls the size of gradient step
	- Ideal termination condition: gradient = 0
- Stochastic Gradient Descent SGD
	- exact gradient requires comupting $\nabla_\theta \mathcal{L} = (y, f(x))$, where x is the entire dataset
	- 每迭代一次，需输入所有样本计算损失函数，开销太大，所以采用随机梯度下降
	- At every step, pick a different minibatch $\cal {B}$ containing a subset of the dataset, use it as input $x$
		- 每迭代一次，只输入 batch size 个样本计算损失函数
- Minibatch SGD
	- Concepts
		- Batch size : 一次迭代输入的样本数
		- Iteration : 1 step of SGD on a minibatch
		- Epoch，一轮 : one full pass over the dataset
	- Loop:
		1. Sample a batch of data 采样生成mini-batch
		2. Forward prop it through the graph(network), get loss 前向推断，求损失函数
		3. Backprop to calculate the gradients 反向传播，求每个权重的更新梯度
		4. Update the parameters using the gradient 优化更新权重
	- SGD is unbiased estimator of full gradient
		- But no guarantee on the rate of convergence
		- In practice, often requires tuning of learning rate
	- Common optimizer: Adam, Adagrad, Adadelta, RMSprop

#### Comparison
- Stochastic，每次喂一个数据进去，求一个数据上的梯度，很振荡
- Mini-batch，每次喂一小批数据进去，求mini-batch梯度，较振荡
- Batch，每次喂全部数据进去，求全局梯度，不振荡

#### Nerual Network Function
- Objective: $\mathop{min}\limits_\theta \mathcal{L}(y, f(x))$
- Back Propagation
	- To minimize $\cal {L}$ , evaluate the gradient:  $\nabla_W \mathcal{L} = (\frac{\partial \mathcal{L}}{\partial W_1}, \frac{\partial \mathcal{L}}{\partial W_2} , \frac{\partial \mathcal{L}}{\partial W_3} ,\cdots )$
	- 复合函数求偏导
	- Chain rule 链式法则
		- 例如 $f(x) = W_2(W_1x), h(x) = W_1x \enspace g(z)=W_2z, f(x) = g(h(x))$
			-根据链式法则 $\frac{\partial f}{\partial x}  = \frac{\partial g}{\partial h}  \cdot \frac{\partial h}{\partial x}$ or ${f}'(x)  = {g}'(h(x)) {h}'(x)$
		- s.t. $\nabla_xf = \frac{\partial y}{\partial W_1x}  \cdot \frac{\partial W_1x}{\partial x}$
	- Forward propagation 前向预测，求损失函数
	- Back-propagation to compute gradient

#### Non-linearity
- 矩阵代表线性变换
	- 线性变换，对空间的挤压伸展。保持网格线平行且等距分布，且保持原点不变
- 非线性激活函数
	- Rectified linear unit ReLU
		- $ReLU(x) = max(x, 0)$
	- Sigmoid
		- $\sigma(x) = \frac{1}{1 + e^{-x}}$

#### Multi-layer Perceptron MLP
- Each layer pf MLP combines linear transformation and non-linearity
	- $x^{(l + 1)} = \sigma(W_lx^{(l)}) + b^l$
		- $W_l$ - weight matrix that transforms hidden representation at layer $l$ to layer $l +1$
		- $b^l$ - bias at layer $l$, and is added to the linear transformation of $x$ 偏置项
		- $\sigma$ is non-linearity function 非线性激活函数
***

## Deep Learning for Graphs

#### Setup
- Assume a graph $G$
	- $V$  -  vertex set
	- $A$  -  adjacency matrix
	- $X \in \mathbb{R}^{m \times |V|}$  -  a matrix of node features
	- $v$  -  a node in $V$
	- $N(v)$  -  the set of neighbors of $v$
	- Node features

#### A Naive Approach
- 朴素想法: 直接输入邻接矩阵 A
- Issues
	- 过拟合
	- 无法泛化到新节点
	- 不具备“变换不变性”

#### Real-World Graphs
- There is no fixed notion of locality or sliding window on the graph
- Graph is permutation invariant

#### Permutation Invariance
- 排列不变性，特征之间没有空间位置关系
- Graph does not have a canonical order of the nodes
	- 与节点编号顺序无关，与图的显示方式无关
- Consider we learn a function $f$ that maps a graph $G = (A, X)$ to a vector $\mathbb{R}^d$
- Then, if $f(A_i, X_i) = f(A_j, X_j)$ for any order plan $i$ and $j$
- $f$ is a **permutation invariant function**

#### Permutation Equivariance
- 置换同变性，排列相等性
- Consider we learn a function $f$ that maps a graph $G = (A, X)$ to a vector $\mathbb{R}^{m \times d}$
	- graph has $m$ nodes, each row is the embedding of a node
- Then, if this property holds for any order plan $i$ and $j$
- $f$ is a **permutation equivariant function**

### Graph Neural Network Overview
- GNN consist of multiple permutation equivariant / invariant functions
	- Other NN architectures e.g. MLPs, are not permutation invariant / equivariant
- Desgin GNN that are permutation invariant / equivariant by *passing and aggregating information from neighbors* 
	- 消息传递和聚合
***

## Graph Convolutional Networks
- Idea：
	- determine node computation graph
	- propagate and transform information

### Aggregate Neighbors
- Key idea  -  Generate node embeddings based on local network neighborhoods
- Nodes aggregate information from their neighbors using neural networks
- Network neighborhood defines a computation graph
	- 每个节点分别构建自己的计算图
- 这个 Model 理论上任意深度
	- nodes have embeddings at each layer
	- layer-0 embedding of node $v$  = its input feature $x_v$
	- layer-k embedding gets information from nodes that are $k$ hops away
	- 图神经网络的层数，是计算图的层数，而不是神经网络的层数
- Neighborhood Aggregation
	- Basic approach  -  avg. information from neighbors and apply a neural network
		- order invariant
		- permutation invariant
	- $h_v^{k + 1} = \sigma(W_k \sum\limits_{u \in N(v)}\frac{h_u^{(k)}}{|N(v)|} + B_kH_v^{(k)}), \forall k \in \{0, \cdots, K -1\}$
		- $h_v^{k + 1}$  -  节点属性特征 = 本层节点 v 的嵌入向量
		- $z_v = h_v^{(K)}$  -  embedding after L layers of neighborhood aggregation，最后一层节点 v 的嵌入向量作为 “表示”
		- $\sigma$  -  非线性激活函数
		- $\sum\limits_{u \in N(v)}\frac{h_u^{(k)}}{|N(v)|}$  -  邻域节点上一层平均嵌入向量
		- $h_v^{(k)}$  -  embedding of v at layer k
		- $K$  -  总层数
		- 多个嵌入向量求和（平均）结果，与节点顺序无关
- Message passing and neighbor aggregation in GCN is permutation equivariant

### Training the Model
- Need to define a loss function on the embeddings
- Model Parameters
	- $h_v^{(0)} = x_v$
	- $h_v^{k + 1} = \sigma(W_k \sum\limits_{u \in N(v)}\frac{h_u^{(k)}}{|N(v)|} + B_kH_v^{(k)}), \forall k \in \{0, \cdots, K -1\}$
		- $W_k \enspace B_k$, 第 k 层图神经网络，是需训练学习得到的权重参数
		- $h_v^k$  -  the hidden representation of node v at layer k
		- $W_k$  -  weight matrix for neighborhood aggregation
		- $B_k$  -  weight matrix for transforming hidden vector of self
	- $z_v$ = final node embedding $h_v^{(k)}$
	- We can feed these embeddings into any loss function and run SGD to train the weight parameters
- Matrix formulation, many aggregations can be performed efficiently by (sparse) matrix operations
	- Let $H^{(k)} = {[h_1^{(k)} \cdots h_{|v|}^{(k)}]}^T$
	- $\sum_{u \in N_v} h_u^{(k)} = A_{v,:}H^{(k)}$
	- Let $D$ be diagonal matrix where $D_{v, v} = Deg(v) = |N(v)|$
		- $D^{-1}$  -  the inverse of $D$  is also diagonal,  $D_{v, v}^{-1} = 1/|N(v)|$
	- Therefore, $\sum\limits_{u \in N(v)}\frac{h_u^{(k)}}{|N(v)|} \rightarrow H^{(k+1)} = D^{(-1)}AH^{(k)}$
	- Re-writing update function in matrix form:  $H^{(k+1)} = \sigma(\tilde{A}H^{(k)}W_k^T + H^{(k)}B_k^T)$,  where $\tilde{A} = D^{-1}A$
		- $\tilde{A}H^{(k)}W_k^T$  -  neighborhood aggregation
		- $H^{(k)}B_k^T$  -  self transformation
		- $\tilde{A}$ is sparse

### How to train a GNN
- Supervised setting
	- $\mathop{min}\limits_\theta \mathcal{L}(y, f(z_v))$
		- $y$  -  节点类别标注
		- $\cal{L}$  -  回归问题 L2,  分为问题 cross entropy
	- Use CE loss
		- $\mathcal{L} = \sum\limits_{v \in V}y_v\log(\sigma(z_v^T\theta)) + (1 - y_v)\log(1 - \sigma(z_v^T\theta))$
			- $\theta$  -  classification weights  分类预测头权重参数
- Unsupervised setting
	- use the graph structure as the supervision 自监督
	- Similar nodes have similar embeddings
		- $\mathcal{L} = \sum\limits_{z_u, z_v} CE(y_{u, v}, DEC(z_u, z_v))$
			- $y_{u, v} = 1$, when node u and v are similar
			- CE -  cross entropy
			- DEC  -  decoder, e.g. inner product
	- Node similarity
		- Random walks
			- node2vec
			- DeepWalk
			- struc2vec
		- Matrix facorization
		- Node proximity in the graph

### Model Design
1. Define a neighborhood aggregation function 聚合邻域信息的方式
2. Define a loss function on the embeddings
3. Train on a set of nodes, i.e. a batch of compute graphs (mini-batch)
4. Generate embeddings for nodes as needed, 泛化到新节点、新图，Inductive Learning 归纳式学习
***

## GNNs subsume CNNs and Transformers
- CNN can be seen as a special GNN with fixed neighbot size and ordering 固定的邻域和固定的顺序
	- the size of filter 卷积核 is pre-defined for a CNN
	- the advg. of GNN is it processes arbitrary graphs with different degrees for each node
	- CNN is not permutation equivariant
		- switching the order of pixels will leads to different outputs
- Tranformer
	- key component  -  self-attention
	- transformer layer can be seen as a special GNN that runs on a fully-connected "word" graph 
		- GAT Graph Attention Network
***


## A General Perspective on Graph Neural Networks

### A General GNN Framework
![[Pasted image 20230302171211.png]]

- GNN Layer = Message + Aggregation
	- GCN GraphSAGE GAT
- Idea: Raw input graph $\neq$ Computational graph
	- Graph feature augmentation  特征扩增
	- Graph structure augmentation  结构扩增

### A Single Layer of a GNN
- Idea: compress a set of vectors into a single vector
- 2-step process
	- Message
	- Aggregation

#### Message Computation
- Message function  $m_u^{(l)} = MSG^{(l)}(h_u^{(l - 1)})$
	- each node will create a message, which will be sent to other nodes later

#### Message Aggregation
- Each node will aggregate the messages from node $v$'s neighbors
	- $h_v^{(l)} = AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \})$
		- aggregator - Sum(·)，Mean(·)，Max(·)
- Issue
	- Information from node $v$ itself could get lost
	- Computation of $h_v^{(l)}$ does not directly depend on $h_v^{(l - 1)}$
- Solution  -  includ $h_v^{(l - 1)}$ when computing $h_v^{(l)}$
	- Message  -  compute messgae from node $v$ itself
	- Aggregation  -  after aggregating from neighbors , we can aggregate the message from node $v$ itself
		- via concatenation or summation
			- $h_v^{(l)} = CONCAT(AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \}), m_v^{(l)})$

- Putting together
	- Message  -  each node computes a message
		- $m_u^{(l)} = MSG^{(l)}(h_u^{(l - 1)}, u \in \{ N(v) \cup v \})$
	- Aggregation  -  aggregate messages from neighbors
		- $h_v^{(l)} = AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \}, m_v^{(l)})$
	- Nonlinearity 非线性表示能力
		- often ReLU(·) Sigmoid(·)
		- can be added to message or aggregation

### Classical GNN layers

#### GCN
- $h_v^{(l)} = \sigma(\sum\limits_{u \in N(v)}W^{(l)}\frac{h_u^{(l - 1)}}{|N(v)|})$
- Written as Message + Aggregation
	- Message
		- each neighbor:  $m_u^{(l)} = \frac{1}{|N(v)|}W^{(l)}h_u^{(l-1)}$
			- normalized by node degree
			- normalized adjacency matrix $D^{-1/2}AD^{-1/2}$
	- Aggregation
		- sum over messages from neighbors, then apply activation
		- $h_v^{(l)} = \sigma(Sum(\{ m_u^{(l)}, u \in N(v) \}))$
			- In GCN graph is assumed to have self-edges

#### GraphSAGE
- $h_v^{(l)} = \sigma(W^{(l)} \times CONCAT(h_v^{(l-1)}, AGG(\{ h_u^{(l-1)}, \forall u \in N(v) \})))$
- Written as Message + Aggregation
	- Message is computed within the AGG(·)
	- 2-step aggregation
		- stage 1  -  aggregate from node neighbors 聚合邻域信息
			- $h_v^{(l)} \leftarrow AGG(\{ h_u^{(l-1)}, \forall u \in N(v) \})$
		- stage 2  -  further aggregate over the node itself
			- $h_v^{(l)} = \sigma(W^{(l)} \times CONCAT(h_v^{(l-1)}, h_{N(v)}^{(l)}))$
- GraphSAGE neighbor aggregation
	- Mean  -  take a weighted avg. of neighbors

***

## 本章总结
- 本章回顾了node embedding，以及其编码器和解码器
- 介绍了深度学习的基础，监督学习，损失函数，优化方法，梯度向量，随机梯度下降，非线性激活函数，以及多层感知机
- 介绍图深度学习的思路，计算图和邻域、自嵌入向量的aggregation
- 介绍了图卷积神经网络，以及基本方法 mean aggregation，以及矩阵形式
- 介绍了图神经网络是一个通用体系结构，CNN和Transformer都可以被看作是一个特殊的GNN
