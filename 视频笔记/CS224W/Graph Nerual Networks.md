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


### GNNs subsume CNNs and Transformers
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
- Idea: Raw input graph $\neq$ Computational graph 进行图增强，对原始输入进行一定处理
	- Graph feature augmentation  特征扩增
	- Graph structure augmentation  结构扩增

### A Single Layer of a GNN 单层设计
- Idea: compress a set of vectors into a single vector
	- 将一系列向量（上一层的自身和邻居的message）压缩到一个向量中(新的节点嵌入)
- 2-step process
	- Message
	- Aggregation
- GNN在进行表征计算时一般会包括对节点的两种操作
	- Message 消息传递，对节点特征信息的处理
	- Aggregation 信息聚合，对邻域节点信息的聚合
		- ordering invariant，结果与聚合的顺序无关

#### Message Computation
- Message function  $m_u^{(l)} = MSG^{(l)}(h_u^{(l - 1)})$
	- each node will create a message, which will be sent to other nodes later
	- e.g. A Linear layer $m_u^{(l)} = W^{(l)}h_u^{(l - 1)}$  -  node features 左乘 weight matrix

#### Message Aggregation
- Each node will aggregate the messages from node $v$'s neighbors
	- $h_v^{(l)} = AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \})$
		- aggregator - Sum(·)，Mean(·)，Max(·)
- Issue
	- Information from node $v$ itself could get lost 导致节点自身信息丢失
	- Computation of $h_v^{(l)}$ does not directly depend on $h_v^{(l - 1)}$
- Solution  -  includ $h_v^{(l - 1)}$ when computing $h_v^{(l)}$
	- Message  -  compute messgae from node $v$ itself
	- Aggregation  -  after aggregating from neighbors , we can aggregate the message from node $v$ itself
		- via concatenation or summation
			- $h_v^{(l)} = CONCAT(AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \}), m_v^{(l)})$

- Putting together 合并上述两部
	- Message  -  each node computes a message
		- $m_u^{(l)} = MSG^{(l)}(h_u^{(l - 1)}, u \in \{ N(v) \cup v \})$
	- Aggregation  -  aggregate messages from neighbors
		- $h_v^{(l)} = AGG^{(l)}(\{ m_u^{(l)}, u \in N(v) \}, m_v^{(l)})$
	- Nonlinearity 非线性表示能力，这两部都可以用这个来增加其表现力
		- often ReLU(·) Sigmoid(·)
		- can be added to message or aggregation

### Classical GNN layers

#### GCN
- Graph Convolutional Networks 图卷积神经网络
- 与CNN具体计算操作实现方式不同，但背后隐含的计算思想是相同的，都是从周围提取信息然后通过执行某种操作而获得新的信息
- GCN 相当于对上一层的节点嵌入用本层的权重矩阵转换，用节点度归一化实现message，然后加总邻居节点，应用激活函数实现聚合
	- 不同GCN论文中会应用不同的归一化方式
- 《Semi-supervised Classification with Graph Convolutional Networks》(ICLR 2017)
-  计算公式:  $h_v^{(l)} = \sigma(W^{(l)}\sum\limits_{u \in N(v)}\frac{h_u^{(l - 1)}}{|N(v)|})$
- Written as Message + Aggregation
	- 分成两部分
		- Message -  $W^{(l)}\frac{h_u^{(l - 1)}}{|N(v)|}$
		- Aggregation  -  $\sum\limits_{u \in N(v)}$
	- Message
		- each neighbor:  $m_u^{(l)} = \frac{1}{|N(v)|}W^{(l)}h_u^{(l-1)}$
			- normalized by node degree
		- normalized adjacency matrix $D^{-1/2}AD^{-1/2}$
	- Aggregation
		- sum over messages from neighbors, then apply activation，累加操作，然后应用非线性激活函数
		- $h_v^{(l)} = \sigma(Sum(\{ m_u^{(l)}, u \in N(v) \}))$
			- In GCN graph is assumed to have self-edges

#### GraphSAGE
- Graph SAmple and aggreGatE，中心思想是小批量采用原有大图中的子图
	- sample  -  对邻域节点采样
	- aggregate  -  邻域节点的特征聚集
- GraphSAGE 可选用多种聚合方式来聚合邻居信息，然后再连接节点本身信息，最后使用L2正则化
- 《Inductive Representation Learning on Large Graphs》
- $h_v^{(l)} = \sigma(W^{(l)} \times CONCAT(h_v^{(l-1)}, AGG(\{ h_u^{(l-1)}, \forall u \in N(v) \})))$
- Written as Message + Aggregation
	- Message is computed within the AGG(·)
	- 2-step aggregation
		- stage 1  -  aggregate from node neighbors 聚合邻域信息
			- $h_v^{(l)} \leftarrow AGG(\{ h_u^{(l-1)}, \forall u \in N(v) \})$
		- stage 2  -  further aggregate over the node itself
			- $h_v^{(l)} = \sigma(W^{(l)} \times CONCAT(h_v^{(l-1)}, h_{N(v)}^{(l)}))$
- GraphSAGE neighbor aggregation
	- Mean  -  take a weighted avg. of neighbors，可以不做拼接操作，直接将上一层自身信息与邻居信息进行均值处理
		- AGG = $\sum\limits_{u \in N(v)}\frac{h_u^{(l - 1)}}{|N(v)|}$
			- aggregation  -  $\sum\limits_{u \in N(v)}$
			- message computation  -  $|N(v)|$
	- Pool  -  transform neighbor vector and apply symmetric vector function Mean(·) or Max(·)，将邻居构成的向量放入全连接网络然后接上一个最大池化或平均池化
		- AGG = Mean ({ MLP($h_u^{(l-1)}$), $\forall u \in N(v)$})
			- aggregation  -  Mean
			- message computation  -  MLP
	- LSTM  -  apply LSTM to reshuffled of neighbors  为了消除序列性，长短时记忆神经网络，打乱节点顺序输入。具有大容量的优点
		- AGG = LSTM (\[ $h_u^{(l-1)} ,\forall u \in \pi(N(v))$ \])
			- aggregation  -  LSTM
- L2 Normalization 每一层都归一化
	- Apply $\mathcal{l}_2$ mnormalization to $h_v^{(l)}$ at every layer
	- $h_v^{(l)} \leftarrow \frac{h_v^{(l)}}{||{h_v^{(l)}||}_2} \enspace \forall v\in V \enspace where \enspace {||u||}_2 = \sqrt{\sum_iu_i^2} \enspace (l_2 - norm)$
- Without L2 normalization, the embedding vectors have different scales for vectors
- After this, all vectors will have the same L2-norm

#### GAT
- Graph Attention Networks
- 《Graph Attention Networks》(ICLR 2018)
- 将注意力机制引入到了GNN模型表征中，使用注意力机制为邻居对节点信息影响程度加权，用softmax过后的邻居信息加权求和来实现节点嵌入embedding的计算
- $h_v^{(l)} = \sigma(\sum_{u \in N(v)} \alpha_{vu} W^{(l)} h_u^{(l-1)})$
	- $\alpha_{vu}$  -  attention weights 注意力权重
- In GCN / GraphSAGE
	- $\alpha_{vu} = \frac{1}{|N(v)|}$  -  the weighting factor of node $u$'s message to node $v$
	- 不同邻居带来的信息权重相同(连接的权重相同)
- In GAT, *not* all node's neighbots are eqaully important
	- Attention is inspired by 认知科学中的“注意力”
	- $\alpha_{vu}$ focuses on the *important* parts of the input data and fades out the rest
		- the NN should devote more computing power on small but important part of data
		- the importance depends on the context and is learned through training
- 通过学习得到注意力权重
	- node-wise 逐顶点的运算
	- Goal: specify arbitrary importance to different neighbors of each node in the graph
	- 给予邻居不同的重要性

##### Attention Mechanism
- Attention Mechanism 分为两步走
	1. 计算注意力系数
		- Let $a_{vu}$ be computed as a byproduct of an attention mechanism $a$ :
		- $e_{vu} = a(W^{(l)}h_u^{(l-1)}, W^{(l)}h_v^{(l-1)})$
			- $e_{uv}$  -  attention coefficients 注意力分数, indicates the importance of $u$'s message to node $v$
			- $a$  -  自注意力函数
		- 归一化 Normalize $e_{vu}$
			- $a_{vu} = \frac{exp(e_{vu})}{\sum_{k \in N(v)}exp(e_{vk})}$
				- use softmax , s.t. $\sum_{u \in N(v)} a_{vu} = 1$
	2. 加权求和 Weighted sum based on above 
		- $h_v^{(l)} = \sigma(\sum_{u \in N(v)} \alpha_{vu} W^{(l)} h_u^{(l-1)})$
- The form of attention mechanism $a$
- 增强鲁棒性 Multi-head attention:  stabilizes the learning process of attention mechanism
	- 多头注意力机制，分别训练不同的a函数，每个a函数对应一套 $\alpha$ 权重
		- 避免偏见，陷入局部最优
	- Create multiple attention scores
		- $h_v^{(l)}[1] = \sigma(\sum_{u \in N(v)} \alpha_{vu}^1 W^{(l)} h_u^{(l-1)})$
		- $h_v^{(l)}[2] = \sigma(\sum_{u \in N(v)} \alpha_{vu}^2 W^{(l)} h_u^{(l-1)})$
		- $h_v^{(l)}[3] = \sigma(\sum_{u \in N(v)} \alpha_{vu}^3 W^{(l)} h_u^{(l-1)})$
	- Aggregate outputs
		- by concatenation or summation
		- $h_v^{(l)} = AGG(h_v^{(l)}[1], h_v^{(l)}[2], h_v^{(l)}[3])$
- Benefits
	- Allows for (implicitly) specifying different importance values $a_{vu}$ to different neighbors 隐式指定节点信息对邻居的importance
	- computationally efficient, can be parallelized
	- storage efficient 稀疏矩阵运算需要存储的元素数不超过 $O ( V + E )$ ，参数数目固定（a 的可训练参数尺寸与图尺寸无关）
	- only attends over local network neighborhoods 只注意局部邻居，赋予权重
	- inductive capability 归纳泛化
		- 共享同一个 $a$ 函数
		- 并泛化到新图

### GNN Layers in Practice
- 通用GNN层模板
- Many modern deep learning modules can be incorporated into a GNN layer
	- 实践应用中的GNN层，往往会应用传统深度神经网络模块
	- Batch Normalization
		- stabilize neural network training
	- Dropout
		- prevent overfitting
	- Attention/Gating
		- ctrl the importance of a message
	- More
		- any other useful deep learning modules

#### Batch Normalization
- Goal:  stabilize neural networks training
- Idea: Given a batch of inputs(node embeddings) 对节点嵌入进行归一化
	- Re-center the node embeddings into zero mean 平均值=0
	- Re-scale the variance into unit variance 方差=1
- Input:  $X \in \mathbb{R}^{N \times D}$  
	- $N$ node embeddings
	- N 个样本，每个样本 D-维向量
- Trainable Parameters:  $\gamma,\beta \in \mathbb{R}^D$
	- 每个维度有 2 个恢复参数 
- Output:  $Y \in \mathbb{R}^{N \times D}$
	- Normalized node embeddings
- Step 1 : compute the mean and variance over N embeddings
	- $\mu_j = \frac{1}{N} \sum\limits_{i=1}^N X_{i,j}$
	- $\sigma_j^2 = \frac{1}{N} \sum\limits_{i=1}^N {(X_{i,j} - \mu_j)}^2$
- Step 2 : normalized the feature using computed mean and variance
	- $\hat{X}_{i,j} = \frac{X_{i,j} - \mu_j}{\sqrt{{\sigma_j + \epsilon}^2}}$
	- $Y_{i, j} = \gamma_j \hat{X}_{i,j} + \beta_j$

#### Dropout


### Stacking GNN Layers

***

## 本章总结
- 本章回顾了node embedding，以及其编码器和解码器
- 介绍了深度学习的基础，监督学习，损失函数，优化方法，梯度向量，随机梯度下降，非线性激活函数，以及多层感知机
- 介绍图深度学习的思路，计算图和邻域、自嵌入向量的aggregation
- 介绍了图卷积神经网络，以及基本方法 mean aggregation，以及矩阵形式。
- 介绍了图神经网络是一个通用体系结构，CNN和Transformer都可以被看作是一个特殊的GNN
- 介绍了GNN的通用体系架构，GNN层包含Message和Aggregation，原始输入图要经过特征扩增、结构扩增才能成为计算图。介绍了经典的GNN层
- 介绍了GCN 图卷积神经网络的计算公式，将其写成 Message + Aggregation的形式，使用节点的度来归一化，聚合所有邻域信息，最后使用激活函数.
- 介绍了GraphSAGE 图采样和聚集，它的思想和计算方法，以及归纳式表示学习
- 介绍了GraphSAGE的消息传递和两步信息聚合，三种不同的聚合方法，最后应用L2 Norm 对每一层进行归一化。GraphSAGE相比于GCN的优势，它保留了节点本身信息特征。
- 介绍了GAT 图注意力网络，注意力机制，将注意力这一概念引入图中，使用注意力机制为邻居对节点信息影响程度加权，用softmax过后的邻居信息加权求和来实现节点嵌入embedding的计算。
- 介绍了注意力机制两步走的计算方法，为了增强鲁棒性，更是采用了多头注意力机制，分别训练不同的 a 函数，每个 a 函数对应一套 $\alpha$ 权重，来避免偏见。
- 介绍了GAT的各种优点，包括隐式指定重要性、并行计算、存储高效、归纳泛化等。
