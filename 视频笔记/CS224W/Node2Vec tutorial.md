## 图嵌入经典算法 Node2Vec 论文精读
	 《node2vec: Scalable Feature Learning for Networks》
	   - Aditya Grover, Jure Leskovec, Stanford University
	   - KDD 2016

### DeepWalk的缺点
- 用完全随机游走，训练节点嵌入向量
- 仅能反映相邻节点的社群相似信息
- 无法反映节点的功能角色相似信息

## Node2Vec
- 2阶有偏的随机游走 Biased $2^{nd}$-order Walks
	- BFS, Micro-view of neighborhood, homophily 同质社群(社交网络)
	- DFS, Macro-view of neighborhood, structural equivalence 节点功能角色(中枢、桥接、边缘)

- Node2Vec解决 图嵌入 问题，将图中的每个节点映射为一个向量(嵌入)。
- 向量(嵌入)包含了节点的语义信息(相邻社群和功能角色)
-   语义相似的节点，向量(嵌入)的距离也近、
-   向量(嵌入)用于后续的分类、聚类、Link Prediction、推荐等任务
- 
-   在DeepWalk完全随机游走的基础上，Node2Vec增加 $p, q$ 参数，实现有偏随机游走。不同的 $p, q$ 组合，对应了不同的探索范围和节点语义
-   DFS深度优先探索，相邻的节点，向量(嵌入)距离相近
-   BFS广度优先探索，相同功能角色的节点，向量(嵌入)距离相近
-   DeepWalk是Node2Vec在 $p = 1, q = 1$ 的特例

### 一些技术细节
- Alias Sampling， $O(1)$
- Other Random Walk Ideas

### Node2Vec 总结与讨论
- 通过调节 $p, q$ 值，实现有偏随机游走，探索节点社群、功能等不同属性
- 首次把node embedding用于 Link Prediction
- 可解释性、可扩展性好，性能卓越
- Stanford CS22W公开课主讲人亲自带货

- 需要大量随机游走序列训练。弱水三千取一瓢，管中窥豹
- 距离较远的两个节点无法直接相互影响。看不到全图信息。(图神经网络)
- 无监督，仅编码图的连接信息，没有利用节点的属性特征。(图卷积)
- 没有真正用到神经网络和深度学习。

## Node2Vec 论文精读
- 《Node2Vec：可扩展的图嵌入表示学习算法》

### ABSTRACT
- Here we propose node2vec, an algorithmic framework for learning continuous feature representations for ndoes in networks.
- In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preseving network neighborhoods of nodes
	- 连续(都是实数)、低维(百维及以下)、稠密(非稀疏)的向量。
- We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods.
	- 一个灵活的、有偏随机游走的范式来有效地探索多样化的网络邻域属性和特征。

#### Categories and Subject Descriptors
- Database Management: Database Applications - Data Mining.
- Artificial Intelligence: Learning.

#### General Terms
- Algorithms; Experimentation.

#### Keywords
- Information networks, Feature learning, Node embeddings, Graph representations.

### INTRODUCTION
- Many important tasks in network analysis involve predictions over nodes and edges.
- In a typical node classification task, we are interested in predictiong the most probable labels of nodes in a network.
- Any supervied machine learning algorithm requires a set of informative, discriminating, and independent features.
	- 任何监督学习算法都需要一套内含丰富语义、有分类区分性以及相互独立的特征。
- In prediction problems on networks this means that one has to construct a feature vector representation for the nodes and edges...
- 传统是基于专家知识来人工设计某领域/业务特有的特征。这不仅仅知识需要大量繁琐的特征工程的工作，而且这些特征专用于特定任务，泛化效果欠佳。
- An alternative approach is to *learn* feature representations by solving an optimization problem
	- 表示学习
- The challenge in feature learning is defining an objective function, which involves a trade-off in balancing computatuinal efficiency and predictive accurary.
	- 先定义一个目标函数，往往是速度和精度的权衡
- On one side of the spectrum, one could directly aim to find a feature representation that optimizes performance of a downstram prediction task.
	- 有监督，针对特定任务优化
- At the other extreme, the objective function can be defined to be independent of the downstream prediction task and the representations can be learned in a purely unsupervised way...
	- 无监督，通用特征，与下游任务无关
- However, current techniques fail to satisfactorily define and optimize a reasonable objective required for scalable unsupervised feature learning in networks
	- 提出了基于矩阵分解的图嵌入
- Classic approaches based on linear and non-linear dimensionality reduction techniques such as Principal Component Analysis, Multi-Dimensional Scaling and their extensions optimize an objective that transforms a representative data matrix of the network such that it maximizes the variance of the data representation. Consequently, these approaches invariably involve eigendecomposition of the appropriate data matrix which is expensive for large real-world networks...
	- 主成分分析，特征值分解，实现降维。但是性能不好，对于大容量的图效果不好。
- Alternatively, we can design an objective that seeks to preserve local neighborhoods of nodes...
	- 基于随机游走的图嵌入，随机梯度下降类似于反向传播
- Specifically, ndoes in networks could be organized based on communities they belong to(*i.e. homophily*); in other cases, the organization could be based on the structural roles of nodes in the network(*i.e. structural equivalence*)
- Real-world networks commonly exhibit a mixture of such equivalences...
- **Present work.**  We propose *node2vec* , a semi-supervised algorithm for scalable feature learning in networks
	- 半监督，自监督
- We optimize a custom graph-based objective function using SGD motivated by prior work on natural language processing.
	- 还是用到word2vec中的SkipGram
- Intuitively, our approach returns feature representations that maximize the likelihood of preserving network neighborhoods of nodes in a $d$-dimensional feature space.
	- 直观上，我们的算法最大化似然概率
- We use a $2^{nd}$ order random walk approach to generate (sample) network neighborhoods for nodes.
	- 使用二阶，有偏随机游走(二阶马尔科夫性)
		- 下一节点不仅与当前节点有关，还与上一节点有关
	- 一阶随机游走(一阶马尔科夫性)
		- 下一节点仅与当前节点有关(DeepWalk，PageRank)
- ...node2vec can learn representations that organize nodes based on their network **roles and/or communities** they belong to.
- The resulting algorithm is flexible, giving us control over the search space through tunable parameters, in contrast to rigid search procedures in prior work.
	- 可调参数来控制
- These parameters can also be learned directly using a tiny fraction of labeled data in a semi-supervised fashion
	- 最优的 $p, q$ 可通过调参得到
- We also show how feature representations of individual nodes can be extended to pairs of nodes(i.e. edges). In order to generate feature representations of edges, we compose the learned feature representations of the individual nodes using simple binary operator
	- 通过二元操作，节点embedding 到 Link(edge) embedding 来做 Link Prediction
- 对于稀疏标注，增删链接扰动，可扩展性表现很好。

### RELATED WORK
- Unsupervised feature learning approaches typically exploit the spectral properties of various matrix representations of graphs, especially the Laplacian and the adjacency matrices.
	- 基于矩阵分解的图嵌入，利用图的谱属性(邻域)，特别是拉普拉斯和邻接矩阵， $L = D - A$ 
- Under this linear algebra perspective, these methods can be viewed as dimensionality reduction techniques. Several linear(e.g. PCA) and non-linear (e.g. IsoMap) dimensionality reduction techniques have been proposed
- These methods suffer from both computational and statistical performance drawbacks...
	- 这些矩阵都需要计算特征值，然而这很难算。而且这些方法也很难扩展。
- Recent advancements in representational learning for natural language processing opened new ways for feature learning of discrete objects such as words
	- word2vec,基于随机游走的。

### FEATURE LEARNING FRAMEWORK
- Let $G = (V, E)$ be a given network. Our analysis is general and applies to any (un)directed, (un)weighted network.
- Let $f: V \rightarrow \mathbb{R}^d$ be the mapping function from nodes to feature representations... Here $d$ is aparameter specifying the number of dimensions of our feature representaion.
	- 都被映射成d-维实数，d是维数
- Equivalently, $f$ is a matrix of size $|V| \times d$ parameters
	- $f$ 是 V行 d维的表格
- For every source node $u \in V$, we define $N_S(u) \subset V$ as a network neighborhood of node $u$ generated through a neighborhood sampling strategy $S$
	- 每个节点，表示出它的邻居节点，S 是采样策略。 u 是当前节点
- We proceed by extending the Skip-gram architecture to networks. We seek to optimize the following objective function, which maximizes the log-probability of observing a network neighborhood $N_S(u)$ for a node $u$ conditioned on its feature representation, given by $f$:  $\mathop{max}\limits_f \sum\limits_{u \in V} \log Pr(N_S(u) | f(u))$
	- Skip-gram损失函数，遍历图中每个节点，输入节点u的embedding，输出它的邻居节点
- In order to make the optimization problem tractable, we make two standard assumptions:
	- Conditional independence. $Pr(N_S(u) | f(u)) = \prod\limits_{n_i \in N_S(u)} Pr(n_i|f(u))$
		- 条件独立等同于马尔科夫假设，周围节点互不影响，相当于事件发生的概率就是独立子事件概率相乘。子事件概率用softmax来计算。
		- $Pr(n_i|f(u)) = \frac{\exp(f(n_i) \cdot f(u))}{\sum_{v \in V} \exp(f(v)\cdot f(u))}$
			- 所有图中节点和v节点的embedding做点乘作分母
	- Symmetry in feature space.
		- 对称性，两个节点之间相互影响的程度一样
- With the above assumptions, the objective simplifies to: $\mathop{max}\limits_f \sum\limits_{u \in V} [ -\log Z_u \sum\limits_{n_i \in N_S(u)}f(n_i) \cdot f(u))]$
- The per-node partition function, $Z_u = \sum_{v \in V} \exp(f(v)\cdot f(u))$，is expensive to compute for large networks and we approximate it using negative sampling
	- 配分函数，归一化分母，很难算。改进成分层softmax、负采样
- The neighborhoods $N_S(u)$ are not restricted to just immediate neighbors but can have vastly different structures depending on the sampling strategy $S$

#### Classis search strategies
- Breadth-first Sampling BFS: The neighborhood $N_s$ is restricted to ndoes which are immediate neighbors of the source
- Depth-first Sampling DFS: The neighborhood consists of nodes sequentially sampled at increasing distances from the source node
- The breadth-first and depth-first sampling represent extreme scnarios in terms...
	- 两种极端策略

#### node2vec
- Building on the above observations, we design a flexible neighborhood sampling strategy which allows us to smoothly interpolate between BFS and DFS

##### Random Walks
- Formally, given a source node $u$, we simulate a random walk of fixed length $l$. Let $c_i$ denote the $i$th node in the walk, starting with $c_0 = u$. Nodes $c_i$ are generated by the following distribution:  $P(c_i = x | c_{i - 1} = v) = \left\{\begin{matrix}  \frac{\pi_{vx}}{Z} \enspace if(v, x) \in E \\   0 \enspace otherwise \\ \end{matrix}\right.$
	- where $\pi_{vx}$ is the unnormalized transition probability between nodes  $v$  and $x$, and $Z$ is the normalizing constant
	- $u$ 是起始节点，$t$ 是上一节点， $N_S(t)$ 上一节点的邻居节点；$v$ 当前节点， $k$ 当前节点的邻居节点个数； $x$ 下一节点，$l$ 随机游走序列节点个数

##### Search bias $\alpha$
- 用连接权重作为游走概率，但无法调节搜索策略
- BFS/DFS 太极端，无法这种平滑调节
- We define a **$2^{nd}$ order random walk** with two parameters $p$ and $q$ which guide the walk: Consider a random walk that just traversed edge $(t, v)$ and now resides at node $v$. The walk now needs to decide on the next step so it evaluates the transition probabilities $\pi_{vx}$ on edges $(v, x)$ leading from $v$. We set the unnormalized transition probability to $\pi_{vx} = \alpha_{pq})(t, x) \cdot w_{vx}$, where $\alpha_{pq}(t, x)\left\{\begin{matrix}  \frac{1}{p} \enspace if \enspace d_{tx} = 0 \\   1 \enspace if \enspace d_{tx}=1 \\ \frac{1}{q} \enspace if \enspace d_{tx} = 2  \end{matrix}\right.$
	- $d_{tx}$ denotes the shortest path distance between nodes $t$ and $x$, must be one of {0, 1, 2}
- ![[Pasted image 20230223170807.png]]
	- $2^{nd}$ order Markovian. 下一时刻的状态与当前状态和上一状态有关
- 空间复杂度 $O(\alpha^2 |V|)$
- 时间复杂度 $O(\frac{l}{k(l - k)})$ per sample, 只有$l$ 可以被人为控制

##### The node2vec algorithm

###### Algorithm 1 The node2vec algorithm
LearnFeatures(Graph $G=(V,E,W)$)， Dimensions $d$, Walks per node $r$, Walk length $l$, Context size $k$, Return $p$, In-out $q$)
	$\pi$ = PreprocessModifiedWeights $(G,p,q)$    # 事先算出图的权重，生成随机游走采样策略
	$G' = (V,E,\pi)$
	Initiralize $walks$ to Empty
	for $iter = 1$ to $r$ do              #
		for all nodes $u \in V$ do    # 每个节点生成 r 个随机游走序列
			$walk$ = node2vecWalk($G', u, l$)    # 生成1个随机游走序列
			Append $walk$ to $walks$
	$f = StochasticGradientDescent(k,d,walks)$    # skip-gram 训练得到节点嵌入表示
	return $f$
node2vecWalk(Graph $G'=(V,E,\pi)$, Start node $u$, Length $l$)    # 生成一个随机游走序列
	initialize $walk$ to $[u]$ 
	for $walk_iter$ = 1 to $l$ do    # $l$ 个节点
		$curr = walk[-1]$    # 当前节点
		$V_{curr}$ = GetNeighbors($curr, G'$)    # 当前节点的邻居节点
		$s$ = AliasSample($V_{curr, \pi}$)    # 根据采样策略（p, q），找到下一个节点
		Append $s$ to $walk$
	return $walk$
- Alias sampling，用于产生下一个随机游走节点，时间复杂度 $O(1)$，用空间（预处理）$O(n)$ 换时间，大量反复抽样情况下优势更突出，离散分布抽样转化为均匀分布抽样

#### Learning edge features
- ...we extend them to pairs of nodes using a bootstrapping approach over the feature representations of the individual nodes
	- Node embedding 扩展到 link embedding
- Given two nodes $u$ and $v$, we define a binary operator $o$...
	- Average
	- Hadamard
	- Weighted-L1
	- Weighted-L2
