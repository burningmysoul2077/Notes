## 随机游走 图嵌入 开山之作 DeepWalk论文精读
	 《DeepWalk: Online Learning of Social Representations》
	   - Bryan Prozzi, Rami Al-Rfou, Steven Skiena, Stony Brook University
	   - KDD 2014

### 一分钟介绍DeepWalk
- DeepWalk将graph的每一个节点编码为一个D-维向量(Embedding)，无监督学习
- Embedding中隐式包含了graph中的社群、连接、结构信息，可用于后续节点分类等下游任务，监督学习

### Embedding 嵌入的艺术
- 表示学习
- 直接套用 word2vec，假设相邻节点应该具有相似的embedding

### 官方PPT讲解
- Traditional: A 1st step in ML for graphs is to extract graph features
	- node: degree
	- pairs: # of common neighbors
	- groups: cluster assignments
- Graph Representation : create features by transforming the graph into a lower dimensional latent representation
- DeepWalk: learns a latent representation of adjacency matrices using deep learning techniques developed for language modeling
	- Pros:
		- Scalable - an online algorithm that does not use entire graph at once, 可增量学习
		- Walks as senteces metaphor 套用语言模型
		- Works great 特别是稀疏标注的图分类
	- Cool Idea: **short random walks = sentences**
-   DeepWalk steps：
    1.  输入一个图
    2.  随机游走，采样出随机游走序列
	    - Random Walks：每个节点作为七点，采样 $\gamma$ 条随机游走序列，每个随机游走序列长度为 $t$
    3.  用随机游走序列去训练Word2Vec
	    - Representation Mapping：构造Skip-gram任务，输入中心词的编码表示去预测周围词
    4.  为了解决分类个数过多的问题，使用Softmax（霍夫曼编码数）
	    - Hierarchical Softmax：使用二叉树(霍夫曼树)，输入节点的嵌入表示，进行逻辑二分类回归 logistic binary classifier
    5.  最后得到节点的图嵌入向量表示
	    - Learning：两套要学习的参数，每一个节点的嵌入表、二分类的权重

-   Future Work：
    -   Streaming：不需要知道全图数据，即来即训
    -   Non-Random Walks：可以不随机，有倾向性

## DeepWalk 论文精读
- 《DeepWalk：用于图节点嵌入的在线机器学习算法》

### ABSTRACT
- DEEPWALK, a novel approach for learning latent representations of vertices in a network
	- 一种新颖方法，用于学习图网络中节点的隐式语义表征
- These  latent representations encode social relations in a continuous vector space, which is easilt exploited bt statistical models
	- 这些隐式学习表征能够将图中的连接信息编码成低维、稠密的连续向量空间，就可以直接使用于统计学习模型
- DEEPWALK generalizes recent advancements in language modeling and unsupervised feature learning(or *deep learning*) from sequences of words to graphs
	- DeepWalk把现有的用于单词序列的语言模型和深度学习进行了扩展到了图
- DEEPWALK uses lcoal information obtained from trucated random walks to learn latent representations by treating walks as the equivalent of sentences
	- DeepWalk 使用截断的(有最大长度限制的)随机游走去学习局部结构的隐式表示，在这里将随机游走序列当成句子
- DEEPWALK is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable
	- DeepWalk是一种在线学习算法，可扩展，可迭代增量学习，显然也可并行。

#### Categories and Subject Descriptors
- Database Management: Database Applications - Data Mining
- Artificial Intelligence: Learning
- Pattern Recognition: Model - Statistical

#### Keywords
- social networks; deep learning; latent representations; learning with partial labels; network calssification; online learning

### INTRODUCTION
- 图/网络的表示是很稀疏的
- Sparsity enables the design of efficient discrete algorithms, but can make it harder to generalize in statistical learning
	- 稀疏性使得一些离散算法(路径相关，最短路径)可行，但对于统计机器学习模型是个大麻烦
- In the relational classification problem, the links between feature vectors violate the traditional $i.i.d$ assumption.
	- 在相关性分类问题中，特征向量之间的连接违反了传统机器学习的独立同分布假设
- Techniques to address this problem typically use approximate inference techniques to leverage the dependency information to imporve classification results
	- 近似推断是用来解决这个问题的，它属于传统方法，它尝试利用节点的相关性信息来改进分类结果
- We distance ourselves from these appraoches by learning label-independent representations of the graph
	- 有别于这些传统方法，我们是采用学习与节点特征、标签无关的图像表示、
- Strong performance with our representations is possible with very simple linear classifiers(e.g. logistic regression)
- Our representations are general, and can be combined with any classificaiton method(including iterative inference methods)
	- 这个方法类似label propagation

### PROBLEM DEFINITION
- We consider the problem of classifying members of a social network into one or more categories
- Let $G = (V, E)$, where $V$ represent the members of the network, $E$ are their connections, $E \subset (V \times V)$, and $G_L = (V, E, X, Y)$ is a partially labeled social network, with attributes $X \in \mathbb{R}^{|V| \times S}$ where $S$ is the size of the feature space for each attribute vector, and $Y \in \mathbb{R}^{|V| \times |\cal Y|}, \cal Y$ is the set of labels
	- $E$ 是 $V$ 行 $V$ 列的邻接矩阵；$X$ 是特征，每个节点都有 $S$ 维特征；$|V|$ 表示节点个数； $Y$ 是类别个数，每个节点的标注
- In the literature, this is known as the relational classification(or the collective classification problem)
- Traditional approaches to relational classification pose the problem as an inference in an undirected Markov network, and then use iterative approximative inference algoraithms(such as the iterative classification algorithm, Gibbs Sampling, or label relaxation) to compute the posterior distribution of labels given the network structure
	- 传统方法把图当成无向马尔科夫网络，用迭代近似预测来计算出标签的后验概率分布
- We propse a different approach to capture the network topology information. Instead of mixing the label space as part of the feature space, we propose an unsupervised method which learns features that capture the graph strcture *independent* of the labels' distribution
	- 我们的新方法不把标签和连接特征混合，只用随机游走序列采样出连接信息，仅在embedding中编码连接信息
- This sepatation between the structural representation and the labeling task avoids cascading errors, which can occur in iterative methods.
- Our goal is to learn $X_E \in \mathbb{R}^{|V| \times d}$, where $d$ is small number of latent dimensions.
	- 得到一个低维 稠密的向量，$d$ 是embedding的维度
- These low-dimensional representations are distributed; meaning each social phenomena is expressed by a subset of the social concepts expressed by the space/
	- 分布式: 各个维度都不为 0，每一个元素都很重要
- Using thers structural features, we will augment(补充) the attributes space to help the classification decision
	- 反映连接信息的embedding + 反映节点本身的特征，来增强节点分类

### LEARNING SOCIAL REPRESENTATIONS
- We seek to learn social representations with the following characteristics:
	- Adaptability - Real social networks are constantly evolving
		- 灵活可变，弹性扩容
	- Community aware - The distance between latent dimensions should represent a metric for evaluationg social similarity between the corresponding members of the network. This allows generalization in networks with homophily
		- 反映社群聚类信息。原图中相近的节点嵌入后依然相近，这样可以对同质性进行进一步分析
	- Low dimensional - When labeld data is scarce low-dimensional models generalize better, and speed up convergence and inference
		- 低维数。低维度嵌入有助于防止过拟合
	- Continuous - In addition to providing a nuanced细微差别(微观) view of community membership, a continuous representation has smooth decision boundaries between communities which allows more robust classification
		- 连续性。拟合出平滑的决策边界，可以用到神经网络

#### Random Walks
- We denote a random walk rooted at vertex $v_i$ as $W_{v_i}$
- It is a stochastic process with random variables $W_{v_i}^1, W_{v_i}^2,\dots,W_{v_i}^k$ such taht $W_{v_i}^{k + 1}$ is a vertex chosen at random from the neighbors of vertex $v_k$
	- $W_{v_i}^k$, $k$ : 第 k 步，$v_i$ : 起始节点；第k+1个点是从k节点的邻居中选择。
- They are also the foundation of a class of *output sensitive* algorithms which use them to compute local community structure information in time sublinear to the size of the input graph
	- output sensitvie, 至少要遍历一遍全图。sublinear，仅看local局部，不需遍历全图
- It is this connection to local structure that motivates us to use a *stream* of short random walks as our basic tool for extracting information from a network.
- Several random walkers can simultaneouslt explore different parts of the same graph
- We can iteratively update the learned model with new random walks from the changed region in time sub-linear to the entire graph

#### Connection: Power laws
- Having chosen online random walks as our primitive for capturing graph structure, we now need a suitable method to capture this information
- If the degree distribution of a connected graph follows a power law(i.e. scale-free), we observe that the frequency which vertices appear in the short random walks will also follow a power-law distribution
	- 无标度图，在图中有一些非常重要的中枢节点，他们的度非常高，形成幂律分布(长尾分布、二八定律)，严重分布不均匀。在实际世界中，少数Hub中枢节点拥有极多连接。
- *Zipf*'s law
	- 词频 与 词频排序名次的常数次幂成反比，意味着只有极少数的词被经常使用

#### Language Modeling
- The goal of language modeling is to estimate the likelihood of a spccific sequence of words appearing in a corpus
	- 语言模型的目标是估计似然概率，也叫通顺度模型
- More formally, given a sequence of words $W_1^n = (w_0, w_1,\dots, w_n), where \enspace w_i \in \cal{V}$($\cal{V}$ is the vocabulary), we would like to maximize the Pr$(w_n|w_0, w_1,\dots, w_{n - 1})$ over all the training corpus
	- 用前 n-1 个词来预测第 n 个词
- Recent work in representation learning has focused on using probabilistic neural networks to build general representations of words...
	- word2vec、glove
- In this work, we present a generalization of language modeling to explre the graph through a stream of short random walks
- These walks can be thought of as short sentences and phrases in a special language; the direct analog is to estimate the likelihood of observing vertex $v_i$ given all the previous vertices visited so far in the random walk, i.e. Pr$(v_i | (v_1, v_2,\dots, v_{i-1}))$
	- 随机游走当做短句和短语，节点类比成单词。用前 i-1 个节点来预测第 i 个节点
- Our goal is to learn a latent representation, not only a probability distribution of node co-occurrences, and so we introduce a mapping function $\Phi : v \in V \rightarrow \mathbb{R}^{|V| \times d}$
	- 我们要用节点的embedding来预测，而不是用节点本身。所以引入一个映射函数，即查表， $\Phi(v_1)$就是 $v_1$ 的embedding
- This mapping $\Phi$ represents the latent social representation associated with each vertex $v$ in the graph
- In practice, we represent $\Phi$ by a $|V| \times d$ matrix of free parameters, which will serve later on as our $X_E$
	- 表是 V行 d列的矩阵，矩阵中的每一个元素没有大小限制
- The problem then, is to estimate the likelihood: Pr$(v_i | (\Phi(v_1), \Phi(v_2), \dots, \Phi(v_{i - 1})))$
	- 用前 i-1 个节点的embedding预测第 i 个节点
- However, as the walk length grows, computing this conditional probability becomes unfeasible
	- $P(v_2| \Phi(v_1)) \times P(v_3|\Phi(v_1), \Phi(v_2)) \times \dots$ 是不可行的
- A recent relaxation in language modeling turns the prediction problem on its head.
	- word2vec
- First, instead of using the context to predict a missing word, it uses one word to predict the context.
	- CBow, Skip-Gram,构造自监督学习场景
- In terms of vertex representation modeling, this yields the optimization problem: $\mathop{minimize}\limits_{\Phi} -\log Pr(\{ v_{i - w},\dots, v_{i + w} \} / v_i | \Phi(v_i))$
	- DeepWalk(Skip-Gram)损失函数，log对数里是最大化似然概率，条件概率里是上文w个节点和下文w个节点，再除以 i 节点自己。让最大似然概率越大越好，也就是-log越小越好
- First, the order independence assumption better captures a sense of 'nearness' that is provided by random walks
	- 随机游走生成的图，顺序没有意义。还能更好捕捉邻近的信息
- Moreover, this relaxation is quite useful for speeding up the training time by building small models as one vertex is given at a time
	- 模型较小，一次输入一个节点，预测周围节点
- Solving the optimization problem from equation builds representations that capture the shared similarities in local graph structure between vertices
- By combining both truncated random walks and language models we formulate a method which satisfies all of our desired properties.
- This method generates representations of social networks that are low-dimensional, and exist in a continuous vector space...
	- 低维、稠密、连续，包含节点的结构、连接、社群特征。虽然不包含类别信息，但可用于预测类别信息

### METHOD

#### Overview
- As in any language modeling algorithm, the only required input is a corpus and a vocabulary
- DEEPWALK considers a set if short truncated random walks its own corpus, and the graph vertices as its own vocabulary($\mathcal{V} = {V}$)

### Algorithm: DEEPWALK
- The algorithm consists of two main compinents: first a **random walk generator**, and second, an **update procedure**

	#### Algorithm 1 DEEPWALK $(G, w, d, \gamma, t)$
	Input: graph $G(V, E)$
		window size $w$           # 左右窗口宽度
		embedding size $d$      # Embedding 维度
		walks per vertex $γ$     # 每个节点作为起始节点生成随机游走的次数
		walk length $t$             # 随机游走最大长度
	Output: matrix of vertex representations $\Phi \in \mathbb{R}^{|V| \times d}$
		1. Initialization: Sample $\Phi$ from $u^{|V| \times d}$
		2. Build a binary Tree $T$ from $V$
		3. for $i$ = 0 to $\Phi$ do          # 重复一下过程 $\gamma$次
		4.     $\cal O$ = Shuffle($V$)         # 随机打乱节点顺序
		5.     for each $v_i \in \cal O$ do    # 遍历graph中的每个点
		6.         $W_{v_i} = RandomWalk(G, v_i, t)$    # 生成一个随机游走序列
		7.         SkipGram($\Phi, W_{v_i}, w$)    # 由中心节点embedding预测周围节点，更新embedding
		8.     end for
		9.  end for

	Algorithm: SkipGram($\Phi, W_{v_i}, w$)
	1:  for each $v_j \in W_{v_i}$ do    # 遍历当前随机游走序列里的每个节点
	2:      for each $u_k \in W_{v_i}[j - w : j + w]$ do    # 遍历该节点周围窗口里的每个点
	3:          $J(\Phi) = -\log Pr(u_k|\Phi(v_j))$    # 计算损失函数
	4:          $\Phi = \Phi - \alpha \ast \frac{\partial J}{\partial \Phi}$    # 梯度下降更新embedding矩阵
	5:      end for
	6:  end for

- The **random walk generator** takes a graph $G$ and samples uniformly a random vertex $v_i$ as the root of the random walk $W_{v_i}$. A walk samples uniformly from the neighbors of the last vertex visited until the maximum length ($t$) is reached.
- In practice, our implementation specifies a number of random walks $\gamma$ or length $t$ to start at each vertex

#### SkipGram
- SkipGram is a language model that maximizes the co-occurence probability among the words that appear within a window, $w$, in a sentence
- Pr$({v_{i - w}, \dots, v_{i + w}} \ v_i | \Phi(v_i)) = \prod\limits_{j=i-w,j\neq i}^{i+w} Pr(v_j|\Phi(v_i))$
	- 输入中心词的embedding来预测周围词, 计算时把中心词的embedding和周围某个词向量做乘积即可
- For example, modeling the previous problem using logistic regression would result in a huge number of labels(that is equal to $|V|$) which could be in millions or billions
- Such models require vast computational resources which could span a whole cluster of computers
- ..., we ubstrad use the **Hierarchical Softmax** to approximate the probability distribution

#### Hierarchical Softmax
- Computeing the partition function(normalization factor) is expensive, so instead we will factorize the conditional probability using Hierarchical softmax
	- partition function, 配分函数，softmax分母归一化项，项太多没法直接计算
- We assign the vertices to the leaves of a binary tree, turning the prediction problem into maximizing the probability of a specific path in the hierarchy
- If the path to vertex $u_k$ is identified by a sequence of tree nodes($b_0, b_1, \dots, b_{\left \lceil \log|V| \right \rceil }$), ($b_0 = root, b_{\left \lceil \log|V| \right \rceil } = u_k$) then  Pr$(u_{k} | \Phi(v_j)) = \prod\limits_{l = 1}^{\left \lceil \log|V| \right \rceil }$ Pr$(b_l|\Phi(v_j))$
	- 每个节点变成叶子节点。非叶子节点，是逻辑回归二分类器，参数量和embedding维度一致. N个节点，会产生 N-1 个逻辑回归
	- ![[Pasted image 20230222090203.png]]
	- Now, Pr$(b_l|\Phi(v_j))$ could be modeled by a binary classifier...Pr$(b_l|\Phi(v_j)) = 1 / (1 + e^{{-\Phi(v_j)} \cdot  {\Psi(b_l)}})$  
		- 词的embedding和逻辑回归的权重维度一致，做数量积，越大，式子越接近1
	- ...calculatung Pr$(u_{k} | \Phi(v_j))$ from $|V|$ to $O(\log|V|)$
	- We can speed up the training process further, by assigning shorter paths to the frequent vertices in the random walks. 
		- 把高频节点放在靠近根节点的地方，这样找的更快
	- Huffman coding is used to reduce the access time of frequent elements in the tree
		- 霍夫曼编码 贪心策略

#### Optimization
- The model parameter set is $\theta = \{ \Phi, \Psi \}$ where the size of each is $O(d|V|)$
	- deepwalk 两套权重，一同优化
		- N 个节点的D-维 embedding
		- N-1 个逻辑回归，每个有 D 个权重
- Stochastic gradient descent(SGD) is used to optimize these parameters. The derivatives are estimated using the back-propagation algorithm. The learning rate  $\alpha$ for SGD is initially set to 2.5% at the beginning of the training and then decreased linearly with the number of vertices that are seen so far

### Parallelizability
- 多线程异步并行

### Algorithm Variants

#### Streaming
- 在未知全图时，直接用采样出的随机游走训练embedding
- ...the learning rate $\alpha$ to a small constant value
- 无需提前构建树

#### Non-random walks
- 不随机的自然游走
- Some graphs are created as a by-product of agents interacting with a sequence of elements
- Graphs sampled in this way will not only capture information related to network structure, but also to the frequency at which paths are traversed
	- 不仅可以反映连接是否存在，更可以反映连接的权重
- Sentences can be viewed as purposed walks through an appropriately designed language network, and language models like SkipGram are designed to capture this behavior

### RELATED WORK
- We *learn* our latent social representations, instead of computing statistics related to centrality or partitioning
	- embedding通过机器 学习得到，而非人工统计构造
- We do not attempt to extend the classification procedure itself
	- 无监督，不考虑节点的label信息，只考虑graph连接信息。再用无监督的embedding和标注训练有监督的分类模型
- We propose a scalable online method which uses only local information. Most methods require global information and are offline
	- 在线学习，随来随学，仅使用graph的局部信息
- We apply unsupervised representationl learning to graphs

### CONCLUSIONS
- In addition to beding effective and scalable, our approach is also an appealing generalizaiton of language modeling. This connection is mutually beneficial.Advances in language modeling may continue to generate improved latent representations for networks. In our view, language modeling is actually sampling from an unobservable language graph
	- 语言模型可以看作是对不可见的隐式graph建模
	- 可见graph的分析方法可以促进非可见graph的研究
