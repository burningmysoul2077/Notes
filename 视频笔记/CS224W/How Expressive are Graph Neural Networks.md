## Theory of GNNs
- Many GNN models have been proposed
	- GCN  GAT  GraphSAGE  deisgn space
- The expressive power of these GNN models
	- 表达、学习、拟合、区分 ability to distinguish different graph structures
- How to design a maximally expressive GNN model
	- GIN

### GNN Model Example

#### GCN (mean-pool)
- Element-wise mean pooling + Linear + ReLU non-linearity
	- 对应元素求均值

#### GraphSAGE (max-pool)
- MLP + element-wise max-pooling

- Note : Node colors
	- 用不同颜色表示不同Embedding
- 计算图，通过连接结构区分不同节点

#### Local Neighborhood Structures
- 节点度不同 Nodes have different neighborhood structures cuz have different node degrees 
- 邻居的节点度不同 Nodes have the same node degree still have different neighborhood structures cuz their neighbors have different node degrees
- But nodes have the same neighborhood structures cuz they are symmetric within the graph
	- 这时，仅通过图的连接结构，无法区分节点
- GNN的表示方法: 计算图

### Computational Graph
- In each layer, a GNN aggregates neighboring node embeddings
	- through a computational graph
	- GNN每一层聚合邻居信息（节点嵌入），即通过其邻居得到的计算图产生节点嵌入。
- GNN 眼中的计算图，编号无意义，连接长度无意义；颜色有意义
- 计算图与rooted subtree structures 相同
	- Rooted substree structrues  -  recursively unfolding neighboring nodes from the root nodes
- GNN的表达能力 = 区分计算图根节点Embedding的能力
- Injective Function
	- Function $f: X \rightarrow Y$ is injective if it maps different elements into different outputs
	- 单射，每个输入对应唯一输出
- 理想GNN: 不同的计算图根节点，输出不同Embedding
	- 理想GNN的聚合操作应该单射。根据计算图作为输入，然后将所有子树单射到node embeddings，得到结果
	- 要保证计算图这个树中每一层汇聚的的过程都是单射,也就是每一层都是都使用单射邻居聚合函数（保留全部信息），把不同的邻居映射到不同的嵌入上。
	- 最完美的单射聚合操作  -  Hash

### Designing the Most Powerful Graph Neural Network 
- Expressive power of GNNs can be characterized by that of neighbor aggregation functions they use
	- GNN的表示能力取决于其应用的邻居聚合函数。
	- 聚合函数表达能力越强，GNN表达能力越强
	- 单射聚合函数的GNN表达能力最强

#### Neighbor Aggregation
- Neighbor aggregation can be abstracted as a function over a multi-set (a set with repeating elements)
	- 满足单射

##### GCN (mean-pool)
- $Mean(\{x_u\}_{u \in N(v)})$  逐元素求平均
- GCN‘s aggregation function cannor distinguish different multi-sets with the same color proportion
- 因为假设节点特征(颜色)是用独热编码来表示

##### GraphSAGE (max-pool)
- $Max(\{x_u\}_{u \in N(v)})$  逐元素求最大值
- Apply an MLP
- GraphSAGE's aggregation function cannot distinguish different multi-sets with the same set of distinct colors

#### 小结
- GNN的表示能力由其邻居聚合函数决定
- 邻居聚合是 a function over multi-set，multi-set是一个元素可重复的集合
- GCN和GraphSAGE的聚合函数都不能区分某些基本的multi-set，因此都不单射，不够具有表达能力
- 结论，GCN 、GraphSAGE都不是最理想的GNN

#### Design A Neural Network
- Design a neural network that can model injective multiset function
	- 用神经网络拟合单射函数

#### Injective Multi-Set Function
- Theorem, 任何多重集合上的单射函数都可以表示为:
	- $\Phi (\sum\limits_{x\in S} f(x))$
		- $\Phi$  -  非线性激活函数
	- $S: multi-set \rightarrow \Phi(f() + f() +f())$
		- 任意一个多重集合上的单射函数，表示成两个作用部分：一部分是作用于单个元素的函数$f$，这是个单射函数（比如将不同颜色的节点，映射为不同的one-hot向量）；另一部分是非线性映射函数 $\Phi$
		- $f$  -  相当于产生颜色(节点特征)，求和  -  记录颜色个数，  $\Phi$  -  单射函数
- How to model $\Phi$ and $f$
	- 使用多层感知机
	- 万能近似定理 Universal Approximation Theorem
		- 对于一个任意的连续函数,都可以用一个神经网络来近似表示
		- 这里的神经网络，我们是用带一层隐藏层的MLP
	- $MLP_\Phi (\sum\limits_{x\in S} MLP_f(x))$
	- In practice, MLP hidden dimensionality of 100 ~ 500

### Most Expressive - GNN
- Graph Isomorphism Network 图同构网络
	- apply an MLP, element-wise sum, followed by another MLP
- GIN's neighbor aggregation function is injective
- GIN is a "neural network" version of the WL graph kernel(传统图机器学习的特征工程中 Weisfeiler-Lehman Kernel)

#### Relation to WL Graph Kernel
- Color refinement algorithm in WL kernel
	- Given: A graph $G$ with a set of nodes $V$
		- Assign an initial color $c^{(0)}(v)$ to each node $v$
		- Iteratively refine node colors by  $c^{(k + 1)}(v) = HASH(c^{(k)}(v), {\{c^{(k)}(u) \}}_{u \in N(v)})$
			- HASH  -  maps different inputs to different colors  硬编码的单射函数
		- After $K$ steps of color refinement,  $c^{(k)}(v)$ summarizes the structure of $K$-hop neighborhood

#### The Complete GIN Model
- GIN uses a neural network to model the injective HASH function
	- $c^{(k + 1)}(v) = HASH(c^{(k)}(v), {\{c^{(k)}(u) \}}_{u \in N(v)})$
	- $c^{(k)}(v)$  -  root node features
	- ${\{c^{(k)}(u) \}}_{u \in N(v)}\}$  -  neighboring node colors
- Theorem:  any injective function over the tuple can be modeled as
	- ${MLP}_\Phi((1 + \epsilon) \times {MLP}_f(c^{(k)}(v)) + \sum\limits_{u \in N(v)} {MLP}_f(c^{(k)}(u)))$
		- $\epsilon$  -  a learnable scalar
- 如果input feature $c^{(0)}(v)$ 已经是one-hot，则不需要 $MLP_f$
- 只需要 $\Phi$ 来保证单射性
	- $GINConv(c^{(k)}(v), {\{c^{(k)}(u) \}}_{u \in N(v)}) = {MLP}_\Phi((1 + \epsilon) \times (c^{(k)}(v) + \sum\limits_{u \in N(v)} (c^{(k)}(u))$
		- $c^{(k)}(v)$  -  root node features
		- ${\{c^{(k)}(u) \}}_{u \in N(v)}\}$  -  neighboring node colors
		- $MLP_\Phi$  -  MLP provide "one-hot" input feature for the next layer
- So, GIN's node embedding updates
	- Given: A graph $G$ with a set of nodes $V$
		- Assign an initial color $c^{(0)}(v)$ to each node $v$
		- Iteratively refine node colors by  $GINConv(c^{(k)}(v), {\{c^{(k)}(u) \}}_{u \in N(v)})$
			- GINConv  -  maps different inputs to different colors
		- After $K$ steps of color refinement,  $c^{(k)}(v)$ summarizes the structure of $K$-hop neighborhood
|                 | Update target                    | Update function |
| --------------- | -------------------------------- | --------------- |
| WL Graph Kernel | Node colors(one-hot)             | HASH            |
| GIN             | Node embeddings(low-dim vectors) | GINConv                |

- Advantages of GIN over the WL graph kernel
	- Node embeddings 是低维、连续、稠密，且包含语义信息
	- 根据下游任务学习优化

#### Expressive Power of GIN
- In term of expressive Power, GIN = WL graph kernel
	- WL 是表达能力的上届
- Ranking by discriminative power
	- Input:  sum - multiset > mean - distribution > max - set
- GNN 不能识别 图中的环

## 本章总结
- 本章回顾了 GNN的通用框架、训练流程
- 介绍了 GNN 模型的表达能力，一个单独GNN层包含message + aggregation，连接结构，以及GNN 通过连接结构区分不同节点
- 介绍了计算图，区分计算图根节点Embedding的能力，即GNN的表达能力，计算图中编号无意义、连接长度无意义、颜色有意义，理想GNN是不同的计算图根节点，输出不同Embedding
- 介绍了单射函数，最理想的GNN应该使用单射聚合操作，而最完美的单射聚合操作正是HASH
- 介绍了设计理想GNN，聚合操作可以抽象为 a function over a multi-set，根据万能近似定理，选择用神经网络拟合单射函数
- 介绍了GIN ， 现在最理想的GNN，而且GIN 类似于一个神经网络版的WL graph kernel，GIN model的整个过程
- 对比了GIN和WL，确定了GIN和WL的表达能力完全相同，并展望提升GIN
