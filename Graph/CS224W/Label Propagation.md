### Node Classification
- Given labels of some nodes
- Predict labels of unlabeled nodes
- 这叫做半监督节点分类，用已知类别的节点预测未知类别的节点
	- Transductive 直推式学习
	- Inductive 归纳式学习

### 半监督节点分类问题-求解方法对比
| 方法               | 图嵌入 | 表示学习 | 使用属性特征 | 使用标注 | 直推式 | 归纳式 |
| ------------------ | ------ | -------- | ------------ | -------- | ------ | ------ |
| 人工特征工程       | 是     | 否       | 否           | 否       | /      | /      |
| 基于随机游走的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 基于矩阵分解的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 标签传播           | 否     | 否      | 是/否        | 是       | 是     | 否     |
| 图神经网络         | 是     | 是       | 是           | 是       | 是     | 是       |
- 人工特征工程: 节点重要度、集群系数、Graphlet等
- 基于随机游走的方法，构造自监督表示学习任务实现图嵌入。无法泛化到新节点。例如: DeepWalk、Node2Vec、LINE、SDNE等。
- 标签传播: 假设"物以类聚，人以群分"，利用邻域节点类别猜测当前节点类别。无法泛化到新节点。例如: Label Propagation、Iterative Classification、Belief Propagation、Correct & Smooth等
- 图神经网络：利用深度学习和神经网络，构造邻域节点信息聚合计算图，实现节点嵌入和类别预测。可泛化到新节点。例如：GCN、GraphSAGE、GAT、GIN等。

### Label Propagation
- Message passing 消息传递机制
- 物以类聚 人以群分
- 集体Collective classification，利用领域节点类别信息
- 标签传播和集体分类主讲五种算法，前三种更重要:
    -   Label Propagation（Relational Classification）
    -   Iterative Classification
    -   Correct & Smooth
    -   Belief Propagation
    -   Masked Label Prediction

##  大自然对图的基本假设
- Correlation在大自然大量存在
	- Homophily，the tendency of individuals to associate and bond with similar others 具有相似属性特征的节点更可能相连且具有相同类别
	- Influence，social connections can influence the individual characteristics of a person, 社交会影响节点类别，物以类聚人以群分、近朱者赤近墨者黑
- 最简单的方法：使用KNN最近邻分类
- 对一个节点进行标签分类，不仅需要节点自身的属性特征，更要利用领域节点类别和属性特征

## 标签传播 Label Propagation（Relational Classification）
-   算法步骤：
    1. Initialization，对已知类别标签设置 = {0, 1}，未知标签 = {0.5}
    2. 第一轮计算某节点周围的所有节点值的总和的平均值，算完所有节点，就是加权平均
    3. 开始迭代
    4. 当节点的值都收敛之后，定义一个阈值，进行类别划分
	    - 也有可能maximum number of iterations is reached
- $P(Y_v = c) = \frac{1}{\sum_{(v,u)\in E} a_{V, U}}\sum\limits_{(v,u)\in E} A_{v,u}P(Y_u = c)$
	- 节点v是类别c的概率 = 平均  加权
- Challenges：
    - 仅使用网络连接信息(网络信息和节点类别)，没有使用节点属性信息
    - 模型并不保证收敛

## Iterative Classification
- Classify node $v$ based on its attributes $f_v$ as well as labels $z_v$ of neighbor set $N_v$
- Input: Graph
	- $f_v$: feature vector for node $v$，节点属性特征
	- some nodes $v$ are labeled with $Y_v$
- Approach: 2 classifiers
    - Base classifier $\phi_1(f_v)$，仅使用节点属性特征
    - Relational classifier $\phi_2(f_v, z_v)$，使用节点属性特征和网络连接特征 $Z_v$（领域节点类别信息）
- Computing the summary $z_v$：
	- $Z_v$ = 包含邻域节点类别信息的向量
	- Phase 1: On the labeled training set，来训练上面两个分类器
	- Phase 2: 迭代直至收敛
		- On test set，用训练好的 $\phi_1$ 预测 $Y_v$，用 $Y_v$ 计算 $z_v$，然后再用 $\phi_2$ 预测出所有节点类别
		- Repeat for each node $v$ :
			- 更新领域节点 $z_v$
			- 用新的 $z_v$ 输入到 $\phi_2$ 更新 $Y_v$
		- Iterate until class labels stabilize or max number of iterations is reached
- Note：不保证收敛
- Collective Classification: simultaneous classification of interlinked nodes using correlations
	- 基于 Markov Assumption，the label $Y_v$  of one node $v$ depends on the labels of its neighbors $N_v$ , $P(Y_v)=P(Y_v∣N_v)$
	- Involves 3 steps:
		- Local Classifier: assign initial labels
		- Relational Classifier: capture correlations between nodes
		- Collective Inference: propagate correlations through network
- 既用到了网络特征，又用到了节点信息

## Correct & Smooth
- 可以被视为一种后处理的技巧
- C&S follows the 3-step procedure：
    1. Train a base predictor that predict soft labels(class probabilities) over all nodes，也包含有类别标注的节点
	    - soft labels, 不是非0即1，都有自己的概率，两个概率加和为1
    2. Use the trained base predictor to predict soft labels of all nodes
    3. Post-process the soft predictions using graph structure to obtain the final predictions of all nodes
	    1. Correct step，计算 training error
		    - Need to correct the degree of the errors of the soft labels are biased，让模型对不确定程度进行扩散
	    2. Smooth step，类似 label propagation
		    - 让最终结果变得平滑
- Correct step：
	- The key idea: errors in the base prediction to be positively correlated along edges in the graph
		- error 在图中也有 homophily
	- compute the training errors of nodes
		- 仅计算有标注的节点， ground-truth label - soft label
		- Defined as 0 for unlabeled nodes
	- Diffuse training errors $E^{(0)}$ along the edges
	- Let $A$ be the adjacency matrix, $\tilde A$ be the diffusion matrix
		- Normallized diffusionmatrix $\tilde {A} \equiv D^{-1/2}AD^{-1/2}$, 归一化
			- add self-loop to the adjacency matrix $A$，主对角线=1
			- let $D \equiv Diag(d_1,\cdots,d_N)$ be the degree matrix
			- ![[Pasted image 20230228092302.png]]
			- 性质:
				- all the eigenvalues $\lambda$'s are in the range of [-1, 1]，保证收敛
					- eigenvector for $\lambda = 1$ os $D^{1/2} 1$ (1 is an all-one vector)
					- 作幂运算，不发散
					- if $i$ and $j$ are connected, the weight $\tilde {A}_{ij}$ is $\frac{1}{\sqrt{d_i}\sqrt{d_j}}$
	- Diffusion of training errors, $E^{(t+1)} \leftarrow (1 - \alpha) \times E^{(t)} + \alpha \times \tilde{A} E^{t}$
		- similar to PageRank
		- $\alpha$: hyper-parameter
	- Add the scaled diffused training errors into the predicted soft labels
- Smooth step：
	- Smoothen the corrected soft labels along the edges
	- Diffuse label $Z^{(0)}$ along the graph structure
	- $Z^{(t+1)} \leftarrow (1 - \alpha) \times Z^{(t)} + \alpha \times \tilde{A} Z^{t}$
		- $Z$ 表示节点的置信度
		- 最后收敛结果不一定求和为1，但是大的值仍然表示置信度高

- C&S summary：
	- Correct & Smooth(C&S) 用图的结构信息进行后处理
	- Correct step, 对不确定性（error）进行扩散
	- Smooth step, 对最终的预测结果进行扩散
	- C&S achieves strong performance on semi-supervised node classification
	- Model Agnostic, 可与 GNN 结合

## Loopy Belief Propagation
- 有点像消息传递，节点之间可以传递消息
- Belief Propagation is a *dynamic programming* , 下一时刻的状态仅取决于上一时刻
- When consensus is reached, calculate final belief
- Messga Passing algorithm 类似报数
- Notation：
    - Label-label potential matrix $\psi$：Dependency between a node and its neighbor. $\psi(Y_i, Y_j)$ is proportional to the probability of a node $j$ being in class $Y_j$ given that it has neighbor $i$ in class $Y_i$
	    - 节点和其邻居之间的依赖关系，本身是一个标量
    - Prior belief $\phi$ ：$\phi(Y_i)$ is proportional to the probability of node $i$ beding in class $Y_i$
    - $m_{i \rightarrow j}(Y_j)$ ：节点 $i$ 认为节点 $j$ 是类别 $Y_j$ 的概率
    - $\cal L$：the set of all classes / labels
- Loopy BP Algorithm：
    1. 初始化所有节点信息都为1
    2. Repeat for each node：![[Pasted image 20230228100247.png]]
    3. After convergence：$b_i(Y_i) = \phi_i(Y_i)\prod_{j \in N_i} m_{j \rightarrow i}​(Y_i), \forall Y_i \in \cal L$
- Advantages:
	- easy to program&parallelized
	- can apply to any graph model with ant form of potentials
- Challenges:
	- convergence not guaranteed, especially if many closed loops

##  Masked Label Prediction
- An alternative approach to explicity include node label information
- Inspired from **BERT** 
- 自监督学习 self-supervised
- Idea: treat labels as additional features
	- concatenate the node label matrix $Y$ with the node feature matrix $X$
- Use partially observed labels $\hat Y$ to predict the remaining unobserved labels
    - Training: 随机把一些节点的labels变成0，然后用 $[X, \tilde Y]$,已有的信息去猜出这些信息的label
    - Inference: emply all $\hat Y$ to predict the remaining unlabeled nodes (in the validation/test set)
    - 构造自监督学习场景，迭代优化
    - Similar to link prediction


## 本章总结
- 本章介绍了半监督节点分类的两种方法，直推式学习、归纳式学习
- 介绍了人工特征方法、基于随机游走的方法、基于矩阵分解的方法、标签传播、图神经网络，并比较这些求解方法
- 介绍了5种标签传播和集体分类，Label Propagation、Iterative Classification、Correct & Smooth、Belief Propagation、Masked Label Prediction
- 总结了目前为止的节点分类问题的方法，引出图神经网络
