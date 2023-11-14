# Pre-trained Models for Natural Language Processing: A Survey

- Recently, the emergence of pre-trained models (PTMs)* has brought natural language processing (NLP) to a new era. In this survey, we provide a comprehensive review of PTMs for NLP. We first briefly introduce language representation learning and its research progress. Then we systematically categorize existing PTMs based on a taxonomy from four different perspectives. Next, we describe how to adapt the knowledge of PTMs to downstream tasks. Finally, we outline some potential directions of PTMs for future research. This survey is purposed to be a hands-on guide for understanding, using, and developing PTMs for various NLP tasks.

>  最近，预训练模型 (PTMs) 的涌现将自然语言处理带入了一个新时代。在此篇综述中，我们将对用于 NLP 的 PTMs 提供一个全面审视。首先，我们简要介绍语言表示学习 (representation learning) 及其研究进展。然后我们会从四个不同角度将现有的 PTMs 系统地进行分类 (taxonomy)。接下来，我们描述如何将 PTMs 知识应用于下游任务。最后，对 PTMs 未来研究方向进行展望。本综述旨在为理解、使用和开发用于各种 NLP 任务的 PTMs 提供上手指南。

---

# 1 Introduction

-   With the development of deep learning, various neural networks have been widely used to solve Natural Language Processing (NLP) tasks, such as convolutional neural networks (CNNs) [1–3], recurrent neural networks (RNNs) [4, 5], graphbased neural networks (GNNs) [6–8] and attention mechanisms [9, 10]. One of the advantages of these neural models is their ability to alleviate the feature engineering problem. Non-neural NLP methods usually heavily rely on the discrete handcrafted features, while neural methods usually use lowdimensional and dense vectors (aka. distributed representation) to implicitly represent the syntactic or semantic features of the language. These representations are learned in specific NLP tasks. Therefore, neural methods make it easy for people to develop various NLP systems. 

>   随着深度学习的发展，各种神经网络被广泛用于解决 NLP 任务，如卷积神经网络 (CNNs)[1-3]、循环神经网络 (RNNs)[4,5]、基于图的神经网络 (GNNs)[6-8]和注意机制 (attention mechanisms)[9,10]。这些神经网络模型的优点之一是它们能够减轻特征工程问题。非神经网络模型的 NLP 方法通常严重依赖于离散的手工特征，而神经方法通常使用低维、稠密向量 (分布式表示) 来隐式地表示语言的语法或语义特征。这些表示是在特定的 NLP 任务中学习到的。因此，神经网络方法为人们开发各种 NLP 系统提供了便利。

-  Despite the success of neural models for NLP tasks, the performance improvement may be less significant compared to the Computer Vision (CV) field. The main reason is that current datasets for most supervised NLP tasks are rather small (except machine translation). Deep neural networks usually have a large number of parameters, which make them overfit on these small training data and do not generalize well in practice. Therefore, the early neural models for many NLP tasks were relatively shallow and usually consisted of only 1∼3 neural layers. 

>  尽管神经网络模型在 NLP 任务中取得了成功，但与计算机视觉 (CV) 领域相比，性能改进可能不那么显著。主要原因是目前大多数有监督的 NLP 任务的数据集相当小 (除了机器翻译)。深度神经网络通常具有大量的参数 (parameters)，这使得它们在这些小的训练数据上过拟合 (overfit)，在实践中不能很好地泛化 (generalize)。因此，许多 NLP 任务的早期神经网络模型相对较浅，通常仅由 1 ~ 3 个神经层组成。

- Recently, substantial work has shown that pre-trained models (PTMs), on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks and can avoid training a new model from scratch. With the development of computational power, the emergence of the deep models (i.e., Transformer [10]), and the constant enhancement of training skills, the architecture of PTMs has been advanced from shallow to deep. The first-generation PTMs aim to learn good word embeddings. Since these models themselves are no longer needed by downstream tasks, they are usually very shallow for computational efficiencies, such as Skip-Gram [11] and GloVe [12]. Although these pre-trained embeddings can capture semantic meanings of words, they are context-free and fail to capture higher-level concepts in context, such as polysemous disambiguation, syntactic structures, semantic roles, anaphora. The second-generation PTMs focus on learning contextual word embeddings, such as CoVe [13], ELMo [14], OpenAI GPT [15] and BERT [16]. These learned encoders are still needed to represent words in context by downstream tasks. Besides, various pre-training tasks are also proposed to learn PTMs for different purposes. 

>  近年来，大量的研究表明，PTMs 在大型语料库上可以学习通用语言表示，这有利于下游的 NLP 任务，并且可以避免从头开始训练新模型。随着计算能力的发展，深层模型 (如Transformer[10]) 的出现，以及训练技能的不断提高，PTMs 的架构已经从浅层向深层推进。__第一代 PTMs 旨在学习好的词嵌入 (word embeddings)__。由于下游任务不再需要这些模型本身，因此它们对于计算效率来说通常非常浅薄，例如 Skip-Gram [11] 和 GloVe [12]。虽然这些预训练的嵌入可以捕获词的语义，但它们是上下文无关的，不能捕获上下文中的更高层次的概念，如多义消歧 (polysemous disambiguation)、句法结构 (syntactic structures)、语义角色 (semantic roles)、指代 (anaphora)。__第二代 PTMs 专注于学习上下文词嵌入 (contextual word embeddings)__，如 CoVe[13]、ELMo[14]、OpenAI GPT[15]和BERT[16]。下游任务仍然需要这些习得的编码器 (encoders) 来表示上下文中的单词。此外，针对不同的目的，还提出了各种预训练任务来学习 PTMs。

>  polysemous disambiguation

-  [#todo]

-  The contributions of this survey can be summarized as follows: 
	1. Comprehensive review. We provide a comprehensive review of PTMs for NLP, including background knowledge, model architecture, pre-training tasks, various extensions, adaption approaches, and applications. 
	2. New taxonomy. We propose a taxonomy of PTMs for NLP, which categorizes existing PTMs from four different perspectives: 1) representation type, 2) model architecture; 3) type of pre-training task; 4) extensions for specific types of scenarios. 
	3. Abundant resources. We collect abundant resources on PTMs, including open-source implementations of PTMs, visualization tools, corpora, and paper lists. 
	4. Future directions. We discuss and analyze the limitations of existing PTMs. Also, we suggest possible future research directions. 

>  本综述的贡献可以总结为以下几点：
> 	 1.  全面审视。 本文提供了包括背景知识、模型架构、与训练任务、各种扩展、自适应方法和应用等在内的对 PTMs 的全面审视。
> 	 2.  新的分类法 (taxonomy)。本文提出了面向 NLP 的 一种 PTMs 分类法，从四个不同角度对现有 PTMs 进行归类：
> 			 1.  表示类型
> 			 2.  模型架构
> 			 3.   预训练任务类型
> 			 4.   针对特殊类型场景的扩展
> 	 3.  丰富资源。本文收集了大量关于 PTMs 的资料，包括 PTMs 的开源实现、可视化工具、语料库 (corpora，corpus 的复数) 和 论文列表。
> 	 4.  未来方向。本文讨论和分析了现有 PTMs 的局限性，并提出了未来可能的研究方向。

-  The rest of the survey is organized as follows. Section 2 outlines the background concepts and commonly used notations of PTMs. Section 3 gives a brief overview of PTMs and clarifies the categorization of PTMs. Section 4 provides extensions of PTMs. Section 5 discusses how to transfer the knowledge of PTMs to downstream tasks. Section 6 gives the related resources on PTMs. Section 7 presents a collection of applications across various NLP tasks. Section 8 discusses the current challenges and suggests future directions. Section 9 summarizes the paper.

>  综述的剩余部分组织如下。第 2 节概述了 PTMs 的背景概念和常用符号。第 3 节给出了 PTMs 的简要概述，并澄清了 PTMs 的分类。第 4 节提供了 PTMs 的扩展。第 5 节讨论了如何将 PTMs 的知识转移到下游任务。第 6 节给出了关于 PTMs 的相关资源。第 7 节介绍了各种 NLP 任务的应用集合。第 8 节讨论了当前的挑战并提出了未来的方向。第 9 节总结全文。