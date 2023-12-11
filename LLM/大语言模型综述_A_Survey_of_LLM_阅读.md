# A Survey of Large Language Models

-  Abstract—Ever since the Turing Test was proposed in the 1950s, humans have explored the mastering of language intelligence by machine. Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable artificial intelligence (AI) algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pretraining Transformer models over large-scale corpora, showing strong capabilities in solving various natural language processing (NLP) tasks. Since the researchers have found that model scaling can lead to an improved model capacity, they further investigate the scaling effect by increasing the parameter scale to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement, but also exhibit some special abilities (e.g., incontext learning) that are not present in small-scale language models (e.g., BERT). To discriminate the language models in different parameter scales, the research community has coined the term large language models (LLM) for the PLMs of significant size (e.g., containing tens or hundreds of billions of parameters). Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT (a powerful AI chatbot developed based on LLMs), which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. Considering this rapid technical progress, in this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Furthermore, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions. This survey provides an up-to-date review of the literature on LLMs, which can be a useful resource for both researchers and engineers.

>  摘要: 自 20 世纪 50 年代提出 图灵测试(Turing Test) 以来，人类一直在探索机器掌握语言智能的方法。语言本质上是一个由语法规则支配的复杂的、精细的人类表达系统。开发出有能力理解和掌握语言的人工智能 (AI) 算法是一项重大挑战。在过去的二十年里，_语言建模(language modelling)_ 作为语言理解和生成的一种主要方法得到了广泛的研究，从 _统计语言模型 statistical language models_ 发展到 _神经网络语言模型 neural language models_。近年来，通过对大规模语料库上的 Transformer 模型进行预训练，提出了预训练语言模型 (PLMs)，该模型在解决各种 NLP 任务方面表现出较强的能力。由于研究人员已经发现 _模型扩容 model scaling_ 可以导致模型能力的提高，他们通过将参数扩容到更大的尺寸来进一步研究扩容效应。有趣的是，当参数规模超过一定水平时，这些扩大的语言模型不仅取得了显著的性能提升，而且还表现出了一些在小规模语言模型 (如BERT) 中不存在的特殊能力(如 _in-context learning 上下文学习_)。为了区分不同参数规模的语言模型，研究界为具有显著规模 (例如，包含数百亿或数千亿参数) 的 PLMs 创造了术语 _大型语言模型(LLM)_。近年来，LLMs 的研究在学术界和工业界都取得了很大的进展，其中最显著的进展是 ChatGPT (基于 LLMs 开发的功能强大的人工智能聊天机器人)的推出，引起了社会的广泛关注。LLMs 的技术发展对整个人工智能社区产生了重要影响，这将彻底改变我们开发和使用 AI 算法的方式。考虑到这种快速的技术进步，在本问中，我们通过介绍背景，关键发现和主流技术来回顾 LLMs 的最新进展。我们特别关注了 LLMs 的四个主要方面，即 _预训练 pre-training、自适应调整 adaptation tuning、利用 utilization 和能力评估 capacity evaluation_。此外，我们还总结了开发 LLMs 的可用资源，并讨论了未来发展方向的剩余议题。此篇综述提供了最新的 LLMs 文献回顾，这对研究人员和工程师来说都是一个有用的资源。

---

# 1 Introduction

>   “ The limits of my language mean the limits of my world."
> 																				  -  Ludwig Wittgenstein

-  LANGUAGE is a prominent ability in human beings to express and communicate, which develops in early childhood and evolves over a lifetime [1, 2]. Machines, however, cannot naturally grasp the abilities of understanding and communicating in the form of human language, unless equipped with powerful artificial intelligence (AI) algorithms. It has been a longstanding research challenge to achieve this goal, to enable machines to read, write, and communicate like humans [3]. 

>  语言是人类一种突出的表达和交流能力，这种能力从幼儿时期开始发展，并在一生中不断演进[1,2]。然而，除非配备强大的人工智能(AI)算法，否则机器无法自然地掌握以人类语言形式进行理解和交流的能力。要实现这一目标，即使机器能够像人类一样阅读、写作和交流，一直是一个长期的研究挑战[3]。

## 语言建模

-  Technically, language modeling (LM) is one of the major approaches to advancing language intelligence of machines. In general, LM aims to model the generative likelihood of word sequences, so as to predict the probabilities of future (or missing) tokens. The research of LM has received extensive attention in the literature, which can be divided into four major development stages: 

>  从技术上讲，_语言建模  language modeling(LM)_ 是提高机器语言智能的主要途径之一。一般来说，LM 旨在对词序列的生成似然进行建模，从而预测未来 (或缺失) tokens 的概率。LM 的研究在文献中得到了广泛的关注，它可以分为四个主要的发展阶段:

### 统计语言模型

-  __Statistical language models (SLM)__. SLMs [4–7] are developed based on statistical learning methods that rose in the 1990s. The basic idea is to build the word prediction model based on the Markov assumption, e.g., predicting the next word based on the most recent context. The SLMs with a fixed context length n are also called n-gram language models, e.g., bigram and trigram language models. SLMs have been widely applied to enhance task performance in information retrieval (IR) [8, 9] and natural language processing (NLP) [10–12]. However, they often suffer from the curse of dimensionality: it is difficult to accurately estimate high-order language models since an exponential number of transition probabilities need to be estimated. Thus, specially designed smoothing strategies such as backoff estimation [13] and Good–Turing estimation [14] have been introduced to alleviate the data sparsity problem. 

>  __统计语言模型(SLM)__。SLM[4-7] 是基于 20 世纪 90 年代兴起的 _统计学习方法 statistical learning methods_ 开发的。其基本思想是建立基于马尔可夫假设的单词预测模型，例如，根据最近的上下文预测下一个单词。具有固定上下文长度 n 的 SLM 也称为 n元语言模型，如 _二元语言模型 bigram_ 和 _三元语言模型 trigram_。SLM 在 _信息检索(IR) information retrieval_[8,9]和 自然语言处理(NLP)[10-12]中被广泛应用于提高任务性能。然而，它们经常遭受维度的诅咒:由于需要估计指数数量级的转移概率，故而很难准确估计高阶语言模型。因此，引入了特殊设计的平滑策略，如 backoff估计[13] 和 Good-Turing 估计[14]来缓解数据稀疏性问题。

### 神经网络语言模型

-  __Neural language models (NLM)__. NLMs [15–17] characterize the probability of word sequences by neural networks, e.g., recurrent neural networks (RNNs). As a remarkable contribution, the work in [15] introduced the concept of distributed representation of words and built the word prediction function conditioned on the aggregated context features (i.e., the distributed word vectors). By extending the idea of learning effective features for words or sentences, a general neural network approach was developed to build a unified solution for various NLP tasks [18]. Further, word2vec [19, 20] was proposed to build a simplified shallow neural network for learning distributed word representations, which were demonstrated to be very effective across a variety of NLP tasks. These studies have initiated the use of language models for representation learning (beyond word sequence modeling), having an important impact on the field of NLP. 

>  __神经语言模型 (NLM)__。 NLM [15-17] 通过神经网络表征词序列的概率，例如，递归神经网络 (RNNs)。作为一个显著的贡献，[15] 中的工作引入了词的 _分布式表示 distributed representation_ 的概念，并建立了基于聚合上下文特征的词预测函数(即 _分布式词向量 the distributed word vector_)。通过扩展学习单词或句子的有效特征的思想，开发了一种通用的神经网络方法来为各种 NLP 任务构建统一的解决方案 [18]。此外，提出了 word2vec[19,20] 用来构建一个简化的浅层神经网络，用于学习分布式单词表示，并被证明在各种 NLP 任务中非常有效。这些研究开创了语言模型在表示学习中的应用 (超越了词序列建模)，对自然语言处理领域产生了重要影响。

### 预训练模型

-  __Pre-trained language models (PLM)__. As an early attempt, ELMo [21] was proposed to capture context-aware word representations by first pre-training a bidirectional LSTM (biLSTM) network (instead of learning fixed word representations) and then fine-tuning the biLSTM network according to specific downstream tasks. Further, based on the highly parallelizable Transformer architecture [22] with self-attention mechanisms, BERT [23] was proposed by pretraining bidirectional language models with specially designed pre-training tasks on large-scale unlabeled corpora. These pre-trained context-aware word representations are very effective as general-purpose semantic features, which have largely raised the performance bar of NLP tasks. This study has inspired a large number of follow-up work, which sets the “pre-training and fine-tuning” learning paradigm. Following this paradigm, a great number of studies on PLMs have been developed, introducing either different architectures [24, 25] (e.g., GPT-2 [26] and BART [24]) or improved pre-training strategies [27–29]. In this paradigm, it often requires fine-tuning the PLM for adapting to different downstream tasks. 

>  __预训练语言模型(PLM)__。作为早期的尝试，ELMo [21] 被提出，通过首先预训练一个双向LSTM (biLSTM) 网络(而不是学习固定的单词表示)，然后根据特定的下游任务对 biLSTM 网络进行 _微调 fine-tuning_，来捕获上下文感知的词表示。此外，基于高度并行化的 Transformer 架构[22]和自注意力机制，BERT [23] 通过在大规模无标记语料库上预训练双向语言模型和专门设计的预训练任务而提出。这些预训练的上下文感知词表示是非常有效的通用语义特征，这在很大程度上提高了 NLP 任务的性能标准。该研究启发了大量的后续工作，建立了 “__预训练和微调 pre-training and fine-tuning__” 的学习 _范式 paradigm_。遵循这一范式，对 PLMs 进行了大量研究，引入了不同的架构[24,25] (例如 GPT-2[26] 和 BART[24]) 或改进的预训练策略 [27-29]。在此范式中，通常需要对 PLM 进行微调以适应不同的下游任务。

-  __Large language models (LLM)__. Researchers find that scaling PLM (e.g., scaling model size or data size) often leads to an improved model capacity on downstream tasks (i.e., following the scaling law [30]). A number of studies have explored the performance limit by training an ever larger PLM (e.g., the 175B-parameter GPT-3 and the 540Bparameter PaLM). Although scaling is mainly conducted in model size (with similar architectures and pre-training tasks), these large-sized PLMs display different behaviors from smaller PLMs (e.g., 330M-parameter BERT and 1.5Bparameter GPT-2) and show surprising abilities (called emergent abilities [31]) in solving a series of complex tasks. For example, GPT-3 can solve few-shot tasks through in-context learning, whereas GPT-2 cannot do well. Thus, the research community coins the term “large language models (LLM)” 1 for these large-sized PLMs [32–35], which attract increasing research attention (See Figure 1). A remarkable application of LLMs is ChatGPT2 that adapts the LLMs from the GPT series for dialogue, which presents an amazing conversation ability with humans. We can observe a sharp increase of the arXiv papers that are related to LLMs after the release of ChatGPT in Figure 1. 

### 大语言模型

>  __大型语言模型(LLM)__。研究人员发现，扩展 PLM (例如扩展模型大小或数据大小) 通常会提高下游任务的模型能力 (即遵循伸缩法则[30])。许多研究通过训练更大的PLM (例如，参数为 175b 的 GPT-3 和参数为 540b 的 PaLM) 来探索性能极限。虽然扩展主要是在模型大小 (具有相似的架构和预训练任务) 上进行的，但这些大型 PLMs 和较小的 PLM (例如 330M 参数的 BERT 和 1.5B 参数的 GPT-2)相比会表现出不同的行为，并且在解决一系列复杂任务时表现出惊人的能力 (称为 涌现能力[31])。例如，GPT-3 可以通过 _上下文学习 in-context learning_ 解决 _少样本任务 few-shot_，而GPT-2则做得不好。因此，研究界将这些大型 PLMs 称为 “_大型语言模型 (LLM) large language models_”[32-35]，并引起了越来越多的研究关注 (见图1)。LLMs 的一个显著应用是ChatGPT，它将 GPT 系列中的 LLMs 用于对话，呈现出与人类惊人的对话能力。如图1所示，我们可以看到 ChatGPT 发布后，与 LLMs 相关的 arXiv 论文急剧增加。

-  In the existing literature, PLMs have been widely discussed and surveyed [36–39], while LLMs are seldom reviewed in a systematic way. To motivate our survey, we first highlight three major differences between LLMs and PLMs. First, LLMs display some surprising emergent abilities that may not be observed in previous smaller PLMs. These abilities are key to the performance of language models on complex tasks, making AI algorithms unprecedently powerful and effective. Second, LLMs would revolutionize the way that humans develop and use AI algorithms. Unlike small PLMs, the major approach to accessing LLMs is through the prompting interface (e.g., GPT-4 API). Humans have to understand how LLMs work and format their tasks in a way that LLMs can follow. Third, the development of LLMs no longer draws a clear distinction between research and engineering. The training of LLMs requires extensive practical experiences in large-scale data processing and distributed parallel training. To develop capable LLMs, researchers have to solve complicated engineering issues, working with engineers or being engineers. 

>  在现有文献中，PLMs 已经被广泛讨论和综述 [36-39]，而很少会系统地回顾 LLMs。为了激励我们的综述，我们首先强调 LLMs 和 PLMs 之间的三个主要区别。首先，LLMs 显示出一些令人惊讶的涌现能力，这在以前较小的 PLMs 中可能没有观察到。这些能力是语言模型在复杂任务上表现的关键，使人工智能算法前所未有地强大和有效。其次，LLMs 将彻底改变人类开发和使用人工智能算法的方式。与小型 PLMs 不同，访问 LLMs 的主要方法是通过提示接口 (例如GPT-4 API)。人们必须理解 LLMs 是如何工作的，并以 LLMs 可以接受的方式格式化他们的任务。第三，LLMs 的发展不再明确区分研究和工程。LLMs 的训练需要大量的大规模数据处理和分布式并行训练的实践经验。为了培养有能力的 LLMs，研究人员必须解决复杂的工程问题，与工程师合作或成为工程师。

-  Nowadays, LLMs are posing a significant impact on the AI community, and the advent of ChatGPT and GPT-4 leads to the rethinking of the possibilities of artificial general intelligence (AGI). OpenAI has published a technical article entitled “Planning for AGI and beyond”, which discusses the short-term and long-term plans to approach AGI [40],and a more recent paper has argued that GPT-4 might be considered as an early version of an AGI system [41]. The research areas of AI are being revolutionized by the rapid progress of LLMs. In the field of NLP, LLMs can serve as a general-purpose language task solver (to some extent), and the research paradigm has been shifting towards the use of LLMs. In the field of IR, traditional search engines are challenged by the new information seeking way through AI chatbots (i.e., ChatGPT), and New Bing3 presents an initial attempt that enhances the search results based on LLMs. In the field of CV, the researchers try to develop ChatGPT-like vision-language models that can better serve multimodal dialogues [42–45], and GPT-4 [46] has supported multimodal input by integrating the visual information. This new wave of technology would potentially lead to a prosperous ecosystem of real-world applications based on LLMs. For instance, Microsoft 365 is being empowered by LLMs (i.e., Copilot) to automate the office work, and OpenAI supports the use of plugins in ChatGPT for implementing special functions. 

>  如今，LLMs 对人工智能社区产生了重大影响，ChatGPT 和 GPT-4 的出现导致人们重新思考 _人工通用智能(AGI) artificial general intelligence_ 的可能性。OpenAI 发表了一篇题为 “Planning for AGI and beyond” 的技术文章，其中讨论了想去接近 AGI 的短期和长期计划[40]，最近的一篇论文认为 GPT-4 可能被视为 AGI 系统的早期版本 [41]。随着 LLMs 的快速发展，人工智能的研究领域正在发生革命性的变化。在自然语言处理领域，LLMs 可以作为通用语言任务求解器 (在某种程度上)，研究范式已经转向使用 LLMs。在 IR 领域，传统的搜索引擎受到了通过 AI 聊天机器人 (即 ChatGPT) 寻找信息的新方式的挑战，New Bing 提出了基于 LLMs 增强搜索结果的初步尝试。在 CV 领域，研究人员试图开发类似 ChatGPT 的视觉语言模型，以更好地服务于多模态对话[42-45]，GPT-4[46] 通过整合视觉信息支持多模态输入。这一新的技术浪潮可能会促使产生一个基于 LLMs 的现实世界应用的繁荣生态系统。例如，Microsoft 365 正在通过 LLMs (即 Copilot) 来实现办公工作的自动化，OpenAI 支持使用 ChatGPT 中的 _插件 plugin_ 来实现特殊功能。

-  Despite the progress and impact, the underlying principles of LLMs are still not well explored. Firstly, it is mysterious why emergent abilities occur in LLMs, instead of smaller PLMs. As a more general issue, there lacks a deep, detailed investigation of the key factors that contribute to the superior abilities of LLMs. It is important to study when and how LLMs obtain such abilities [47]. Although there are some meaningful discussions about this problem [31, 47], more principled investigations are needed to uncover the “secrets“ of LLMs. Secondly, it is difficult for the research community to train capable LLMs. Due to the huge demand of computation resources, it is very costly to carry out repetitive, ablating studies for investigating the effect of various strategies for training LLMs. Indeed, LLMs are mainly trained by industry, where many important training details (e.g., data collection and cleaning) are not revealed to the public. Thirdly, it is challenging to align LLMs with human values or preferences. Despite the capacities, LLMs are also likely to produce toxic, fictitious, or harmful contents. It requires effective and efficient control approaches to eliminating the potential risk of the use of LLMs [46]. 

>  尽管取得了进展和影响，LLMs 的基本原则仍然没有得到很好的探索。首先，为什么涌现能力出现在 LLMs 中，而不是较小的 PLMs 中，这是一个谜。作为一个更普遍的问题，缺乏对 LLMs 卓越能力的关键因素的深入、详细的调查。研究 LLMs 何时以及如何获得这种能力是很重要的[47]。虽然关于这个问题已经有了一些有意义的讨论[31,47]，但要揭开 LLMs 的“秘密”，还需要更多的原则性研究。其次，研究界很难培养出有能力的 LLMs。由于对计算资源的巨大需求，为了调查各种 LLMs 训练策略的效果，进行重复的、有针对性的研究是非常昂贵的。事实上，LLMs 主要是由产业界训练的，许多重要的训练细节 (如数据收集和清理) 并没有向公众透露。第三，将 LLMs 与人类的价值观或偏好结合起来是一项挑战。LLMs 尽管很有能力，但也可能产生有毒，虚构或有害的内容。它需要有效和高效的控制方法来消除使用 LLMs 的潜在风险[46]。

-  Faced with both opportunities and challenges, it needs more attention on the research and development of LLMs. In order to provide a basic understanding of LLMs, this survey conducts a literature review of the recent advances in LLMs from four major aspects, including pre-training (how to pretrain a capable LLM), adaptation (how to effectively adapt pre-trained LLMs for better use), utilization (how to use LLMs for solving various downstream tasks) and capability evaluation (how to evaluate the abilities of LLMs and existing empirical findings). We thoroughly comb the literature and summarize the key findings, techniques, and methods of LLMs. For this survey, we also create a GitHub project website by collecting the supporting resources for LLMs, at the link https://github.com/RUCAIBox/LLMSurvey. We are also aware of several related review articles on PLMs or LLMs [32, 36, 38, 39, 43, 48–54]. These papers either discuss PLMs or some specific (or general) aspects of LLMs. Compared with them, we focus on the techniques and methods to develop and use LLMs and provide a relatively comprehensive reference to important aspects of LLMs. 

>  机遇与挑战并存，LLMs 的研究与发展需要更多的关注。为了对 LLMs 有一个基本的了解，本文从预训练 (如何预训练一个有能力的 LLMs)、适应 (如何有效地改造 LLMs 以更好地使用)、利用 (如何使用 LLMs 解决各种下游任务)和能力评估(如何评估 LLMs 的能力和现有的实证结果) 四个主要方面对 LLMs 的最新进展进行了文献综述。我们彻底梳理文献，总结 LLMs 的主要发现、技术和方法。为了这项调查，我们还创建了一个 GitHub 项目网站，通过收集 LLMs 的支持资源，[链接](https://github.com/RUCAIBox/LLMSurvey)。我们也知道一些关于 PLMs 或 LLMs 的相关综述文章[32,36,38,39,43,48 - 54]。这些论文要么讨论 PLMs，要么讨论 LLMs 的一些具体(或普遍)方面。与他们相比，我们侧重于开发和使用 LLMs 的技术和方法，并对 LLMs 的重要方面提供了相对全面的参考。

-  The remainder of this survey is organized as follows: Section 2 introduces the background for LLMs and the evolution of GPT-series models, followed by the summarization of available resources for developing LLMs in Section 3. Sections 4, 5, 6, and 7 review and summarize the recent progress from the four aspects of pre-training, adaptation, utilization, and capacity evaluation, respectively. Then, Section 8 discusses the practical guide for prompt design, and Section 9 reviews the applications of LLMs in several representative domains. Finally, we conclude the survey in Section 10 by summarizing the major findings and discuss the remaining issues for future work.

-  本文的其余部分组织如下: 第 2 节介绍了 的背景和 GPT 系列模型的演变，然后在第 3 节总结了开发 LLMs 的可用资源。第 4、5、6 和 7 节分别从预训练、改造、利用和能力评估四个方面回顾和总结了最近的进展。然后，第 8 节讨论了 _提示设计 prompt design_ 的实践指南，第 9 节回顾了 LLMs 在几个代表性领域的应用。最后，我们在第 10 节中总结了调查的主要发现，并讨论了未来工作中有待解决的问题。

----

# 2 概述

-  在本节中，我们概述了 LLMs 的背景，然后总结了 GPT 系列模型的技术演变。

## 2.1 LLMs 的背景

-  通常，大语言模型 (LLMs) 是指包含数千亿 (或更多) 参数的 Transformer 语言模型[32]，这些模型是在海量文本数据上训练的，如GPT-3[55]、PaLM[56]、Galactica[35] 和 LLaMA[57]。LLMs 在理解自然语言和解决复杂任务 (通过文本生成) 方面表现出强大的能力。为了快速了解 LLMs 的工作原理，本部分介绍了 LLMs 的基本背景，包括 _伸缩法则 scaling laws_，_涌现能力 emergent abilities_ 和 _key techniques关键技术_。

### 伸缩法则/缩放定律

- __LLMs 的伸缩法则__。目前，LLM 主要建立在 Transformer 架构上[22]，其中 _多头注意力 multi-head attention_ 层堆叠在一个非常深的神经网络中。现有的 LLMs 采用类似的 Transformer 架构和预训练目标 (例如，语言建模) 作为小语言模型。然而，LLMs 显著地扩展了模型大小、数据大小和总计算 (放大数量级)。大量研究表明，_规模 scaling_ 可以在很大程度上提高 LLMs 的模型能力[26,55,56]。因此，建立一个定量的方法来表征 _伸缩效应 scaling effect_ 是有用的。接下来，我们为 Transformer 语言模型引入两个具有代表性的伸缩法则[30,34]。



  -  _KM 伸缩法则 KM scaling law_ 2020年，Kaplan 等人 [30](OpenAI团队) 首次提出对神经语言模型分别针对 _模型大小 (N) model size_、_数据集大小 (D) dataset size_ 和 _训练计算量 (C) the amount of training compute_ 三个主要因素建立模型性能的 _幂律 power-law_ 关系模型。给定 _计算预算 $c$ compute budget_，他们经验地提出了伸缩法则的三个基本公式:

    -  $L(N) = (\frac{Nc}{N})^{\alpha_N}, \alpha_N ~ 0.076, N_c ~ 8.8 \times 10^{13}$ (1)

    -  $L(D) = (\frac{D_c}{D})^{\alpha_D}, \alpha_D ~ 0.095, D_c ~ 5.4 \times 10^{13}$

    -  $L(c) = (\frac{C_c}{C})^{\alpha_C}， \alpha_C ~ 0.050, C_c ~ 3.1 × 10^8$

  -  其中 $L(·)$ 表示交叉熵损失 (nats)。这三个定律是在一些假设 (例如，对其中一个因素的分析不应该受到其他两个因素的瓶颈影响)下，通过拟合不同数据大小 (22M 到 23B 个 tokens)、模型大小 (768M 到 1.5B 个 _非嵌入参数 non-embedding parameters_ )和训练计算的模型性能得出的。结果表明，模型的性能对这三个因素有很强的依赖关系。

	  -  _Chinchilla 伸缩法则_。作为另一项代表性研究，Hoffmann 等人 [34](Google DeepMind团队) 提出了一种替代形式的伸缩法则来指导 LLMs 的计算最优训练。他们通过改变更大范围的模型大小 (70M 到 16B) 和数据大小 (5B 到 500B) 来指导了严格的实验，并拟合了一个类似的但不同系数的伸缩法则，如下 [34]:

	    -  $L(N, D) = E + \frac{a} {N^α} + \frac{B} {D^β}$，(2)

	    -  其中 E = 1.69, A = 406.4, B = 410.7， α = 0.34 和 β = 0.28。通过优化约束$C≈6ND$ 下的损失 $L(N, D)$，他们发现计算预算对模型大小和数据大小的最优分配可以推导为:
	    -   $N_{opt}(C) = G \frac{C}{6}^a$, $D_{opt}(C) = G^{−1} \frac{C}{6}^b$，(3)
	    -   其中 $a = \frac{α}{α+β}$， $b = \frac{β}{α+β}$， G 是可由 a、b、α 和 β 计算的 _伸缩系数 scaling coefficient_。[34] 分析出,基于计算预算的增长, _KM scaling law_ 支持将更大的预算分配给模型大小而不是数据大小, 而 _Chinchilla scaling law_ 认为, 这两个大小应该增加同等的规模,即 方程(3) a 和 b 有着同样的值。
	    
-  尽管一些限制的假设的是, 这些伸缩法则提供了一个直观的理解效果,使其预测在训练中的 LLMs 性能变成可能[46]。然而，根据伸缩法则，有些能力 (例如上下文学习[55]) 是不可预测的，只有当模型大小超过一定水平时才能观察到(如下所述)。

### 涌现能力

-  __LLMs 的涌现能力 Emergent Abilities of LLMs__。 在文献中， LLMs 的涌现能力被正式定义为 “在小模型中不存在但在大模型中出现的能力”，这是 LLMs 区别于以往 PLMs 的最显著特征之一。它进一步引入了涌现能力发生时的一个显著特征[31]: 当规模达到一定水平时，性能显著高于随机。由此类推，这种涌现模式与物理学中的 _相变现象 phase transition_ 有着密切的联系[31,58]。原则上，涌现能力可以定义为一些复杂的任务[31,59]，而我们更关心的是可以解决各种任务的一般能力。在这里，我们简要地介绍了 LLMs 的三种典型的涌现能力和具有这种能力的代表性模型。

	-  _上下文学习/语境学习 In-context learning_。语境学习 (in-context learning, ICL) 能力由GPT-3正式引入[55]:假设语言模型已经提供了 _自然语言指令 natural language instruction_ 和/或多个任务演示，它可以通过补充完整输入文本的词序列来生成对测试实例的预期输出，而不需要额外的训练或 _梯度更新 gradient update_。在 GPT 系列模型中，175B GPT-3 总体表现出较强的 ICL 能力，而 GPT-1 和 GPT-2 表现不佳。这种能力还取决于具体的下游任务。例如，ICL 能力可以出现在13B GPT-3 的算术任务 (例如，3位数的加减法)中，但175B  GPT-3 甚至不能很好地工作在波斯语 QA 任务中[31]。

	-  _指令遵循 instruction following_。通过对通过自然语言描述格式化的多任务数据集的混合进行 _微调 fine-tuning_ (称为 _指令微调 instruction tuning_) ，LLMs 在以指令形式描述的未曾见过的任务上表现良好[28,61,62]。通过指令微调，LLMs 可以在不使用显式示例的情况下遵循新任务的任务指令，从而提高泛化能力。根据[62]中的实验，当模型大小达到 68B 时，指令微调过的 LaMDA - PT[63]在未曾见的任务上开始显著优于未调优的 LaMDA - PT，而在 8B 或更小的模型尺大小下则没有。最近的一项研究[64]发现，PaLM 至少需要 62B 的模型大小才能在四个评估基准 (即MMLU, BBH, TyDiQA 和 MGSM) 中的各种任务上表现良好，尽管更小的大小可能足以满足某些特定任务 (例如 MMLU)。

	-  _逐步的推理 step-by-step reasoning_。对于小语言模型，通常很难解决涉及多个推理步骤的复杂任务，例如:数学文字题。相比之下，使用 _思维链 (CoT) chain-of-thought_ 提示策略[33]，LLMs 可以通过使用包含中间推理步骤的提示机制来获得最终答案来解决这类任务。据推测，这种能力可能通过代码训练获得[33,47]。一项实证研究[33]表明，当将 CoT 提示应用于模型大小大于 60B 的 PaLM 和 LaMDA 变体时，可以带来性能提升 (在算术推理基准上)，而当模型大小超过 100B 时，其优于标准提示的优势变得更加明显。此外，CoT提示对不同任务的性能改善似乎也有所不同，例如，PaLM 中 GSM8K > MAWPS > SWAMP[33]。

### 关键技术

-  __LLMs 的关键技术__。LLMs 已经走了很长一段路才发展到现在的状态: _一般和有能力的学习 general and capable learners_。在开发过程中，提出了许多重要的技术，这些技术在很大程度上提高了 LLMs 的能力。在这里，我们简要列出了几个(潜在的)促使 LLMs 成功的重要技术，如下所示。

	-  _规模/伸缩 Scalling_。 如前所述，Transformer 语言模型中存在明显的规模效应: 更大的模型/数据大小和更多的训练计算通常会导致模型能力的提高[30,34]。GPT-3 和 PaLM 作为两个代表性模型，分别将模型大小增加到 175B 和 540B 来探索伸缩极限。由于计算预算通常是有限的，因此可以进一步使用伸缩法则来进行计算效率更高的计算资源分配。例如，在相同的计算预算下，Chinchilla (拥有更多的训练 tokens) 通过增加数据规模来优于其对应的模型 Gopher (拥有更大的模型大小)[34]。此外，由于预训练数据的质量在模型能力中起着关键作用，因此数据伸缩应该伴随着仔细的清理过程。
	-  _训练 Training_。由于模型大小巨大，成功培养一个有能力的 LLM 是非常具有挑战性的。对于 LLMs _网络参数 network parameters_ 的学习，需要采用 _分布式训练算法 distributed training algorithms_，其中往往联合使用多种并行策略。为了支持分布式训练，已经发布了几个优化框架来促进并行算法的实现和部署，如 DeepSpeed[65] 和 Megatron-LM[66-68]。此外，_优化技巧 optimization tricks_ 对于训练稳定性和模型性能也很重要，例如，重新启动以克服训练损失峰值[56]和混合精度训练[69]。最近，GPT-4[46] 提出开发特殊的基础设施和优化方法，用小得多的模型可靠地预测大型模型的性能。
	-  _能力诱导 Ability eliciting_。经过大规模语料库的预训练，LLMs 被赋予了作为通用任务求解者的潜在能力。LLMs 在执行某些特定任务时，这些能力可能不会被明确展示出来。作为一种技术方法，设计合适的任务指令或特定的上下文学习策略来激发这种能力是有用的。例如，思维链提示已被证明可以通过包含中间推理步骤来解决复杂的推理任务。此外，我们可以使用自然语言表达的任务描述对 LLMs 进行指令微调，以提高 LLMs 在未曾见任务上的泛化性。这些诱导技术主要与 LLMs 的涌现能力相对应，在小型语言模型上可能不会显示出相同的效果。
	-  _对齐调优 Alignment tuning_。 由于 LLMs 是为了捕获预训练语料库的数据特征 (包括高质量和低质量数据) 而训练的，因此它们很可能会对人类产生有毒、有偏见甚至有害的内容。有必要使 LLMs 与人类价值观保持一致，例如，乐于助人，诚实和无害。为此，InstructGPT[61]设计了一种有效的调优方法，使 LLMs 能够遵循预期的指令，该方法利用了 _带有人类反馈的强化学习技术 reinforcement learning with human feedback_[61,70]。它将人类与精心设计的标签策略结合在训练循环中。ChatGPT 确实是在与 InstructGPT 类似的技术基础上开发的，后者在产生高质量、无害的响应方面显示出强大的对齐能力，例如，拒绝回答侮辱性的问题。
	-  _操纵工具 Tools manipulation_。 从本质上讲，LLMs 被训练为大量纯文本语料库上的文本生成器，因此在不能最好地以文本形式表达的任务 (例如，数值计算) 上表现不佳。此外，它们的能力也只限于训练前的数据，例如，无法获取最新信息。为了解决这些问题，最近提出的一种技术是使用外部工具来弥补 LLMs 的不足[71,72]。例如，LLMs 可以利用计算器进行精确计算[71]，利用搜索引擎检索未知信息[72]。最近，ChatGPT 启用了使用外部插件 (现有或新创建的应用程序) 的机制，这等同于 LLMs 的 “眼睛和耳朵”。这种机制可以广泛扩展 LLMs 的能力范围。

-  此外，许多其他因素 (如硬件的升级) 也促成了 LLMs 的成功。目前，我们的讨论仅限于开发 LLMs 的主要技术方法和关键发现。

## 2.2 GPT 系列模型的技术演变 Technical Evolution of GPT-series Models

![[Pasted image 20230807104709.png]]

![[Pasted image 20230807104846.png]]

-  由于与人类交流的出色能力，ChatGPT 自发布以来就引发了人工智能社区的兴奋。ChatGPT 是基于强大的 GPT 模型开发的，具有特别优化的会话能力。考虑到人们对 ChatGP T和 GPT 模型日益增长的兴趣，我们特别讨论了 GPT 系列模型的技术演变，简要总结了它们在过去几年中的进展。_GPT 模型的基本原则是通过语言建模将世界知识压缩到仅含解码器的 Transformer 模型中，这样它就可以恢复 (或记忆) 世界知识的语义，并充当通用任务求解器_。成功的两个关键点是: (1)训练能够 _准确预测下一个单词 accurately predict the next word_ 的仅含解码器的 Transformer 语言模型和 (2) _扩展语言模型的大小 scaling up the size of language models_。总体而言，OpenAI 在 LLMs 上的研究大致可以分为以下几个阶段。

-  __早期探索 Early Explorations__ 。根据对 Ilya Sutskever (OpenAI 的联合创始人兼首席科学家) 的一次采访，在 OpenAI 的早期，就已经探索过用语言模型来接近智能系统的想法，并尝试使用递归神经网络 (RNN)[104]。随着 Transformer 的出现，OpenAI 开发了两个初始的 GPT 模型，即 GPT-1[105] 和 GPT-2[26]，这可以作为随后更强大的模型，即 GPT-3 和 GPT-4 的基础。

	-  _GPT-1_。2017年，Google 引入了 Transformer 模型[22]，OpenAI 团队迅速将他们的语言建模工作适应了这种新的神经网络架构。他们在 2018 年发布了第一个 GPT 模型，即 GPT-1[105]，并创造了缩写术语 GPT 作为模型名称，代表 _生成式预训练 Generative Pre-Training_。GPT-1是基于生成式、仅含解码器的 Transformer 架构开发的，采用了 _无监督预训练 unsupervised pretraining_ 和 _有监督微调 supervised fine-tuning_ 的混合方法。GPT-1 建立了 GPT-系列模型的核心架构，建立了自然语言文本建模的基本原则，即预测下一个单词。
	
	-  _GPT-2_。GPT-2[26] 采用与 GPT-1 类似的架构，将参数规模增加到 1.5B，使用大型网页数据集 WebText 进行训练。正如 GPT-2 论文中所述，它试图通过 _无监督语言建模 unsupervised language modeling_ 来执行任务，而无需使用标记数据进行明确的微调。为了激励该方法，他们引入了一种用于多任务求解的概率形式，即 p(输出|输入，任务) (文献[106]中也采用了类似的方法)，它预测了以输入和任务信息为条件的输出。为了对这种条件概率进行建模，语言文本可以天然作为统一的方式来格式化输入、输出和任务信息。通过这种方式，解决任务的过程可以转换为生成解决方案文本的词预测问题。此外，他们为这一想法引入了一个更正式的声明: “由于 (特定任务的) 监督目标与无监督 (语言建模) 目标相同，但仅在序列的子集上进行评估，因此无监督目标的全局最小值也是监督目标的全局最小值 (对于各种任务)”[26]12。对这一说法的基本理解是，每个 (NLP) 任务都可以被视为基于世界文本子集的词预测问题。因此，如果对无监督语言建模进行训练，使其具有足够的恢复世界文本的能力，则无监督语言建模可以解决各种任务。GPT-2 论文中的这些早期讨论在 Jensen Huang 对 Ilya Sutskever 的采访中得到了回应: “神经网络学习的是产生文本的过程的一些表示。这些文本实际上是世界的投影。你对下一个词的预测越准确，保真度就越高，你在这个过程中获得的分辨率就越高……“。

-  __能力飞跃 Capacity Leap__。尽管 GPT-2 旨在成为一个 “_无监督多任务学习者 unsupervised multitask learner_”，但与 _监督微调的最先进方法 supervised fine-tuning state-of-the-art methods_ 相比，它的总体性能较差。由于它具有相对较小的模型大小，因此它在下游任务中进行了广泛的微调，特别是对话任务 [107,108]。在 GPT-2 的基础上，GPT-3通过扩展 (几乎相同的) 生成预训练架构展示了关键的能力飞跃。

	-  _GPT-3_。GPT-3[55] 于 2020 年发布，将模型参数扩大到 175B 的更大规模。在 GPT-3 的论文中，它正式引入了 _上下文学习(ICL) in-context learning_ 的概念，该概念以 _小样本或零样本 few-shot or zero-shot_ 的方式利用 LLMs。ICL 可以教导 (或指导) LLMs 以自然语言文本的形式理解任务。使用 ICL, LLMs 的预训练和使用收敛到相同的语言建模范式: _预训练预测基于上下文的以下文本序列，而 ICL 预测正确的任务解决方案，如果给定任务描述和演示，也可以将其格式化为文本序列_。GPT-3 不仅在各种 NLP 任务中表现出非常出色的性能，而且在一些需要 _推理 reasoning 或领域适应 domain adaptation_ 能力的特殊设计任务中也表现出色。虽然 GPT- 3的论文没有明确讨论 LLMs 的涌现能力，但我们可以观察到可能超越基本伸缩法则的较大性能飞跃[30]，例如，较大的模型具有明显更强的 ICL 能力 (见GPT-3论文的原始图1.2[55])。总的来说，GPT-3 可以被视为从 PLMs 到 LLMs 演变过程中的一个重要里程碑。经验证明，将神经网络扩展到相当大的规模可以导致模型能力的巨大增加。

-  __能力增强 Capacity Enhancement__。由于强大的能力，GPT- 3 已经成为为 OpenAI 开发更强大的 LLMs 的基础模型。总的来说，OpenAI 探索了两种主要方法来进一步改进GPT-3模型，即 _对代码数据进行训练 training on code data 和与人类偏好保持一致 alignment with human preference_，具体如下。

	-  _代码数据训练 Training on code data_。原始 GPT-3 模型 (在纯文本上预训练) 的一个主要限制是缺乏对复杂任务的推理能力，例如：完成代码和解决数学问题。为了增强这种能力，OpenAI 于 2021 年 7 月引入了 Codex[89]，这是一个在大型 GitHub 代码语料库上进行微调的 GPT 模型。研究表明，Codex 可以解决非常困难的编程问题，并在解决数学问题时显著提高性能[109]。此外，2022 年 1 月报道了一种用于训练文本和代码嵌入的对比方法[110]，该方法被证明可以改进一系列相关任务 (即 _线性探测分类 linear-probe classification_、文本搜索和代码搜索)。实际上，GPT-3.5 模型是基于基于代码的 GPT 模型(即 code- davincii -002) 开发的，这表明对代码数据的训练是提高 GPT 模型的建模能力、特别是推理能力的一个非常有用的实践。此外，也有人推测，对代码数据进行训练可以大大提高 LLMs 的思维链提示能力[47]，但这仍值得进一步研究，并进行更彻底的验证。
	
	-  _人类对齐 Human alignment_。OpenAI对 人类对齐的相关研究可以追溯到2017 年(或更早) :OpenAI 博客上发表了一篇题为 “_从人类偏好中学习 learning from human preferences_” 的博客文章，描述了一项应用 强化学习(RL) 从人类注释的 _偏好对比 preference comparisons_ 中学习的工作[70] (类似于图 9 中 InstructGPT对齐算法中的 _奖励训练 reward training_ 步骤)。在这篇 RL 论文发表后不久[70]，_近端策略优化(PPO) Proximal Policy Optimization_ 的论文[111]于 2017 年 7 月发表。它现在已经成为从人类偏好中学习的基础强化学习算法[61]。随后在2020 年 1 月，使用上述 RL 算法对 GPT-2 进行了微调[70,111]，该算法利用人类偏好来提高 GPT-2 在 NLP 任务上的能力。同年，另一项研究[112]以类似的方式训练了一个优化人类偏好的总结模型。在这些前期工作的基础上，2022 年 1 月提出了 InstructGPT[61]，用于改进 GPT-3 模型的人类对齐，该模型正式建立了一种三阶段的 _基于人类反馈强化学习(RLHF) reinforcement learning from human feedback_ 算法。请注意，OpenAI 的论文和文档中似乎很少使用“_指令微调 instruction tuning_”的措辞，取而代之的是 _对人类演示的有监督微调 supervised fine-tuning on human demonstrations_ (即 RLHF 算法的第一步[61])。除了提高指令遵循能力外，RLHF 算法在减轻 LLMs 产生有害或有毒物质的问题上特别有用，这是 LLMs 在实践中安全部署的关键。OpenAI 在一篇技术文章[113]中描述了他们的对齐研究方法，其中总结了三个有前途的方向: “_训练AI系统使用人类反馈，协助人类评估并进行对齐研究 training AI systems to use human feedback, to assist human evaluation and to do alignment research_”。
	
	-  这些增强技术导致改进后的 GPT-3 模型具有更强的能力，OpenAI 将其称为 GPT-3.5 模型(参见 3.1 节关于 OpenAI API 的讨论)。

-  __语言模型的里程碑 The Milestones of Language Models__。基于所有的探索努力，OpenAI 已经实现了两个重要的里程碑，即 ChatGPT[114] 和 GPT-4[46]，这在很大程度上提高了现有AI系统的容量门槛。

	-  _ChatGPT_。2022 年 11 月，OpenAI 发布了基于 GPT 模型 (GPT-3.5 和 GPT-4) 的会话模型 ChatGPT。正如官方博客文章所介绍的[114]，ChatGPT 的训练方式与 InstructGPT 类似 (原文中称为 “ InstructGPT 的兄弟模型”)，但专门针对对话进行了优化。他们报告了 ChatGPT 和 InstructGPT 在数据收集设置上的训练差异: 人工生成的对话 (扮演用户和人工智能的角色) 与 InstructGPT 数据集以对话格式相结合，用于训练 ChatGPT。ChatGPT 在与人类交流方面表现出了卓越的能力: 拥有丰富的知识储备，对数学问题进行推理的技能，在多回合对话中准确追踪上下文，并且与人类安全使用的价值观非常一致。后来，ChatGPT 支持 _插件机制 plugin mechanism_，这进一步扩展了 ChatGPT 与现有工具或应用程序的能力。到目前为止，它似乎是人工智能历史上最强大的聊天机器人。ChatGPT 的推出对未来的人工智能研究具有重大影响，它为探索类人人工智能系统提供了启示。
	
	-  _GPT-4_。另一个显著的进步是 GPT-4[46] 于 2023 年 3 月发布，它将文本输入扩展到多模态信号。总体而言，GPT-4 比 GPT-3.5 解决复杂任务的能力更强，在许多评估任务上表现出较大的性能提升。最近的一项研究[41]通过对人为生成的问题进行定性测试来研究 GPT-4 的能力，这些问题跨越了各种各样的困难任务，并表明 GPT-4 可以比之前的 GPT 模型 (如 ChatGPT) 实现更优越的性能。此外，由于六个月的迭代校准 (加上在 RLHF 训练中额外的安全奖励信号)，GPT-4 对恶意或挑衅性查询的响应更安全。在技术报告中，OpenAI 强调了如何安全地开发 GPT-4，并应用了一些干预策略来缓解 LLMs 可能出现的问题，如幻觉、隐私和过度依赖。例如，他们引入了称为 _红队 red teaming_ 的机制[115]，以减少危害或有毒物质的产生。作为另一个重要方面，GPT- 4是在一个完善的深度学习基础设施上开发的，具有改进的优化方法。他们引入了一种称为 _可预测伸缩 predictable scalling_ 的新机制，可以在模型训练期间使用一小部分计算准确预测最终性能。

-  尽管取得了巨大的进步，但这些高级 LLMs 仍然存在局限性，例如，在某些特定环境中产生带有事实错误或潜在风险反应的幻觉[46]。LLMs 的更多限制或问题将在第 7 节讨论。开发功能更强、更安全的 LLMs 是长期存在的研究挑战。从工程角度来看，OpenAI 采用迭代部署策略[116]，遵循五个阶段的开发和部署生命周期来开发模型和产品，旨在有效降低模型使用的潜在风险。在下文中，我们将深入研究技术细节，以便对它们是如何开发的有一个具体的了解。

# 3 LLMs 的资源 RESOURCES OF LLMs

-  考虑到具有挑战性的技术问题和对计算资源的巨大需求，开发或复制 LLMs 绝不是一件容易的事情。一种可行的方法是学习现有 LLMs 的经验，并复用公共可用资源进行增量开发或实验研究。在本节中，我们简要总结了用于开发 LLMs 的公开可用资源，包括 _模型检查点 model checkpoints_ (或 api)、_语料库 corpora_ 和 _库 libraries_。

## 3.1 公共可用的模型检查点或 APIs Publicly Available Model Checkpoints or APIs

-  考虑到模型预训练的巨大成本，训练有素的模型检查点对于研究界的 LLMs 研究和开发至关重要。由于 _参数规模是使用 LLMs 时需要考虑的关键因素_，我们将这些公共模型分为两个规模级别 (即 _数百亿参数 tens of billions of parameters_ 和 _数千亿参数 hundreds of billions of parameters_)，这有助于用户根据自己的资源预算确定合适的资源。此外，对于 _推理 inference_，我们可以直接使用公共 APIs 来执行我们的任务，而无需在本地运行模型。接下来，我们将介绍公开可用的模型检查点和 APIs。

-  __Models with Tens of Billions of Parameters 具有数百亿参数的模型__。除 LLaMA[57](最大版本包含65B个参数)、NLLB[82](最大版本包含54.5B个参数)、Falcon[117](最大版本包含40B个参数)外，该类模型的参数尺度大多在 10B - 20B 之间。该范围内的其他模型包括 mT5[74]、PanGu-α[75]、T0[28]、GPTNeoX - 20B[78]、CodeGen[77]、UL2[80]、Flan-T5[64] 和 mT0[84]。其中 Flan-T5 (11B 版本)可以作为研究 _指令微调 instuction tuning_ 的首选模型，因为它从三个方面探索指令微调[64]: _增加任务数量 increasing the number of tasks_，_伸缩模型大小 scaling the model size_，以及使用 _思维链提示数据进行微调 fine-tuning with chain-of-thought prompting data_。此外，CodeGen (11B 版本) 作为一种为生成代码而设计的 _自回归语言模型 autoregressive language model_，可以认为是探索代码生成能力的一个很好的候选。它还引入了一个专门针对多回合程序综合的新基准 MTPB[77]，组成包括 115 个专家生成问题。为了解决这些问题，LLMs 需要掌握足够的编程知识 (如数学、数组操作和算法)。对于多语言任务，mT0 (13B 版本) 可能是一个很好的候选模型，它已经对具有多语言提示的多语言任务进行了微调。此外，PanGu-α[75] 在基于深度学习框架 MindSpore[118] 开发的零样本或少样本设置的中文下游任务中表现良好。请注意，PanGu-α[75] 包含多个版本的模型 (多达 200B 个参数)，而最大的公开版本有 13B 个参数。LLaMA (65B 版本)[57] 作为一种流行的 LLM，其包含的参数大约是其他模型的 5 倍，在指令遵循相关任务中表现出了优越的性能。由于 LLaMA 的开放性和有效性，它已经引起了研究界的极大关注，许多努力 [119-122] 都致力于对其不同的模型版本进行微调或持续预训练，以实现新的模型或工具。最近，Falcon[117] 作为另一个开源 LLM，也在开放基准测试中取得了非常出色的表现。它的特点是更仔细的数据清洗过程来准备预训练数据 (使用公开共享的数据集 RefinedWeb[123])。通常，这种规模的预训练模型需要数百甚至数千个 GPUs 或 TPUs。例如，GPT-NeoX-20B 使用 12 台超微服务器，每台服务器配备 8 个 NVIDIA A100-SXM4-40GB GPUs，而 LLaMA 在其原始版本中使用 2,048 个 A100-80G GPUs。为了准确估计所需的计算资源，建议使用 FLOPS (即每秒浮点数操作数)等度量所涉及的计算次数的指标[30]。

-  __具有数千亿参数的模型 Models with Hundreds of Billions of Parameters__。对于这个类别的模型，只有少数模型已经公开发布。例如，OPT[81]、OPT- IML[85]、BLOOM[69]、BLOOMZ[84] 的参数数量与 GPT-3 (175B 版本) 几乎相同，而 GLM[83] 和 Galactica[35] 的参数数量分别为 130B 和 120B。其中，OPT (175B 版本) 具有指令调优版本 OP-IML，专门用于开放共享，旨在使研究人员能够大规模地进行可重复的研究。对于跨语言泛化的研究，可以使用 BLOOM (176B版本) 和 BLOOMZ (176B版本) 作为基础模型，因为它们可以胜任多语言的语言建模任务。作为一个双语 LLM, GLM 还提供了一个流行的小型中文聊天模型 ChatGLM2-6B (ChatGLM-6B的更新版本)，该模型在效率和容量方面有许多改进 (例如 _量化 quantization_，_32K-长度上下文 32K-length context_，_快速推论率 fast inference rate_)。这种规模的模型通常需要数千个 GPU 或 TPU 来训练。例如，OPT (175B版本) 使用 992 个 A100-80GB GPU，而 GLM (130B版本) 使用 96 个 NVIDIA DGX-A100 (8x40G) GPU节点的集群。

-  __LLaMA 模型家族 LLaMA Model Family__。LLaMA 模型集合[57]是 Meta AI 在 2023 年 2 月引入的，包括四种大小 (7B、13B、30B 和 65B)。自发布以来，LLaMA 引起了学术界和工业界的广泛关注。LLaMA 模型在各种开放基准测试中取得了非常出色的性能，成为迄今为止最流行的开放语言模型。大量研究人员通过指令调优或持续预训练来扩展 LLaMA 模型。特别是，由于计算成本相对较低，指令微调 LLaMA 已成为开发定制或专用模型的主要方法。为了有效地适应非英语语言中的 LLaMA 模型，它通常需要扩展原始词汇表 (主要在英语语料库上训练)，或者使用目标语言中的指令或数据对其进行微调。在这些扩展模型中，Stanford Alpaca[124] 是第一个基于 LLaMA进 行微调的开放式指令跟随模型 (7B)。它是通过使用 text-davinci-003通过 _self-instruct_[125] 生成的 52K 指令遵循演示来训练的。被命名为 Alpaca- 52K 的指令数据和训练代码在随后的工作中被广泛采用，如 AlpacaLoRA[126] (使用 LoRA[127]复制 Stanford Alpaca)、Koala[128]和 BELLE[129]。此外，Vicuna [120]是另一种流行的 LLaMA 变体，通过从 ShareGPT 收集的用户共享对话进行训练。由于 LLaMA 模型族的优异性能和可用性，大量多模态模型都将其作为基础语言模型，以实现较强的语言理解和生成能力。与其他变体相比，Vicuna 在多模态语言模型中更受青睐，因此出现了各种流行的模型，包括 LLaVA[130]、MiniGPT- 4[131]、InstructBLIP[132] 和 PandaGPT[133]等。LLaMA 的发布极大地推动了 LLMs 的研究进展。为了总结对 LLaMA 的研究工作，我们在图4中给出了一个简短的进化图。

![[Pasted image 20230809145453.png]]

-  __LLMs 的公共 API Public API of LLMs__。API 不是直接使用模型副本，而是为普通用户提供了一种更方便的方式来使用 LLMs，而不需要在本地运行模型。作为使用 LLMs 的代表性接口，GPT 系列模型的 API [46,55,61,89] 已被学术界和工业界广泛使用17。OpenAI 为 GPT-3 系列中的模型提供了七种主要接口: ada、babbage、curie、davinci  (GPT-3系列中最强大的版本)、text-ada-001、text-babbage-001 和 text-curie-001。其中，前四个接口可以在 OpenAI 的主机服务器上进一步微调。其中，baggage、curie 和 davinci 分别对应于 GPT-3 (1B)、GPT-3 (6.7B) 和 GPT-3 (175B) 模型[55]。此外，还有两个与 Codex [89]相关的 APIs，称为 code-cushman-001 (Codex (12B)[89]的强大多语言版本)和 code- davinci -002。此外，GPT-3.5 系列包括一个基本型号代码 davinci-002 和 三个增强版本，即 text-davinci-002, text-davinci-003 和 GPT-3.5-turbo-0301。值得注意的是，gpt-3.5-turbo-0301 是调用 ChatGPT 的接口。最近，OpenAI 也为 GPT-4 发布了相应的 APIs，包括 gpt-4、gpt-4-0314、gpt-4-32k 和 gpt-4-32k-0314。总的来说，API 接口的选择取决于具体的应用场景和响应需求。详细的用法可以在他们的项目网站上找到。

## 3.2 常用语料库 Commonly Used Corpora

-  与早期的 PLMs 相比，由大量参数组成的 LLMs 需要更多的训练数据，这些数据涵盖了广泛的内容。为了满足这一需求，有越来越多的可访问的训练数据集已经发布用于研究。在本节中，我们将简要总结几种广泛使用的培训 LLMs 的语料库。根据其内容类型，我们将这些语料库分为六组: _书籍、CommonCrawl、Reddit链接、维基百科、代码和其他_。
	-  _书籍 Books_。BookCorpus[134] 是以前小规模模型 (例如 GPT[105] 和 GPT-2[26]) 中常用的数据集，由超过 11,000 本书组成，涵盖了广泛的主题和体裁 (例如小说和传记)。 另一个大规模的图书语料库是 _古登堡计划 Project Gutenberg_[135]，包括超过 70,000 本文学书籍，包括小说，散文，诗歌，戏剧，历史，科学，哲学和其他类型的公共领域的作品。它是目前最大的开源书库之一，被用于 MT-NLG[97] 和 LLaMA[57] 的训练。 对于GPT-3[55] 中使用的 Books1[55] 和 Books2[55]，它们比 BookCorpus 要大得多，但到目前为止还没有公开发布。
	-  _通用爬虫 CommonCrawl_ CommonCrawl[144] 是最大的开源 web 爬虫数据库之一，包含 PB 级的数据量，已被广泛用作现有 LLMs 的训练数据。由于整个数据集非常大，现有的研究主要是提取特定时间段内的网页子集。然而，由于 web 数据中普遍存在噪声和低质量的信息，因此在使用前需要对数据进行预处理。基于 CommonCrawl，现有工作中常用的过滤数据集有四种: C4[73]、CCStories[136]、CC-News[27]和RealNews[137]。Colossal Clean Crawled Corpus(C4)包括 5 个变体，即en (806G)， en.noclean (6T)、realnewslike (36G)、webtextlike (17G) 和 multilingual (38T)。en 版本已被用于 T5[73]、LaMDA[63]、Gopher[59]和 UL2[80]的预训练。多语种C4，也称为mC4，已在 mT5 中使用[74]。 CC-Stories (31G) 由 CommonCrawl 数据的一个子集组成，其中的内容以类似故事的方式生成。由于 CC-Stories 的原始来源现在不可用，我们在表2 中包含了一个复制版本 CC-Stories-R [145]。此外，从 CommonCrawl 中提取的两个新闻语料库REALNEWS (120G) 和 CC-News (76G) 也是常用的预训练数据。
	- _Reddit Links_。Reddit 是一个社交媒体平台，用户可以提交链接和文本帖子，其他人可以通过 “赞 upvotes” 或 “贬 downvotes” 对这些帖子进行投票。高赞的帖子通常被认为是有用的，可以用来创建高质量的数据集。WebText[26] 是一个著名的语料库，由来自 Reddit 的高支持率链接组成，但它不公开可用。作为替代品，有一个易于访问的开源替代方案，称为 OpenWebText[138]。从 Reddit 提取的另一个语料库是 PushShift.io[139]是一个实时更新的数据集，由 Reddit 自创建之日起的历史数据组成。Pushshift 不仅提供每月的数据转储，还提供有用的实用工具来支持用户搜索、汇总和对整个数据集进行初步调查。这使得用户很容易收集和处理 Reddit 数据。
	-  _Wikipedia_。维基百科[140]是一个在线百科全书，包含大量关于不同主题的高质量文章。这些文章大多以说明性的写作风格组成 (带有辅助参考文献)，涵盖了广泛的语言和领域。通常情况下，维基百科的纯英文过滤版本在大多数 LLMs 中被广泛使用 (例如GPT-3 [55]， LaMDA[63]和LLaMA[57])。维基百科有多种语言版本，因此它可以在多语言环境中使用。
	-  _Code_。为了收集代码数据，现有的工作主要是从互联网上抓取开源许可代码。两个主要来源是开放源代码许可下的公共代码库 (例如 GitHub) 和与代码相关的问答平台 (例如 StackOverflow)。Google 已经公开发布了 BigQuery 数据集[141]，其中包括大量开源许可的各种编程语言代码片段，作为具有代表性的代码数据集。CodeGen 已经使用 BIGQUERY [77]， BigQuery 数据集的一个子集，来训练CodeGen 的多语言版本 (CodeGen- multi)。
	-  _Others_。The Pile[142] 是一个大规模的、多样化的、开源的文本数据集，由来自多个来源的超过 800GB 的数据组成，包括书籍、网站、代码、科学论文和社交媒体平台。它由 22 个不同的高质量子集组成。Pile 数据集被广泛应用于不同参数尺度的模型中，如 GPT-J (6B)[146]、CodeGen (16B)[77]、Megatron-Turing NLG (530B)[97]。ROOTS[143] 由各种较小的数据集 (总共 1.61 TB的文本)组成，涵盖了 59 种不同的语言(包括自然语言和编程语言)，这些语言已被用于训练 BLOOM[69]。

-  在实践中，预训练 LLMs 通常需要混合不同的数据源(见图5)，而不是单一的语料库。因此，现有的研究通常将几个现成的数据集 (如 C4、OpenWebText 和 the Pile) 混合在一起，然后进行进一步的处理，得到预训练语料库。此外，为了训练适应特定应用的LLMs，从相关来源 (例如 Wikipedia 和 BigQuery) 中提取数据以丰富预训练数据中的相应信息也很重要。为了快速参考现有 LLMs 中使用的数据源，我们给出了三个具有代表性的 LLMs 的预训练语料库:
	-  GPT-3 (175B)[55]是在一个包含 300B 个 tokens 的混合数据集上训练的，包括CommonCrawl[144]、WebText2[55]、Books1[55]、Books2[55] 和 Wikipedia[140]。
	-  PaLM (540B)[56] 使用 780B tokens 的预训练数据集，这些 tokens 来自社交媒体对话、过滤的网页、书籍、Github、多语言维基百科和新闻。
	-  LLaMA[57] 从各种来源提取训练数据，包括 CommonCrawl, C4 [73]， Github, Wikipedia, books, ArXiv 和 StackExchange。LLaMA (6B) 和 LLaMA (13B) 的训练数据大小为 1.0T tokens，LLaMA (32B) 和 LLaMA (65B) 的训练数据大小为 1.4T tokens。

## 3.3 工具库资源 Library Resource

-  在这一部分中，我们简要介绍了一系列可供 LLMs 开发的工具库。
	-  _Transformers_[147] 是一个开源 Python 库，用于使用 Transformer 架构构建模型，由 Hugging Face 开发和维护。它有一个简单和用户友好的 API，使其易于使用和定制各种预训练的模型。它是一个强大的库，拥有大量活跃的用户和开发人员社区，他们定期更新和改进模型和算法。
	-  _DeepSpeed_[65] 是微软开发的深度学习优化库 (与 PyTorch 兼容)，已被用于训练许多 LLMs，如 MTNLG[97] 和 BLOOM[69]。它为分布式训练提供了各种优化技术的支持，例如内存优化 (ZeRO 技术、梯度检查点) 和 pipeline 并行性。
	-  _Megatron-LM_[66-68]是由 NVIDIA 开发的用于训练大规模语言模型的深度学习库。它还为分布式训练提供了丰富的优化技术，包括模型和数据并行、混合精度训练和 FlashAttention。这些优化技术可以大大提高训练效率和速度，实现高效的跨 GPUs 分布式训练。
	-  _JAX_[148]是 Google 开发的用于高性能机器学习算法的 Python 库，允许用户通过硬件加速 (例如 GPU 或 TPU) 轻松执行数组计算。它可以在各种设备上进行有效的计算，并且还支持一些功能，例如 _自动微分 automatic differentiation_ 和 _即时编译 just-in-time compilation_。
	-  _Colosal -AI_ [149]是由 HPC-AI Tech 开发的用于训练大规模 AI 模型的深度学习库。它是基于 PyTorch 实现的，支持丰富的并行训练策略集合。此外，它还可以使用 PatrickStar[150] 提出的方法优化异构内存管理。最近，一个名为 ColossalChat 的类似 ChatGPT 的模型[122]已经公开发布，有两个版本 ( 7B 和 13B)，该模型使用基于 LLaMA 的 Colossal-AI 开发[57]。
	-  _BMTrain_[151]是 OpenBMB 开发的用于分布式训练大规模参数模型的高效库，强调代码简单、低资源、高可用性。BMTrain 已经将几个常见的 LLMs (例如，Flan-T5[64] 和 GLM[83]) 合并到其 ModelCenter 中，开发人员可以直接使用这些模型。
	-  _FastMoE_[152] 是 MoE (即混合专家) 模型的专门训练库。它是基于 PyTorch 开发的，在设计中优先考虑效率和用户友好性。FastMoE 简化了将 Transformer 模型转换为 MoE 模型的过程，并在训练期间支持数据并行性和模型并行性。除了上述库资源外，现有的深度学习框架 (如PyTorch [153]， TensorFlow [154]， MXNet [155]， PaddlePaddle [156]， MindSpore[118]和OneFlow[157]) 也提供了对并行算法的支持，这些算法通常用于训练大规模模型。

---

# 4 预训练 Pre-Training

-   _预训练为 LLMs 的能力奠定了基础_。通过大规模语料库的预训练，LLMs 可以获得必要的语言理解和生成技能[55,56]。在这个过程中，预训练语料库的规模和质量对于 LLMs 获得强大的能力至关重要。此外，为了有效地预训练 LLMs，需要设计好 _模型架构 model architectures、加速方法 accelaration methods 和优化技术 optimization techniques_。接下来，我们首先在 4.1 节中讨论数据的收集和处理，然后在 4.2 节中介绍常用的模型架构，最后在4 .3 节中介绍稳定有效地优化 LLMs 的训练技术。

## 4.1 数据收集 Data Collection

-  与小规模语言模型相比，LLMs 对模型预训练的高质量数据有更强的需求，其模型能力很大程度上依赖于预训练语料库及其预处理方式。在这一部分中，我们讨论了预训练数据的收集和处理，包括数据来源，预处理方法，以及预训练数据如何影响 LLMs 性能的重要分析。

### 4.1.1 数据来源 Data Source

-  为了开发一个有能力的 LLMs，从各种数据源中收集大量的自然语言语料库是关键。现有的法学硕士主要利用各种公共文本数据集的混合物作为预训练语料库。图5显示了一些有代表性的 LLMs 的预训练数据源的分布。_预训练语料库的来源大致可以分为两类: 通用数据和专用数据_。一般数据，如网页、书籍和会话文本，由于其庞大、多样和可访问的性质，被大多数 LLMs 使用[55,56,81]，这可以增强 LLMs 的语言建模和泛化能力。鉴于 LLMs 所表现出的令人印象深刻的泛化能力，也有研究将其预训练语料库扩展到更专业的数据集，如多语言数据、科学数据和代码，赋予 LLMs 特定的任务解决能力[35,56,77]。接下来，我们将描述这两种类型的预训练数据源及其对 LLMs 的影响。关于常用语料库的详细介绍，可以参考第 3.2 节。

-  _通用文本数据 General Text Data_。正如我们在图 5 中所看到的，绝大多数 LLMs 采用通用的预训练数据，如网页、书籍和会话文本，这些数据提供了各种主题的富文本源。接下来，我们简要总结三种重要的一般数据。
	-  Webpages。由于互联网的普及，创建了各种类型的数据，这使得 LLMs 能够获得多样化的语言知识，并增强其泛化能力[26,73]。为了方便使用这些数据资源，在之前的工作中，大量的数据都是从web中抓取的，如 CommonCrawl[144]。然而，抓取的 web 数据往往既包含高质量的文本 (如Wikipedia)，也包含低质量的文本 (如垃圾邮件)，因此对网页进行过滤和处理以提高数据质量非常重要。
	-  Conversation text。会话数据可以增强 LLMs 的会话能力[81]，并有可能提高他们在一系列问答任务中的表现[56]。研究人员可以利用公共对话语料库的子集 (例如，PushShift.io Reddit语料库)[139,158] 或从在线社交媒体收集会话数据。由于在线会话数据通常涉及多个参与者之间的讨论，因此一种有效的处理方法是将会话转换为树状结构，其中话语与它所响应的话语相关联。这样，多方对话树就可以被分成多个子对话，这些子对话可以被收集到预训练语料库中。此外，一个潜在的风险是，将对话数据过度整合到 LLMs 中可能会产生副作用 [81]: 陈述性指令和直接疑问句被错误地视为对话的开始，从而导致指令的有效性下降。
	-  Books。与其他语料库相比，书籍提供了正式长文本的重要来源，这对 LLMs 学习语言知识、建立长期依赖关系模型以及生成叙事和连贯的文本有潜在的好处。为了获得开源图书数据，现有研究通常采用 Books3 和 Bookcorpus2 数据集，这两个数据集在 Pile 数据集中可以获得[142]。

-  _专业的文本数据 Specialized Text Data_。专业的数据集有助于提高 LLMs 在下游任务中的特定能力。接下来，我们介绍三种专门化数据。
	-  Multilingual text。除了目的语的文本外，整合多语语料库还可以提高语言理解和生成的多语能力。例如，BLOOM[69] 和 PaLM[56] 在其预训练语料库中分别策划了涵盖 46 种和 122 种语言的多语言数据。这些模型在多语言任务中表现出令人印象深刻的性能，例如翻译、多语言摘要和多语言问答，并且与在目标语言语料库上进行微调的 SOAT 模型实现了相当或更好的性能。
	-  Scientific text。科学出版物的增长见证了人类对科学的探索。为了增进对通用文本数据的科学理解。将科学语料库纳入模型预训练是有用的[35,159]。通过对大量科学文本进行预训练，LLMs 可以在科学和推理任务中取得令人印象深刻的表现[160]。为了构建科学语料库，现有的工作主要是收集 arXiv 论文、科学教科书、数学网页以及其他相关的科学资源。由于科学领域中数据的复杂性，例如数学符号和蛋白质序列，通常需要特定的标记化和预处理技术来将这些不同格式的数据转换为可以由语言模型处理的统一形式。
	-  Code。程序合成 Program synthesis 已经在研究界得到了广泛的研究[89,161 - 164]，特别是使用经过代码训练的 PLMs[146,165]。然而，对于这些 PLMs (例如GPT-J[146])来说，生成高质量和准确的程序仍然具有挑战性。最近的研究[89,164]发现，在一个庞大的代码语料库上训练 LLMs 可以大大提高合成程序的质量。生成的程序可以成功地通过专家设计的单元测试用例[89]或解决竞争性编程问题[98]。一般来说，两种类型的代码语料库通常用于预训练 LLMs。第一个来源是来自编程问答社区，如 Stack Exchange[166]。第二个来源来自公共软件存储库，如 GitHub[77,89,164]，其中收集代码数据 (包括注释和文档字符串) 以供使用。与自然语言文本相比，代码具有编程语言的格式，具有较长的依赖关系和准确的执行逻辑[167]。最近的一项研究[47]也推测，代码训练可能是复杂推理能力 (例如，思维链能力[33])的来源。此外，有研究表明，将推理任务格式化为代码可以帮助 LLMs 生成更准确的结果[167]。

### 4.1.2 Data Preprocessing

-  在收集了大量文本数据后，为了构建预训练语料库，必须对数据进行预处理，特别是去除有噪声的、冗余的、不相关的和潜在有毒的数据[56,59,168]，这些数据可能在很大程度上影响 LLMs 的能力和性能。在这一部分中，我们回顾了详细的数据预处理策略，以提高收集数据的质量[59,69,96]。一个典型的预处理 LLMs 预训练数据的流水线如图6所示。

![[Pasted image 20230810161254.png]]

-  __Quality Filter__。为了从收集的语料库中去除低质量的数据，现有的工作通常采用两种方法: (1) _基于分类器的方法 classifier-based_，(2) _基于启发式的方法 heuristic-based_。前一种方法基于高质量文本训练一个选择分类器，并利用它来识别和过滤掉低质量的数据。通常，这些方法[55,56,96]训练一个二元分类器，将精心策划的数据 (例如，维基百科页面) 作为正实例，将样本候选数据作为负实例，并预测衡量每个数据示例质量的分数。然而，一些研究[59,96]发现，基于分类器的方法可能会导致无意中删除方言、口语和社会语言中的高质量文本，这可能会导致预训练语料库中的偏见，并减少语料库的多样性。作为第二种方法，BLOOM[69] 和 Gopher[59] 等几项研究采用了基于启发式的方法，通过一组精心设计的规则来消除低质量的文本，可以总结如下:
	-  Language based filtering。如果 LLM 主要用于某些语言的任务，则可以过滤其他语言的文本。
	-  Metric based filtering。关于生成文本的评估指标，例如，困惑度 perlexity，可以用来检测和删除不自然的句子。
	-  Statistic based filtering。语料库的统计特征，如 标点分布、符号词比、句子长度等，可以用来衡量文本质量，过滤低质量的数据。
	-  Keyword based filtering。基于特定的关键字集，可以识别和删除文本中嘈杂或无用的元素，如 HTML 标记、超链接、样板和冒犯性词语。
-  __De-duplication__。已有研究[169]发现，语料库中重复的数据会降低语言模型的多样性，这可能导致训练过程变得不稳定，从而影响模型的性能。因此，有必要对预训练语料库进行去重复处理。特别地，重复数据删除可以在不同的粒度上执行，包括 _句子级 sentence-level、文档级 document-level和数据集级 dataset-level_ 重复数据删除。首先，应该删除包含重复单词和短语的低质量句子，因为它们可能会在语言建模中引入重复模式[170]。在文档层面，现有的研究大多依靠文档之间表面特征的重叠比例 (如单词和 n-grams 重叠) 来检测和去除内容相似的重复文档[57,59,69,171]。此外，为了避免数据集污染问题，通过从训练集中删除可能的重复文本，防止训练集和评估集之间的重叠也至关重要[56]。研究表明，三个层次的去重复对提高 LLMs 的训练是有用的[56,172]，在实践中应共同使用。
-  __Privacy Redaction__。大多数预训练文本数据都是从 web 来源获得的，包括用户生成的涉及敏感或个人信息的内容，这可能会增加隐私泄露的风险[173]。因此，有必要从预训练语料库中删除 _个人身份信息 (PII) personally identifiable information_ 。_一种直接而有效的方法是采用基于规则的方法 rule-based methods_，如 关键字识别 keyword spotting，来检测和删除个人身份信息，如姓名、地址和电话号码[143]。此外，研究人员还发现，LLMs 在隐私攻击下的脆弱性可归因于预训练语料库中存在重复的 PII 数据[174]。因此，重复数据删除也可以在一定程度上降低隐私风险。
- __Tokenization__。Tokenization 也是数据预处理的关键步骤。它旨在将原始文本分割成单个 tokens 序列，这些 tokens 随后用作 LLM 的输入。在传统的 NLP 研究中 (如条件随机场序列标注[175])，word-based tokenization 是主要的方法，它更符合人类的语言认知。然而，在某些语言中，基于词的分词可能会对相同的输入产生不同的分词结果 (例如中文分词)，产生包含许多低频词的巨大词库，并且还存在 “OOV” 问题。因此，一些神经网络模型使用 _字符 character_ 作为最小单元来派生单词表示 (例如，ELMo 中的 CNN 单词编码器[21])。近年来，_subword tokenizers_ 在基于 Transformer 的语言模型中得到了广泛的应用，主要包括 _BytePair Encoding tokenization、WordPiece tokenization 和 Unigram tokenization_。HuggingFace 在 tokenizer 上维护了一个优秀的在线 NLP 课程，并提供了运行示例，我们向初学者推荐本课程。接下来，我们简要介绍三种代表性的 tokenization 方法。
	-  _Byte-Pair Encoding (BPE) tokenization_。BPE 最初是在 1994 年作为一种通用的数据压缩算法提出的[176]，然后将其应用于 NLP 用于 tokenization[177]。它从一组基本符号 (例如，字母和边界字符) 开始，迭代地将语料库中两个连续的 tokens 频繁成对组合为新 toekn (称为 _合并 merge_)。_对于每次合并，选择标准是基于两个连续标记的共同出现频率: 将选择频率最高的对_。合并过程将继续进行，直到达到预定义的大小。此外，通过将 字节 bytes 作为合并的基本符号，Byte-level BPE 已被用于提高多语言语料库 (例如，包含 non-ASCII  字符的文本) 的 tokenization quality。使用这种 tokenization 方法的代表性语言模型包括 GPT-2、BART 和LLaMA。
	-  _WordPiece tokenization_。WordPiece 是谷歌内部的子词标记算法。它最初是由Google 在开发语音搜索系统时提出的[178]。随后，它于 2016 年被用于神经机器翻译系统[179]，并于 2018 年被用作 BERT 的 word tokenizer[23]。WordPiece与 BPE 有一个非常相似的想法，通过迭代地合并连续的 tokens，而对合并采取略微不同的选择标准。为了进行合并，它首先训练一个语言模型，并使用它对所有可能的配对进行评分。然后，在每次合并时，它选择导致训练数据可能性增加最多的对。由于 Google 没有发布 WordPiece 算法的官方实现，HuggingFace 在其在线 NLP 课程中给出了一个更直观的选择度量: 基于训练语料库，将共现次数除以对中两个 tokens 的出现次数的乘积来对一对进行评分。
	-  _Unigram tokenization_。与 BPE 和 WordPiece 不同，Unigram tokenization[180]从语料库中足够大的可能 substrings or subtokens 开始，迭代地删除当前词汇表中的 tokens，直到达到预期的词汇表大小。作为选择标准，它通过假设从当前词汇表中删除某些 token 来计算由此产生的训练语料库的可能性的增加。这一步是基于训练好的 unigram language model 进行的。对于 unigram language model 的估计，采用 _期望最大化(EM) expectation-maximization_ 算法: 在每次迭代中，我们首先基于旧的语言模型找到当前最优的 tokenization of words，然后重新估计 the probabilites of unigrams 来更新语言模型。在此过程中，使用 _动态规划算法 dynamic programming_ (即 Viterbi 算法) 来有效地找到给定语言模型的单词的最优分解方式。采用这 tokenization approach 的代表性模型包括 T5 和 mBART。
	- 尽管利用现有的 tokenization 是方便的 (例如，OPT[81] 和 GPT-3[55] 利用 GPT- 2[26] 的标记器)，使用专门为预训练语料库设计的 tokennization 可能是最好的[69]，特别是对于由不同领域，语言和格式组成的语料库。因此，最近的 LLMs 经常使用 SentencePiece 库[181] 来训练专门针对预训练语料库 customized tokenization，其中包括 byte-level BPE 和Unigram tokennization。值得注意的是，BPE  中的规范化技术，如 NFKC[182]，可能会降低 tokenization 性能[34,59,69]。在扩展现有的 LLMs (即，持续的预训练或指令微调) 时，我们还应该意识 customized tokenization 的潜在副作用。例如，LLaMA 基于一个主要由英语文本组成的预训练语料库来训练BPE tokenizer，派生的词汇可能在处理非英语数据时能力较差，例如，需要较长的推理延迟来生成中文文本。

### 4.1.3 Effect of Pre-training Data on LLMs

-  与小规模 PLMs 不同，由于对计算资源的巨大需求，LLMs 的预训练通常无法多次迭代。因此，在训练 LLMs 之前，构建一个准备充分的预训练语料库尤为重要。在这一部分中，我们讨论了预训练语料库的质量和分布如何潜在地影响 LLMs 的性能。

-  _Mixture of Sources_。如前所述，来自不同领域或场景的预训练数据具有不同的语言特征或语义知识。通过对来自不同来源的混合文本数据进行预训练，LLMs 可以获得广泛的知识范围，并可能表现出强大的泛化能力。因此，在混合不同的数据源时，建议尽可能多地包含高质量的数据源，并小心设置预训练数据的分布，因为这也可能影响 LLMs 在下游任务上的性能[59]。Gopher[59] 对数据分布进行消融实验，考察混合源对下游任务的影响。在 LAMBADA 数据集[183]上的实验结果表明，增加书籍数据的比例可以提高模型从文本中捕获长期依赖关系的能力，增加 C4 数据集[73]的比例可以提高 C4 验证数据集的性能[59]。然而，作为一个副作用，对某一领域的过多数据进行训练会影响 LLMs 在其他领域的泛化能力[35,59]。因此，建议研究人员仔细确定预训练语料库中不同领域数据的比例，以开发更符合其特定需求的 LLMs 。读者可以参考图5来比较不同 LLMs 的数据源。

-  _Amount of Pre-training Data_。为了预训练一个有效的 LLM，重要的是收集足够的高质量数据，满足 LLM 的数据量需求。已有研究发现，随着 LLM 中参数规模的增大，需要更多的数据来训练模型[34,57]: 在模型性能方面，数据规模也存在与模型规模相似的 scaling law。最近的一项研究表明，由于预训练数据不足，许多现有的 LLMs 存在 _次优训练问题 sub-optimal training_[34]。通过进行广泛的实验，它进一步证明了在给定的算力预算下，以相同的规模增加模型大小和数据大小可以导致 _算力效率更高的模型 computer-efficient model_ (即 Chinchilla model)。最近，LLaMA[57]表明，在更多的数据和更长的训练时间下，较小的模型也可以获得良好的性能。综上所述，我们建议研究人员在充分训练模型时，特别是在缩放模型参数时，应该更加关注高质量数据的数量。

-  _Quality of Pre-training Data_。现有的研究表明，在低质量的语料库上进行预训练，如有噪声的、有毒的和重复的数据，可能会损害模型的性能 [59,169,171,174]。为了开发一个性能良好的 LLM，同时考虑所收集的训练数据的数量和质量是至关重要的。最近的研究，如 T5[73]、GLaM[96] 和 Gopher[59]，研究了数据质量对下游任务性能的影响。通过比较在过滤和未过滤的语料库上训练的模型的性能，他们得出了相同的结论，即在清洗过的数据上预训练 LLMs 可以提高性能。更具体地说，数据的重复可能导致 “_双下降 double descent_” (指性能最初恶化，随后提高的现象)[169,184]，甚至使训练过程不堪重负[169]。此外，有研究表明，重复数据会降低 LLMs 从上下文中复制的能力，这可能进一步影响 LLMs 使用上下文学习的泛化能力 [169]。因此，如文献 [56,59,69] 所建议的，在预训练语料库上谨慎地加入预处理方法 (如 4.1.2 节所示)，以提高训练过程的稳定性，避免影响模型性能。

## 4.2 Architecture

>  回顾 LLMs 的架构设计，即主流架构 mainstream architecture，预训练目标 pre-training objective 和细节设置 detailed configuration

### 4.2.1 Mainstream Architectures

-  由于良好的并行性和能力， Transformer 架构成为了开发不同 LLMs 的事实上骨干，它使得上百千亿放大语言模型参数成为可能。总体来说，现有主流 LLMs 的架构粗略分为三类：__encoder-decode, 因果解码器 causal decoder 和 前置编码器 prefix decoder__

![[Pasted image 20230824092834.png]]

#### Encoder-decoder Architecture

-  原始 Transformer 模型是建立在 encoder-decoder architecture，其中包含两个 Transformer blocks 堆叠成 _编码器 encoder_ 和 _解码器 decoder_。 The encoder 包括 多层 multi-head self-attention layers 来编码输入序列，产生它的潜在表示，而 the decoder 在这些表示上执行 cross-attention ，并 _自回归 autoregressively_  生成目标序列。 Encoder-decoder PLMs (即 T5 和 BART) 在一系列 NLP 任务中展示了有效性。目前为止，只有一小部分 LLMs 基于 encoder-decoder 架构构建，只有 Flan-T5。我们在 4.2.4 节架构选择详细讨论。

#### Causal Decoder Architecture

-  Causal Decoder Architecture 结合了单向注意力任务，来保证每个输入 token 只关注以前的 tokens 和它自己。输入和输出 tokens 以相同方式 in the same fashion 通过解码器。作为表示语言模型， GPT-series 模型基于 causal-decoder architecture 开发。特别是 GPT-3 已经成功证明这种架构的有效性，同时展示出了 LLMs 的 _in-context learning_ 能力。GPT-1 和 GPT-2  并没有展现出类似 GPT-3 那样的优质能力，看起来规模在增强模型能力上起到了重要作用。目前为止，causal decoders 被现有 LLMs 广泛接受，如 OPT、BLOOM 和 Gopher。注意 causal decoder 和 prefix decoder 都属于 decoder-only 架构。

#### Prefix Decoder Architecture

-  The prefix decoder architecture (a.k.a, non-causal decoder) 修正了 causal decoders 的masking mechanism，使得对 _前缀 prefix_ tokens 使用双向注意力，并仅对生成的 tokens 使用单向注意力。如同 encoder-decoder， the prefix decoders 可以双向编码前缀序列和自回归预测一个一个输出的 tokens，而且在编码和解码使用相同参数。不用从头开始预训练，一种实践建议是继续训练 causal decoders 然后将它们转化为 prefix decoders 来加速收敛，如 PaLM 的衍生 U-PaLM。现有的代表性的基于 prefix decoders 的 LLMS 包括 GLM-130B 和 U-PaLM。

### 4.2.2 详细配置 Detailed Configuretion

-  自从 Transformer 的出现，为了增强它的训练稳定性、性能和计算效率，提出了各种改进方法。我们将讨论对 Transformer 四个主要部分的相应配置，包括 normalization，position embeddings，activation functions 和 attention and bias。

![[Pasted image 20230824110605.png]]

#### Normalization Methods

-  预训练 LLMs 最大的挑战就是 _训练的不稳定性 training instability_，normalization 是用来稳定神经网络训练的广泛接受策略。原始 Transformer 使用 LayerNorm。最近，几个更先进的 normalization 技术被提出来取代 LayerNorm，如 RMSNorm 和 DeepNorm。

##### _层归一化 LayerNorm_

-  最初研究中， BatchNorm 是一个普遍使用的归一化方法。然后，它难以处理变长的序列数据和小批量 small-batch 数据。因此，LayerNorm 被用来实现 逐层 layerwise normalization。计算每层所有激活的均值和方差以 recenter 和 重新缩放激活 re-scaling the activations

##### _RMSNorm_

-  为了提升 LayerNorm 的训练速度，提出了 RMSNorm，它使用了 _均方根差 root mean square_ 以 re-scaling 激活，而不是均值和方差。相关研究已证实它在 Transformer 的训练速度和性能上优势巨大。使用 RMSNorm 的代表性模型包括 Gopher 和 Chinchilla。

  ##### _DeepNorm_
  
  -  由 Microsoft 提出用来稳定深层 Transformer 训练。使用 DeepNorm 作为 残差连接 residual connections， Transformers 可以被伸展到 1000 层，并表现出稳定性和优异的性能。GLM-130B 在使用它。

#### 归一化位置 Normalization Position

-  除了 normalization method 之外，_归一化位置 normalization position_ 也是 LLMs 中一个重要角色。总体有三种 normalization position 的选择，即 _post-LN, pre-LN 和 sandwich-LN_。

 ##### _Post-LN_
 
 -  Post-LN 在原始 Transformer 中使用，就放在 residual blocks 之间。然而，由于输出层附件的梯度过大，训练有 post-LN 的 Transformers 是不稳定的。因此，除非和其它策略联合使用 (如 GLM-130B 中 combing post-LN with pre-LN)，否则现有 LLMs 极少应用。

##### _Pre-LN_

-  与 post-LN 不同，pre-LN 被应用于每个 sub-layer 之前，有一个额外的 LN 被放在最后预测输出之前。与 post-LN 相比，有 pre-LN 的 Transformers 训练更稳定。然后，它比有 post-LN 的变型表现更差。尽管性能下降，大多数 LLMs 由于更看重训练稳定性而采用了 pre-LN。然后有个特例，当 GLM 训练模型超过 100B 参数时，pre-LN 表现的很不稳定

##### _Sandwich-LN_

-  以 pre-LN 为基础，Sandwich-LN 在残差连接之前增加其它 LN 以避免在 Transformer 层输出出现值爆炸。然而也发现 Sandwich-LN 有时会在训练会失去稳定性，导致训练崩溃。

#### 激活函数 Activation Functions 

-  为了好的性能表现，激活函数也需要被正确放置在前馈神经网络中。现有的 LLMs 中，广泛使用 GeLU 激活。特别是近期的 LLMs (如 PaLM 和 LaMDA)，利用了 GLU 激活的变种，特别是 SwiGLU 和 GeGLU 变种，它们在实践中的效果最好。然后，与 GeLU 相比，它两在前馈神经网络需要更多参数 (约 50%) 。

#### Position Embeddings

-  由于 self-attention 模块在 Transformer 中是 permutation equivariant，_position embeddings PE_ 用于给建模序列 inject 绝对或相对位置信息。

##### _Absolute position embedding_

-  在原始 Transformer 中使用了绝对位置嵌入。在 encoder 和 decoder 最底层，绝对位置嵌入被加到了输入嵌入 input embeddings。在原始 Transformer 中提出了两种绝对位置嵌入的变种，即 sinusoidal 和 learned position embeddings，后者常见于现在的预训练语言模型。
	
##### _Relative position embedding_

-  与绝对不同，相对位置嵌入是根据 keys 和 queries 之间的 offsets 生成的。Transformer-XL 中使用了它的一个变种。对 keys 和 queries 之间的注意力分数的计算进行了修改，引入了对应相对位置的learnable embeddings。T5 更多简化了相对位置嵌入，随后被 Gopher 采纳。特别是，它在 attention scores 加入了可学习的标量这些标量可基于 query 和 key 之间的位置来计算。与 绝对 PE 相比，带 relative PE 的 Transformers 能够泛化到比训练序列更长的序列，即 _外推法 extrapolation_。

##### _旋转位置编码 Rotary Position Embedding_

-  RoPE 根据每个 token 的绝对位置设置专有的旋转矩阵。使用相对位置信息就可以计算 keys 和 queries 之间的 scores。正式由于优秀的性能和长期衰减特性 long-term decay property， RoPE 被广泛应用于最新的 LLMs，如 PaLM 和 LLaMA。以 RoPE 为基础， xPos 进一步提升了 Transformer 的 _位移不变性 translation invariance_ 和 _length extrapolation_。在 _旋转度矩阵 rotation degree vector_ 的每一个维度，xPos 增加一个特殊 _指数衰减 exponential decay_，当旋转度增大时，指数衰减变小。这样可以减轻训练时由于距离增大而造成的不稳定现象。

##### _ALiBi_

-  ALiBi 被用于提升 Transformer 的外推。与相对位置嵌入类似，它会根据 keys 和 queries 的距离对 attention scores 有一个 penalty 的偏差。与 T5 的相对位置嵌入方法不同，ALiBi 中的 penalty scores 是不用任何训练参数就能预定义的。经验结果标明 ALiBi 在比训练时更长的序列上有更好的外推表现，比其他流行的位置嵌入方法如 sinusoidal PE, RoPE, 和 T5 bias 都要好。另外，也标明 ALiBi 可以提升 BLOOM 中的训练稳定性。

#### Attention and Bias

-  注意力机制是 Transformer 中最重要的组成部分。它允许 tokens 跨序列与其他 tokens 互动并计算出输入和输出序列的表示。

##### _Full attention_

- 原始 Transformer，注意力机制以成对的方式进行，考虑了序列中所有 token 对之间的关系。它采用 _缩放点积注意力 scaled dot-product attention_，其中隐藏状态被映射成 queries，keys 和 values。Transformer 采用 _多头注意力 multi-head attention_  而不是单注意力，在不同头上使用不同投射来投影 queries，keys 和 values。

##### _Sparse attention_

-  全注意力的一个重要挑战的二次方计算复杂度，这在处理长序列时是一个负担。因此为了减轻复杂度提出了各种高效 Transformer 变种。例如，_局部带状稀疏注意力 locally banded sparse attention_ (即 GPT-3 中的 _Factorized Attention_。每个 query 只注意基于位置的一个 tokens 的子集，而不是整个序列 )。

##### _Multi-query attention_

-  attention 变种，不同头在 keys 和 values上共享相同线性转换。这能显著节省计算成本，只会在模型质量上有一个微小的牺牲。带有 multi-query attention 的代表性模型包括 PaLM 和 StarCoder。

##### _FlashAttention_

-  与现有的牺牲模型质量来提升计算效率的近似注意力方法不同，FlashAttention 提出从 IO敏感角度来优化 GPUs 上注意力模块的速度和内存消耗。现在 GPUs 存在着不同级别的内存，如 SRAM 有着快速 IO 以及 HBM 有着相对缓慢的 IO。FlashAttention 将输入组织成 _块 blocks_ 而且提出必要的重新计算，都是为了更高效地使用 SRAM 快速内存。FlashAttention 被作为 CUDA 中一个 fused kernel 执行，被集成到了 PyTorch，DeepSpeed，和 Megatron-LM。

-  综上，我们总结了现有文献中的详细配置。为了更好地泛化和训练稳定性，建议在 layer normalization 选择 pre RMSNorm，激活函数选择 SwiGLUster或 GeGLU。另外， LM 可以不在 embedding 层后立即使用，因为有可能造成性能下降。至于位置嵌入， RoPE 或 ALibi 都是一个好的选择，它们在长序列上都表现优秀。

### 4.2.3 预训练任务 Pre-Training Tasks

-  在从大规模语料库的一般知识到大量模型参数的编码过程中，预训练起到了关键作用。至于训练 LLMs，有两种常用的预训练任务，即 _语言建模 language modeling_ 和 _降噪自编码 denoising autoencding_。

 #### __语言建模 Language Modeling__
 
 -  语言建模任务 LM_ 是最常见用于预训练decoder-only LLMs，如 GPT3 和 PaLM。有一个 tokens 序列 $\boldsymbol {\text x} = {x_1, \cdots, x_n}$ ， LM 任务就是根据的序列在此之前的 tokens 自回归预测目标 tokens $x_i$。一般的训练目标是最大化似然：
	-  $\mathcal L_{LM}(\boldsymbol {\text x}) = \sum\limits_{i=1}^n \log P(x_i|x_{<i})$
-  大多语言任务可以转换成根据输入做预测，所以 decoder-only LLMs 有潜力隐式学习如何以统一 LM 方式完成这些任务。一些研究标明 decoder-only LLMs 通过自回归预测以后的 tokens 天生可以被转换成某些任务，而不需要微调。LM 的一个重要变种就是 _prefix language modeling_ 任务，它被设计成带有 prefix decoder 架构的 PLMs。在计算前缀语言建模的损失时，不会使用随机选择前缀的 tokens。对于在预训练阶段见到的相同数量 tokens，前缀语言建模的性能略差于语言建模，这是因为序列中用于模型预训练的 tokens 较少。

#### 降噪自编码 Denoising Autoencoding

-  除了传统的 LM， _降噪自编码任务 DAE_ 也广泛用于预训练语言模型。DAE 任务的 输入 $\text x_{\textbackslash \hat {\text  x}}$ 是带有随机替换跨度的损坏文本。然后，训练语言模型去复原被替换的 tokens $\hat {\text x}$。DAE 的训练目标被记作：
	-  $\mathcal L_{DAE}(\boldsymbol {\text x}) = \log P(\hat {\text x} | \boldsymbol {\text x}_{\textbackslash \hat {\text x}})$

-  然后，DAE 任务执行时比 LM 任务复杂的多。因此，它并没有被广泛用于大语言模型预训练。现有 LLMs 使用 DAE 作为预训练目标的有 T5 和 GLM-130B。这些模型主要自回归训练去复原被替代的跨度。

#### Mixture-of-Denoisers

-  MoD 也被称作 UL2 loss，是预训练语言模型的一个统一目标。MoD 把 LM 和 DAE 目标当作不同种类的降噪任务，即 S-denoiser (LM)，R-denoiser (DAE，short psan and low corruption), 以及 X-denoiser (DAE, long span or high corruption)。这三种降噪任务中，S-denoiser 近似于常规 LM 目标，然后 R-denoiser 和 X-denoiser 近似于 DAE 目标，但它们的 spans 长度和 corrupted text 的比例也各不相同。对于开头带有不同特殊 tokens (即 { [R], [S], [X] } ) 的输入句子，模型会使用相应的降噪器去优化。MoD 已应用于最新的 PaLM 2 模型。

### 4.2.4 总结和探讨

-  对于 LLMs ，体系结构和预训练任务的选择可能产生不同的归纳偏差，从而导致不同模型能力。在这部分，我们讨论有关 LLM 体系结构的两个开放性问题。

#### Architecture Choice

-  预训练语言模型的早期文献，有很多讨论关于不同体系结构的影响。然后，大多数 LLMs 是基于 causal decoder architecture 的，目前还缺乏对其相对于其它替代方案的优势的理论分析。后面我们简要就这个问题总结现在的讨论

-  通过 LM 目标的预训练，因果编码器体系架构可以实现优势的零样本和少样本泛化能力。目前研究标明，不做多任务微调的话，因果编码器比其它都有更好的零样本标新。GPT-3 的成功证明了大型因果编码模型可以称为一个好的少样本学习者。此外，_指令微调 instruction tuning_ 和 _对齐微调 alignment tuning_ 被证明可以进一步增强大型因果编码模型的能力。

-  Scaling law 在因果编码器中被广泛发现。通过伸缩模型大小，数据集大小和总算力，因果编码器的性能有着很可观地提升。因此，伸缩是提升因果编码器模型能力的一个重要策略。然而，还是缺少更多详细的关于 encoder-decoder 模型的研究，需要更努力的去研究大规模上的 encoder-decoder 模型性能。

-  需要更多关于体系架构和预训练目标的讨论来分析体系架构和预训练任务的选择是如何影响 LLMs 的能力，尤其是 encoder-decoder 体系架构。除了主要的体系架构外，LLM 的详细配置也值得注意。

#### Long Context

-  基于 Transformer 的语言模型的主要缺点之一是上下文长度有限，由于涉及时间和内存的二次方计算成本。同时，对于 LLM 应用又有上下文窗口长度的增长需求，如 PDF 处理和写作。ChatGPT 最近发布一个更新，将上下文窗口大小从 4K tokens 提升到了 16K。此外， GPT-4 退出了带有 32K tokens 的上下文窗口的变体。洗下面，我们将会谈论支持 LLMs 长上下文建模的两个重要因素。

##### Extrapolation

-  现实世界应用，很可能 LLMs 需要处理比训练预料库长的多的长输入文本。LLMs 能够编码长文本的能力被叫作 _extrapolation capability_。很多位置嵌入方法，如 RoPE 和 T5 bias，已经实证携带某些外推能力。特别是带有 ALiBi 的语言模型，在比训练长十倍的序列上能够保持相对稳定困惑度。也有像 xPos 通过改进旋转矩阵设计来增强 RoPE 的外推能力。

##### Efficiency

-  为了减轻 attention modules 的二次方计算成本，一些研究设计了高效注意力计算方法，可以是内存消耗的规模近乎于线性，例如稀疏或线性注意力。除了算法改进，另一项重要工作——FlashAttention，从 system-level 角度提升了效率 (即 GPU memory IO efficiency)。在相同算力预算情况下，能够用更长的上下文窗口来训练 LLMs。一些研究还旨在为语言建模设计 Transformers 以外新的架构，包括参数化状态控件模型 (如 S4, GSS 和 H3)，像 RWKV 等递归机制的堆叠线性注意力模块

---

## 4.3 Model Training

### 4.3.1 优化设置 Optimization Setting

-  对于 LLMs 的参数优化，我们提供了 batch training，learning rate， optimizer 和 training stability 常用设置。

#### 批量训练 Batch Training

-  对于语言模型预训练，现在的工作普遍把 _训练样本 batch size_ 设置为一个大的数字 (如 2048 examples 或 4M tokens) 以提高训练稳定度和 _吞吐量 throughput_。 对于像 GPT-3 和 PaLM 这样的 LLMs 。有一种新的策略可以在训练过程中动态提升 batch size，最终达到一个百万规模。确切的说，GPT-3 的 batch size 从 32K 逐步涨到 3.2M tokens。经验表明 batch size 的动态计划调整可以有效地稳定 LLMs 的训练过程。

#### 学习率 Learning Rate

-  现有的 LLMs 通常在预训练使用一个类似学习率计划，包括 warm-up 和 decay strategies。在最初训练步骤的 0.1% 到 0.5%，一个线性的 warm-up schedule 被用于逐渐增长学习率知道最大值，范围大致从 5 x 10e(-5) 到 1 x 10e(-4) (比如 GPT-3 是 6 x 10e(-5))。然后，一个 cosine decay strategy 被用于随后的步骤，逐渐减小学习到最大值的 10%，直到训练损失收敛。

#### 优化器 Optimizer

-  _Adam optimizer_ 和 _AdamW optimizer_ 被广泛集成在 LLMs 训练 (如 GPT-3)，它是基于一阶梯度优化的低阶矩的自适应估计。它的超参数设置如下： $\beta_1 = 0.9, \beta_2 = 0.95$ 和 $\epsilon = 10^{-8}$ 。同时 _Adafactor optimizer_ 也被集成到 LLMs 训练中 (如 PaLM 和 T5)，它是 _Adam optimizer_ 的便携性，专用于在训练阶段消费 GPU 内存。它的超参数被设置为 $\beta_1 = 0.9, \beta_2 = 1.0 - k^{-0.8}$, $k$ 记作训练步骤的次数。

#### 稳定训练 Stabilizing the Training

-  LLMs 的预训练阶段，很容易因为训练不稳定问题导致模型崩溃。为了解决这个问题，广泛使用 weight decay 和 gradient clipping，其中现有研究普遍把 gradient clipping 阈值设置为 1.0、weight decay rate 设置为 0.1。然而，随着 LLMs 的伸缩，训练损失的尖峰很容易发生，导致了训练不稳定。为了减轻这个问题， PaLm 和 OPT 使用了一种简单策略，即在尖峰出现之前，保存一个 checkpoint，然后跳过可能引起问题的数据，重新开始一个训练流程。GLM 更进一步发现通常导致尖峰的是 embedding 层的不寻常的梯度，并提议收缩 embedding 层的梯度。

### 4.3.2 可扩展的训练技术 Scalable Training Techniques

-  随着模型的数据大小的增长，在有限的算力资源下，进行有效地 LLMs 训练变成了一个挑战。特别是有两个主要技术难题待解决，即 increasing training throughput 和 loading larger models into GPU memory。本节，我们将检视现在解决以上两个挑战的方法，即 3D parallelism，ZeRO 和 混合精度训练 mixed precision training，以及给出如何在训练中使用它们的几条建议。

#### 3D Parallelism

-  它实际上是三种常用的平行训练技巧的集合，即 data parallelism, pipeline parallelism 和 tensor parallelism。

##### Data parallelism

-  它是提升训练吞吐量的基本方法之一。它在多个 GPUs 上赋值模型参数和优化器状态，然后将整个训练语料库分布到这些 GPUs。通过这样，每个 GPU 只需处理分配给它的数据，并进行正向和反向传播，获得梯度。不同 GPUs 计算后的梯度将会聚合来获得整个 batch 的梯度，用来更新所有 GPUs 中的模型。由于在各个 GPUs 上独立执行梯度计算，data parallelism 机制是高度可扩展的，可以通过增加 GPUs 数量的方式提高训练吞吐量。进一步说，这种技术实施起来很简单，现在很多流行的深度学习库已经实行 data paralleism，如 TensorFlow 和 PyTorch。

##### Pipeline parallelism

-  致力于将 LLM 的不同层分布到多个 GPUs 上。考虑一个 Transformer 模型，pipeline parallelism 在同一个 GPU 上加载连续层，为了减少计算 GPUs 之间隐藏层或梯度的传输成本。然而，实施一个朴素的 pipeline parallelism 也许会导致低 GPU 利用率，因为每个 GPU 都得等待上一个完成计算，这导致 bubbles overhead 的不必要开支。为了在 pipeline parallelism 中减少这种开销，GPipe 和 PipeDream 提出多批量数据的填充技术和异步梯度更新来提升管道效率。

##### Tensor parallelism

-  这是一种常见技术，旨在于分解 LLM 用来 multi-GPU 加载。不像 pipeline parallelism， tensor parallelism 关注与分解 LLM 的 _张量 tensors_ (即 _参数矩阵 parameter matices_ )。对于 LLM 中的 一个 _矩阵乘法运算 matrix multipliation operation_ $Y = XA$，参数矩阵 $A$ 可以被分解为两个子矩阵 $A_1$  $A_2$，按列，可以被表示为 $Y = [XA_1, XA_2]$。通过将矩阵 $A_1$ 和 $A_2$ 放到不同 GPUs 上，矩阵乘法运算会两个 GPUs 上并行调用，最终结果可以通过 跨GPU 通信合并两个 GPUs 的输出来获得。当前，张量并行已经被很多开源库支持，如 Megatron-LM，且可以被扩展到高维张量。Colossal-AI 已经为高维张量实施张量并行，并且专为序列数据提出序列并行，这样可以进一步分解 Transformer 模型中的注意力操作。

#### ZeRO 

-  ZeRO 技术由 DeepSpeed 库提出，关注于 data parallelism 的内存 redundancy 冗余问题。正如前文所提，data parallelism 要求每个 GPU 保存一个 LLM 的相同副本，包含模型参数、模型梯度和优化器参数。Whereas 然而，上面的数据不是所有都需要在每个 GPU 上都保存，这会导致内存冗余问题。为了解决， ZeRO 技术致力于在每个 GPU 上保存一小部分数据，其它剩余数据有需要时从其它 GPUs 获得。ZeRO 特别推出了三种解决方案，根据三部分数据，即 optimizer state partitioning、gradient partitioning和parameter partitioning， 是怎样保存的。经验结果显示头两个解决方案并没有提升通信开销 overhead，第三种方案提升大约 50% 通信开销，但是与 GPUs 数量成正比节省内存。PyTorch 实施了一个类似 ZeRO 的技术，叫作 FSDP。

#### 混合精度训练 Mixed Precision Training

-  在以往的 PLMs (如 BERT)， 32 位浮点数 (FP32)，已经被预训练主要使用。最近几年，为了预训练超大语言模型，一些研究开始利用 16-bit 浮点数 (FP16)，它会减少内存使用和通信开销。此外，最流行的 NVIDIA GPUs (如 A100) 已经是两倍于 FP32 的 FP16 计算单元，FP16 的计算效率能被进一步提升。然而，有工作发现 FP16 会导致计算精确度的损失，这会影响最后模型性能。为了缓解，一种叫作 _Brain Floating Point BF16_ 的替代方法被用于训练，相比 FP16，会分配更多的指数位 exponent bits 和更少的有效位 significant bits。对于预训练，BF16 普遍在表示精确度上比 EP16 好。

## Overall Training Suggestion

-  实践中，以上训练技巧，特别是 3D parallelism，经常联合使用来提升训练吞吐量和大模型加载。例如，研究人员结合 incorporated 8-way data parallelism，4-way tensor parallelism 和 12-way pipeline parallelism，在 384 个 A100 上训练 BLLOM。现在 像 DeepSpeed、Colossal-AI 和 Alpa 这样的开源库已经能够很好的支持3种并行训练方法。为了减少内存冗余， ZeRO、FSDP和激活重新计算 activation recomputation 技术，同样用于训练 LLMs，都已被集成如 DeepSpeed、PyTorch 和 Megatron-LM。此外，混合精度训练如 BF16 也促使训练效率提升和 GPU 内存使用的减少，然而这需要硬件支持 (如 A100)。因为大模型训练是 _一个耗时的过程 a time-intensive process_，早期如果能预测模型性能和发现不正常现象那就太好了。为了达到这个目的，GPT-4 最近推出了一个新的机制—— 在深度学习堆叠上的 _predictable scaling_  ，使用一个小模型来预测大模型的性能，这对开发 LLMs 相当游泳。实践中，人们可以进一步利用主流深度学习框架的支持训练技术。例如，PyTorch 支持数据并行训练算法 _FSDP (即 fully shared data parallel)_ ，如果需要的话，它允许将部分训练计算卸载到 CPUs。

---

# 5 大语言模型的自适应调整 ADAPTATION OF LLMs

-  在预训练后，LLMs 能够获得解决各种任务的一般能力。然后，越来越多的研究表明 LLM 的能力可以对特定任务做进一步自适应调整。在本小节，我们介绍两种主要方法来自适应调整预训练 LLMs，即 __指令微调 instruction tuning__ 和 __对齐微调 alignment tuning__。前者主要致力于增强 (或解锁) LLMs 的能力，而后者致力于将 LLMs 的行为与人类价值观或偏好对齐。我们还会讨论在资源有限下 __为了模型自适应的有效微调__ 和 __量化 quantization__

## 5.1 Instruction Tuning

-  基本上，指令微调是在自然语言形式的格式化实例集上对预训练 LLMs 进行微调的方法。为了实现指令微调，首先需要收集或构建指令格式化的实例。然后，我们通过自监督学习方法 (如，training with the sequence-to-sequence loss) 使用上述格式化后的实例微调 LLMs。指令微调后，即使是多语言设置，LLMs 都可以在未见的任务上表现出泛化的卓越能力。

-  一项最近研究提出指令微调的一项系统研究检视。与其相比，我们主要关注指令微调在 LLMs 上的作用，以及提供关于实例搜集和微调的详细指导和策略。此外，我们还会讨论如何使用指令调优来满足用户的真正需求，这已经广泛用于现有的 LLMs，如 InstructGPT 和 GPT-4。

### 5.1.1 格式化实例的构建 Formatted Instance Construction

-  总体来说，一个 instruction-formatted 实例包含：一个任务描述 (叫做 _instruction_ )，一个可选的输入、相对应的输出，和一些示例(可选)。作为重要的公共资源，现有的研究已经发布了大量自然语言格式的标注数据。下一步，我们将介绍三种主要的构建格式化实例的方法。

![[Pasted image 20230901134646.png]]

#### Formatting Task Datasets

-  在提出 instcuction tuning 之前，若干早期研究从各种各样的任务 (如 text summarization、text classification and translation) 收集实例以创造监督多任务训练数据集。作为指令微调实例的主要是剧院，很容易用自然语言任务描述来格式化这些多任务训练数据集。最近有工作使用 human-written 任务描述来增强标注数据集，用来通过解释任务目标来指导 LLMs 去理解这些任务。例如：一个任务描述 "Please answer this question" 被添加到 question-answering 任务的每个示例里。在指令微调后，LLMs 可以根据任务描述对未见的任务有很好的泛化能力。特别是它能够展示出指令在 LLMs 任务泛化能力中有多重要：如果在用于指令微调模型的标注数据集上去除任务描述，模型性能会有一个夸张的下落。为了为指令微调提供更好的标注示例，PromptSource —— 一个众包平台，被提出用来有效地制造、分享和认证不同数据集的任务描述。为了丰富训练实例，很多研究也试着使用专门设计的任务描述和反转现有实例的输入-输出对，以便进行指令调优。例如，有一个问答对，我们创造一个实例，根据回答来预测问题 (e.g. "Please generate a question based on the answer")。

#### Formatting Daily Chat Data

-  尽管已经有一大批带有指令的训练实例，他们主要来自于公开 NLP 数据集，所以有的缺少指令多样性或是不满足真实人类需求。为了克服这个问题，InstructGPT 提出使用真实用户提交到 OpenAI API 的查询 queries 作为任务描述。用户查询是用自然语言描述的，特别适合引出 eliciting LLMs 的指令跟随能力。此外，为了丰富任务多样性，人工标注员被要求为真实生活任务撰写 compose 指令，包括 open-ended genetation，open question answering，brainstorming, and chatting。然后他们让一组其他的标注员直接回答这些指令作为输出。最后，他们把一个指令 (即 收集的用户查询) 和预期的输出 (即 人类编写的回答)配对作为一个训练实例。_Note that 请注意_ InstructGPT 也使用这些真实世界任务格式化成自然语言，用来做对齐微调。因此，GPT-4 设计了潜在高风险指示，通过 _supervised fine-tuning SFT_ 指导模型拒绝这些指令，都是为了安全考虑。最近，研究人员也在手机用户的聊天请求作为输入数据，然后使用 ChatGPT 或 GPT-4 来回应这些请求作为输出数据。典型就是 ShareGPT 的会话数据。

#### 格式化合成的数据 Formatting Synthetic data

-  为了减少人工注释或手动收集的负担，已经提出了集中半自动化方法，通过将现有实例喂给 LLMs 来合成不同的任务描述和实例以构造实例。 Self-insturct 方法只需要大约 100 个实例作为初始任务池。然后，他们随机从池中挑选一些实例作为示范和 提示一个 LLM 去生成新的指令和相应的输入-输出对。在经过质量和多样性过滤后，新生成的实例将会加入到任务池。因此，合成方法是一种有效且经济的方法，专用于为 LLMs 生成大规模指令数据。


#### Key Factors for Instance Construction

-  指令实例的质量关乎着模型的性能。这里，讨论一些基本的实例构建的因素。

##### Scaling the instuctions

-  已被广泛证明扩大任务数量可以极大地增强 LLMs 的泛化能力。随着任务数量的增长，模型性能初始展现一种连续增长模式，然而当达到一定水平就变得微不足道。一个合理推测是，一定数量的代表性任务就能够提供相对足够的知识，增加更多任务可能不会带来额外收益。同理，从长度、结构及创造力等方面增强任务描述的多样性也是有好处的。对于每个任务的实例数，现在发现，少量实例通常会使模型的泛化性能 _饱和 saturate_。然而，将某些任务的实例数增加到一个很大的数字很可能会导致过拟合问题和削弱模型性能。

##### 格式化设计 Formatting design

-  自然语言格式的设计作为一个重要因素，会高度影响 LLMs 的泛化性能。具体地说，我们可以在现有数据集的输入-输出对上加入任务描述和可选演示，其中任务描述对于 LLMs 理解任务很重要。更进一步说，通过使用适当数量的示例作为演示，它可以导致实质性的改进，这也减轻了模型对指令工程的敏感性。然而，将其它组件 (如，要避免的事情、原因和建议) 纳入指令可能对 LLMs 的性能产生微不足道甚至不利的影响。为了引出 LLMs 的 _step-by-step reasoning ability_ ，一些研究工作提出了推理数据集要包含 _chain-og-thought CoT_ 示例，比如算术推理。有研究显示使用 CoT 和 non-CoT 示例来微调 LLMs 可以在各种推理任务中获得良好的性能，包括那些需要 multi-hop 推理能力 (如，常识问答和岁数推理) 以及那些不需要这种推理方式的任务 (如，情感分析和抽取式问答)。

#### 小结

-  总而言之，自从 InstructGPT 表现出优越性能以及 Alpaca 集成了比 Flan-series LLMs 更少但更多样的指令 (实例)，看起来指令的多样性和质量比数量更重要。而且，邀请标注者创造人类需求的任务比使用特定数据集任务更有用。然而仍缺少一般的指导用来注释人类需求的实例和使任务构成具有某种启发式 heuristic。为了减少人的努力，我们可以重复使用现有的格式化的数据集或是使用现有的 LLMs 自动化构造指令。

### 5.1.2 Instruction Tuning Strategies

-  与预训练不同，instruction tuning 可能更有效率，因为训练时使用了数量适中的实例。由于 instruction tuning 可以被认为是一个有监督训练流程，它的优化与预训练有以下不同：training objective (i.e. sequence-to-sequence loss) 和 optimization configuration (如，小批量样本和学习率)，它要求实践中特别关注。在这些优化配置之外，instruction tuning 还有两个重点需考虑:

#### Balancing the Data Distribution

-  由于 instruction tuning 牵扯到一系列不同任务的混合，在微调时平衡不同任务的比例就很重要。常见的方法有 _examples-proportional mixing_ 策略，即将所有数据集合并，从混合数据集中平等地采样每个实例。根据最近发现更进一步，增加高质量集合 (如，FLAN 和 P3) 的采样比例通常可以提升性能。再进一步的话，常见的是设置一个 _maximum cap_ 来控制 instruction tuning 期间一个数据集能包含的最大示例数，用来防止大数据集压倒整个分布。实践中，最大上限通常根据不同数据集设置为几千或几万。

#### Combining Instruction Tuning and Pre-Training

-  为了使微调过程更有效、更稳定，OPT-IML 包含了 instruction tuning 期间的预训练数据，它可以作为模型微调的正则化项。一些研究视图从一开始就使用混合了预训练数据 (即 plain texts) 和 指令微调数据 (即 格式化数据集)的数据集通过 multi-task learning 来训练一个模型。特别是，GLM-130B 和 Galactica 在预训练语料库中集成了一小部分的指令格式化数据集来预训练 LLMs，这很可能能够同时利用预训练和指令微调的优势。

### 5.1.3 The Effect of Instruction Tuning

-  从三种角度来讨论

#### Performance Improvement

-  尽管已经被数量适中的实例微调过，instruction tuning 还是提升或解锁 LLMs 能力的关键途径。最近研究证实不同规模的语言模型 (从 77M 到 540B) 都从 instruction tuning 中受益，随着参数规模增加，性能得到提高。指令微调后，即便是小模型也表现强于未微调的大模型。除了模型规模，instruction tuning 演示了在多种模型架构，预训练目标的模型自适应方法上的持续进步。实践中，instruction tuning 提供了一种一般方法来增强现有语言模型 (包括 小规模 PLMs) 的能力。同样，它比预训练花费少，因为 LLMs 需要的指令数据数量显著的小于预训练数据。

#### Task Generalization

-  指令微调鼓励模型领会为了完成任务的自然语言指令。它赋予 endow LLMs  以某种能力 (经常被认为试试 emergent ability) 在不需要演示的情况下跟随人类指令来执行特定任务，即使是未见过的任务。大量研究证实 instruction tuning 在已见和未见的任务表现出卓越的性能。同样，instruction tuning 在减轻 LLMs 的若干弱点 (如无需完成某项任务就可以重复生成或补充输入)，从而为 LLMs 提供解决现实任务的卓越性能。指令微调过的 LLMs 可以泛化解决跨语言的相关问题。例如 BLOOMZ-P3 是在 BLOOM 上用只包含英文的 P3 任务集微调过的。有趣的是 BLOOMZ-P3 与 BLOOM 相比在多语言语句完成任务上有 50% 的提升，这表明指令微调能够帮助 LLMs 从 只有英文的数据中获得一般任务能力并转移到其它语言上。此外也发现使用只有英文的指令可以在多语言任务上产出满意的结果，这有助于减轻为了某个特定语言的指令工程的工作量。

#### Domain Specialization

-  现有的 LLMs 在传统 NLP 任务 (如，生成和推理) 和日常问题中展现出过人一等的能力。然而，他们仍缺少完成特定任务的领域知识，例如：医学、法律和金融。指令微调是使得现有通用 LLMs 成为特定领域专家的有效方法。例如，研究院提出使用药物数据集微调 Flan-PaLM 来创造 Med-PaLM，一个提供与专家临床医生相当水平的医药知识助手。最近一项研究微调了 FLAN-T5，使其支持带有自然语言指令的电商推荐系统，在各种推荐任务中显示出强大的性能。也有很多基于 LLaMA 微调过的开源医学模型，如 BenTsao。同样，研究人员探索在法律、金融和算术上的微调。

### 5.1.4 Empirical Analysis for Instruction Tuning

-  被不同指令集微调过的 LLMs 往往会在下游任务上具有不同性能的模型变体。本节，我们将探索不同类型指令微调 LLMs (即 7B LLaMA) 的效果，同时考察集中指令改进策略的有效性。

#### Instruction Datasets

-  主要考虑三种指令：

##### Task-specific instructions

-  第一种指令类型，我们使用最常用的多任务指令数据集 FLAN-T5，它包括 1836 个任务和超过 15M 指令，由之前工作的四种数据混合而成。

##### Daily chat instructions

-  这种类型指令是用户日常的对话，更接近真实生活场景。我们使用 ShareGOT 指令集，包括 63K 真实用户指令。它被 Vicuna 用作核心指令。

##### Synthetic instructions

-  除了重复使用现有指令外，我们还可以使用 LLMs 自动合成大量指令。我们使用正流行的合成指令数据集 Self-Instruct-52K，包括带有 82K 实例输入和输出的 52K 指令对。这些生成的指令与人类编写的种子任务有相同的数据分布 (如 语法检查，头脑风暴)。

-  由于原始 FLAN-T5 数据集非常大 (超过 15M)，我们从中随机采样了 80,000 指令来与其它指令数据集 (ShareGPT 和 Self-Instruct-52K) 在相似规模上进行公平比较。在我们试验中，我们测试每个独立指令集，以探索它们自身的影响，并检查它们对模型性能的组合影响。

#### Improvement Strategies

-  尽管现实世界从人类用户发出的指令更适合用于 LLMs 微调，但同时也很难大规模收集它们。作为替代，现在大多数研究主要使用 LLMs 生成的合成指令。然而，合成也有一些问题，如话题多样性欠缺和指令难度不平衡 (要不太简单要不太难)。因此，很有必要提升合成指令的质量。下面，我们总结四种主要提升策略:

##### Enhancing the instruction complexity

-  正如现有工作所讨论的，增强指令复杂度可以提升 LLMs 跟踪复杂指令的模型能力，如：包含跟多任务需求或要求更多推理步骤。为了验证这种策略，我们使用 WizardLM 来逐渐提升复杂度水平，如：增加约束、增加推理步骤和将输入复杂化。我们利用公开发布的 WizardLM-70K 指令作为复杂性增强指令数据集，该数据集是通过上述基于 Self-Instruct-52K 数据集的增强方法生成的。

##### Increasing the topic diversity

-  除了复杂性之外，提升指令数据集的主题多样性可以帮助引出 LLMs 在真实世界多样任务上的不同能力。然而，很难直接干预 self-instruct 过程来生成多样指令。依照 YuLan-Chat，我们使用 ChatGPT 改写 Self-Instruct-52K 数据集的指令，通过特定指令来使它们自适应 293 个主题。最终，我们收获了 70K 指令作为增长多样性数据集。

##### Scaling the instruction number

-  除了上面这些以外，指令的数量也是影响模型性能的一个重要因素。特别是使用可以扩展 LLMs 的任务知识和提升指令遵循能力的指令。为了检验这个策略，我们从 MOSS project 发布的合成指令集采样了新的指令，他们同样使用相同 self-instruct 方法合成的。我们将它们与 Self-Instruct-52K 数据集混合起来来制作一个包含 220K 指令的数据集。

##### Balancing the instruction difficulty

-  由于合成指令会太过于简单或太过于难，很容易导致 LLMs 训练不稳定或过拟合。为了探索这种潜在的影响，我们利用 LLMs 的 _困惑分数 perplexity score_  来估计指令的难度，并去掉太容易或太难的指令。为了生成相同规模的指令来进行公平的比较，我们使用 LLaMA-7B 模型来为来自大型指令数据集中的 220K 指令进行困惑度计算，然后保留困惑度分数适中的 70K 指令作为 difficulty-balanced 数据集。

#### 试验设置 Experimental Setup

-  为了试验指令数据的有效性，我们使用新的指令数据集来微调 LLaMA-7B，它是一个流行的 LLM 基座并被广泛用于指令微调。我们在试验中使用来自 YuLan-Chat 的代码，并在 8 个 A800-80G GPUs 上训练模型。保持和 Stanford Alpaca 相同的超参数设置。为了更好的评估微调后模型的指令遵循能力，我们考虑了两种设定，即 _Chat setting_ 和 _QA setting_。 _聊天设定 chat setting_ 主要利用用户日常的指令和提问，而 _问答设定 QA setting_ 主要利用现有 NLP 数据集中的问答示例。基于 AlpacaFarm 评估集对聊天设定进行评估。我们没有使用完整的两两比较，二是选择已在 Self-Instruct-52K 上微调的 LLaMA-7B 模型作为参考基线，然后将其与其它所有微调后的模型进行比较。由于我们关注的是检验生成指令的不同策略的作用性，在 Selfg-Instruct-52K  上微调过的模型可以作为一个很好的参考。遵循 AlpacaFarm，在每次比较时，我们使用 ChatGPT 来自动标注每次比较的两个模型中哪个对用户查询响应最好的，并报告 win rate (%) 作为评估指标。至于 QA 设置，我们选择两个基准，MMLU 和 BBH3K，并基于默认设置、通过使用启发式规则解析来自这些 LLMs 的答案来评估准确性。

-  对于指令微调和评估，我们采用以下 prompt：

>  The following is a conversation between a human and an AI assistant. The AI assistant gives helpful, detailed, and polite answers to the user’s questions.\\n
>   \[|Human|]: {input} \\n 
>   \[|AI|]:

-  具体参见 [https://github.com/RUCAIBox/LLMSurvey/tree/main/Experiments.]

#### Results and Analysis

-  我们试着总结和分析一下。

-  _Task-formatted instructions are more proper for the QA setting, but may not be useful for the chat setting 任务格式化的指令更适合 QA 设置，也许并不适用聊天设置_。通过比较使用 FLAN-T5 和使用 ShareGPT 和 Self-Instruct-52K 进行指令微调的性能，可以发现 FLAN-T5 总是在 QA 基准上表现的更好，而在聊天设置上表现不如 ShareGPT。原因是 FLAN-T5 是由来自现有 NLP 任务的指令和示例混合组成的，比如，翻译和阅读理解。因此， 使用 FLAN-T5 微调的 LLaMA 在 QA 任务上表现更好，但在用户查询上乏善可陈。相反，ShareGPT 包含真实世界的 human-ChatGPT 对话，这可以更好的引导 LLaMA 在日常生活中去遵循用户指示，而很可能不适合 QA 任务。

-  _A mixture of different kinds of instructions are very helpful in improving the comprehensive abilities of LLMs 多种多样的指示对提升 LLMs 的理解能力有很大的帮助_。在混合了三种微调指令后，可以看到产生的 LLaMA 变种在两种任务设置上都表现出色。它标明混合多种来源的指令数据集对于提升指令微调的 LLMs 性能有很大帮助，它扩展了指令数量又增加了多样性。

-  _Enhancing the complexity and diversity of instructions leads to an improved model performance 增强指令的复杂度和多样性会提升模型性能_。 

- _Simply increasing the number of instructions may not be that useful, and balancing the difficulty is not always helpful 简单的增长指令的数量也许并没有那么有用，平衡困难度也不是那么有用_。

---

## 5.2 对齐微调 Alignment Tuning

-  这部分首先介绍 _对齐 alignment_ 的背景包括它的定义和标准，然后关注为了对齐 LLMs 收集人工反馈数据，最后讨论对齐微调 RLHF 的核心技术。

### 5.2.1 Background and Criteria for Alignment

#### Background

-  LLMs 在一系列 NLP 任务上展示出卓越的能力。然而，这些模型有时会 _表现出意想不到的行为 exhibit unintended behaviors_，比如，捏造错误信息，追求不准确的目标，产生有害的、错误引导的、有偏见的表达。对于 LLMs，语言建模目标是通过单词预测对模型参数进行预训练，而缺乏对人的价值观和偏好的考虑。为了 _防止 avert_ 意想不到的行为，提出了 _human alignment 人工对齐_ 来让 LLMs 表现的符合人类期望。与原始的预训练的自适应微调 (如指令微调) 不同，这种对齐需要考虑多种标准，如帮助性、真实性和无害性。某种程度上，对齐也许会伤害 LLMs 的一般能力，这也被叫做 _对齐税 alignment tax_。

#### 对齐标准 Alignment Criteria

-  现在越来越多的关注，如何制定五花八门的标准来约束 LLMs 的行为。在这里，我们提出三个代表性对齐标准 _helpful、honest、harmless_ 作为讨论对象，它们都已被广泛接受。此外，还有很多其它关于 LLMs 各种角度的对齐标准，如 _behavior、intent、incentive and inner aspects_，都基本与前面三个大差不差。同样也可以根据特定需求来完善三个标准，如用正确性代替诚实性。

##### Helpfulness

-  LLM 的帮助性体现在清楚地帮助用户解决它们的任务或回答问题，并尽可能的简洁有效。更高层次的话，当需要进一步澄清时，LLM 能够展示出通过相关问询获得额外相关信息的能力，并表现出适当的敏感性、洞察力和审慎性。由于 LLMs 很难准确地定义和衡量用户的意图，实现帮助性行为的对齐对于 LLMs 是个挑战。

##### Honesty

-  基本层面上，一个被 _诚实性对齐_ 过的 LLM 应该向用户展现出准确的内容，而不是捏造信息。此外，至关重要的是，LLM 在其输出中传达适当程度的不确定性，以避免任何形式的欺骗或虚假陈述信息。这就要求模型了解自己的能力和知识水平。Honesty 是一个比 helpfulness 和 harmlessness 更客观的标准，增强诚实性对齐潜在的可以减少对人类的依赖。

##### Harmlessness

-  为了达到无害性，模型产生的语言不应是攻击性或歧视性的。模型应尽其所能的检测那些为了恶意目的而请求的隐蔽行为。理想状况，当模型被诱导作出危险行为， LLM 应礼貌地拒绝。然而，什么行为被认为是有害的，对于个人或是社会有害到多大程度，是取决于谁使用 LLM，所提问题的类型，以及 LLM 被使用时的上下文。

-  正如我们所见，这些标准相当的主观，都是基于人类认知。因此，很难将它们直接转化为 LLMs 的优化目标。当今的 LLMs 对齐工作，有大量实现这些标准的途径。其中一个有前途的技术叫做 _红队测试 red teaming_，其中包括使用手动或自动手段以对抗方式探测 LLMs 去生成有害输出，然后更新 LLMs 来防止此类输出。

### 5.2.2 收集人工反馈 Collectiong Human Feedback

-  预训练阶段，通过在大规模语料库上使用语言建模目标来训练 LLMs。但它并没有考虑到人类对 LLM 输出的主观和定性评价，本文叫做 _人工反馈 human feedback_。高质量人工反馈有着人类偏好和价值对于对齐 LLMs 有着极重要作用。

#### 人工标注者选择 Human Labeler Selection

-  当今，占据领先地位的生成人工反馈数据的方法是 _人工标注 human annotation_。这突出了选择合适人工标签员的关键。人工标签员需要能够有合格的教育水平和熟练的英文。例如，Sparrow 需要人工标签员是 UK 背景的英语母语的本地人，而且至少是接受过本科教育。即使这样，几项研究发现仍然存在研究人员的意图与人工标注者之间的不匹配，这会导致低质量的人工反馈，并导致 LLMs 产生意想不到的输出。为了解决这个问题，InstructGPT 进一步通过评估人工标注者和研究人员之间的一致性，进行面试过程来筛选标注者。详细的说，研究人员首先标注一小部分数据，然后衡量和人工标注者之间的一致性。有着高一致性的标注者会被选中去进行下一步标注工作。其它工作中，“super raters” 被用来保证高质量的人工反馈。研究人员评估人工标注者的表现，并选出一组表现优异的作为 super raters。Super raters 被赋予在下一步研究中与研究人员合作的机会。当人工标注者去标注 LLMs 的输出，为他们提供详细的说明和即时的指导，有助于进一步规范标注者的标注。

#### Human Feedback Collection

-  当今，主要有三种收集人工标注员的反馈和偏好数据的方法。

##### 基于排序 Ranking-based approach

-  早期工作中，人工标注者通常以粗颗粒度的方式评估模型生成的输出 (即只选择最好的)，而不考虑细颗粒度的对齐标准。尽管如此，不同的标注者可能会对最佳输出候选集的选择持有不同意见，这种方法会忽略未选择的样本，这可能导致不正确和不完整的人工反馈。进一步的研究介绍了 _Elo rating system_ ，它通过比较候选输出得出偏好排序。输出的排序作为训练信号来引导模型偏爱某些输出，从而产生更可靠更安全的输出。

##### 基于问题 Question-based approach

-  更有甚者，人工标注者通过回答研究人员设计的某些问题来提供更详细的反馈，涵盖对齐标准以及对 LLMs 的额外限制。在 WebGPT，为了帮助模型筛选和利用从检索到的文档中获取相关信息，人工标注者被要求回答多个问题，关于检索到的文档是否对回答给定输入有用的选项。

##### 基于规则 Rule-based approach

-  也有很多研究开发了基于规则的方法来提供更详细的人工反馈。典型示例，Sparrow 不仅选择标注者认为最好的回答，而且使用一系列的规则来测试模型生成回答是否符合 helpful、correct 和 harmless 的对齐标准。在这种情况下，能够获得两种人工反馈数据：
1.  通过成对比较模型生成的输出的质量，得到反映偏好反馈。
2.  通过生成人工标注者的评价，得到违反规则反馈 (即，打分反映生成的输出在多大程度上违反了规则)
-  GPT-4 进一步利用了一套零样本分类器 (基于 GPT-4自身) 作为 _基于规则的奖励模型 rule-based reward models_，它能自动决定模型生成的输出是否违反了一套人工编写的规则。

-  接下来，关注与 __reinforcement learning from human feedback RLHF__，它已经广泛应用于最近的 LLMs 如 ChatGPT。

![[Pasted image 20231031153210.png]]


### 5.2.3 人工反馈的强化学习 Reinforcement Learning from Human Feedback

-  RLHF 被提出通过收集的人工反馈数据来微调 LLMs，是为了用人类价值观来对齐 LLMs，RLHF 对提升对齐标准很有用。RLHF 采用 _强化学习算法 reinforcement learning algorithms_ ，如 _Proximal Policy Optimization PPO_，通过学习一个 _奖励模型 reward model_ 使 LLMs 适应人工反馈。这种方法将人类纳入开发良好对其的 LLMs 的训练循环，正如 InstructGPT。

#### RLHF System

-  主要包括三个关键组件：_待对齐的预训练语言模型_、_一个通过人工反馈学习的奖励模型_以及 _一个训练 LM 的 RL 算法_。详细说，_pre-trained LM_ 是典型的一个生成模型，通过已有的预训练 LM 参数初始化过。例如 OpenAI 使用 175B GPT-3 作为它的首个 RLHF 模型 InstructGPT， DeepMind 使用 280B 参数模型 Gopher 作为它的 GopherCite 模型。此外， _reward model RM_ 提供 (已学习的) 指导信号，这些信号通常以标量值的形式，反映了人类对 LM 生成的文本偏好。 RM 有两种形式：一个微调过的 LM 或一个适用人类偏好数据重新训练过的 LM。现有工作通常采用与对齐过的 LM 不同参数尺度的奖励模型。例如，OpenAI 使用 6B GPT-3 ，DeepMind 使用 7B Gopher。最终，使用奖励模型的信号来优化 pre-trained LM，需要为大规模模型微调设计一个特定的 RL algorithm。具体地说， _PPO_ 是现在被广泛使用的 RL 算法。

#### RLHF关键步骤 Key Steps for RLHF

##### Supervised fine-tuning

-  想要 LM 表现理想中的行为，通常需要需要收集一个监督学习数据集，包含输入提示 (指令)和为了微调 LM 的理想的输出。为了保证任务的多样性，这些体术和输出可以由人工标注者为某些特定任务编写。例如，InstructGPT 要求人工标注者为若干生成任务如 开放性 QA，头脑风暴，聊天和重写来编写提示。注意，在特定设置和情态，第一步是可选的。

##### Reward model training

-  第二步是使用人工反馈数据训练 RM。具体地，我们使用 LM 的样本提示来生成特定数量的输出文本作为输入。然后邀请人工标注者来为这些对标注偏好。标注过程分成几种形式，一种普遍方法是通过排序生成的候选集来标注，这样可以减少标注者之间的不一致性。然后，训练 RM 来预测人类骗号的输出。在 InstructGPT，标注者从 best 到 worst 排序模型生成的输出，训练 RM 来预测这个排序。

##### RL fine-tuning

-  在这一步中，对齐 LM 被格式化为一个 RL 问题。在这种设置下，预训练过的 LM 像一个 policy，接受一个 prompt 作为输入然后返回输出文本，_action space_ 是整个词典，_state_ 是当前生成的 token 序列，由 RM 提供奖励。为了避免明显偏离初始 (微调前) LM，奖励函数中通常纳入一项惩罚条款。例如，InstructGPT 针对 RM 使用了 PPO 算法优化了 LM。对于每个输入提示，InstructGPT 计算当前 LM 和初始 LM 的 _KL 散度 KL divergence_ 作为惩罚。注意，为了更好地对齐 LLMs，第二步和最后一步可能在多轮中多次迭代。由于 RL 算法的不稳定性，最近的研究使用另一种有监督微调——重复使用具有更高奖励的最佳排名样本，来取代 RL 微调。

## 5.3 参数有效模型自适应 Parameter-Efficient Model Adaption

-  以上讨论了根据特定目标，指令微调和对齐微调 LLM。由于 LLMs 包含海量模型参数，运行 _全参数微调 full-parameter tuning_ 将会花费巨大。本小节，我们将会讨论如何对 LLMs 高效微调。

### 5.3.1 参数有效微调方法 Parameter-Efficient Fine-Tuning Methods

-  现有文献，参数有效微调已经是一个重要主题，它在保证良好性能的同时减少了可训练参数的数量。以下是四种 Transformer 语言模型的参数有效微调方法，包括 _adapter tuning_、_prefix tuning_、_prompt tuning_ 和 _LoRA_。

#### 适配器微调 Adapter tuning

-  Adapter tuning 将一些小的神经网络模块 (即 _adapter_)  混合到 Transformer 模型中。为了执行 adapter 模块，提出了一种 _瓶颈架构 bottleneck architecture_ ，该架构首先将原始特征向量压缩到一个更低的维度 (后接一个 _非线性转换 nonlinear transformation_ ) ，然后将其恢复到原始维度。Adapter modules 被集成到 Transformer 每一层，通常在两个核心层 (即 attention layer 和 feed-forward layer) 后使用 _一系列插入 a serial insertion_。另一种，并行 adapters 也可以在 Transformers 层中使用，它将在 attention layer 和 feed-forward layer 中并行放置两个 adapter modules。在微调期间，根据特定任务目标优化 adapter modules，同时冻结原始语言模型的参数。这样就可以有效的减少微调期间的训练参数量级。

#### 前缀微调 Prefix Tuning

- Prefix tuning 在语言模型的每个 Transformer layer 添加前缀序列，该序列是一组可训练的连续向量。这些前缀向量是用于特定目的，可以被认作是虚拟 token embeddings。为了优化前缀向量，提出了一种 reparameterization 技巧，即通过学习一种将一个小矩阵映射到前缀参数矩阵的 MLP 函数，而不是直接优化前缀。对于稳定训练，这个技巧已经被证实很有用。优化后，映射函数会被丢弃，只保留派生的前缀向量来提高特定任务的表现。由于只会训练前缀参数，这会导致参数有效的模型优化。与 prefix tuning 类似，p-tuning v2 将每层提示向量包含到 Transformer 架构，特别是为了 NLU，它也可使用多任务学习来联合优化共享的提示。它在 NLU 任务上以不同参数规模提升模型性能的作用已经被证实了。

#### 提示微调 Prompt Tuning

-  与 prefix tuning 不同，提示微调只关注在输入层加入可训练的提示向量。基于 _离散提示方法 discrete prompting methods_，它可以增强输入文本，因为其包含了一组 soft prompt tokens (以自由形式或前缀形式)，然后接受提示增强过的输入来解决特定下游任务。执行中，特定任务提示嵌入和输入文本嵌入结合，然后输入到语言模型中。P-tuning 提出一种自由形式来结合上下文、提示和目标 tokens，可以被应用于 NLU 和 NLG 架构中。进一步通过 bidrectional LSTM 学习到 soft prompt tokens 的表示。另一种代表性方法，叫做 _提示微调 prompt tuning_ 直接在输入头部追加前缀提示。训练中根据特定任务监督，只有 prompt embeddings 会被学习到。由于这种方法只会在输入层包含一小部分数量的可训练参数，发现它表现好坏依赖于下层语言模型的模型能力。

#### 低秩自适应 Low-Rank Adaptation LoRA

-  LoRA 在每个密基层为逼近更新矩阵施加了低秩约束。如要优化一个参数矩阵 $W$，以一般形式写出更新过程: $W \leftarrow W + \Delta W$。LoRA 的主思想是冻结原始矩阵 $W \in \mathbb R ^{m \times n}$ 同时通过低秩分解矩阵逼近参数更新 $\Delta W$，即 $\Delta W = A \cdot B^{\top}$ $A \in \mathbb R^{m \times k}$ 和 $B \in \mathbb R^{n \times k}$   是任务自适应的可训练参数， $k << min(m, n)$ 是降低的秩 rank。LoRA 最大的价值是能够极大地节省内存和存储空间。此外，一个只能保留一个大的模型副本，同时为了适应不同的下游任务维护许多特定任务的低秩分解矩阵。此外，很多研究也讨论了如何以更有原则的方法设置秩，比如，机遇重要性分数的分配和无搜索的最优排名选择。


![[Pasted image 20231109135736.png]]

-  除了以上方法，有众多关于 Transformer 有效微调的延伸研究。

### 5.3.2 Parameter-Efficient Fine-Tuning on LLMs

-  随着 LLMs 的兴起，有效微调吸引了大量研究关注。
-  特别的是，LoRA 被广泛应用于开源 LLMs 作为参数有效微调。例如 AAlpaca-LoRA 使用 LoRA 作为 Alpaca 的一个轻量微调版本。一项最近研究 LLaMA-Adapter 在每个 Transformer 层插入可学习的提示向量，zero-initialized attention 被提出用来提升训练，他可以减轻 under-fitted prompt vector 的影响。他们同样延伸到多模态设置，如 视觉问答。

-  此外，实证研究不同微调方法对 LM 的影响。比较了四种有效微调方法包括 _序列适配器微调 serial adapter tuning_，_并行适配器微调 parallel adapter tuning_ 和 LoRA , 在三个开源 LLMs 即 GPT-J 6B、BLOOM 7.1B 和 LLaMA 7B 上评估。基于在六个数学推理数据集上的实验结果，表明这些高效的调优方法在困难任务上的表现低于参考基准 GPT-3.5，而在简单任务上则达到了相当的性能。总体来说，所有方法中 LoRA 相对表现很好，同时使用更少的训练参数。

-  作为重要资源，PEFT 库在 GtHub 上已开源。其中有着大量流行的有效微调方法，包括 LoRA / AdaLoRA，prefix-tuning，P-Tuning 和 prompt-tuning。。此外，还支持想 GPT-2 和 LLaMA 这样的 LM，和几个代表性的 vision Transformer models (例如 ViT 和 Swin Transformer)。

-  如上所说，现在市面上提出了太多的高效微调方法。然后，它们中的大多数都只是在小规模预训练 LM 上进行了测试，而不是 LLMs。目前为止，仍缺少这方面的调查研究。

## 5.4 高效内存模型自适应 Memory-Efficient Model Adaptation

-  由于模型参数的巨大数量级，LLMs 需要极大的内存占用用于推理，现实世界应用的部署就需要巨大花费。随后我们讨论如何使用模型压缩方法 (即 _模型量化 model quantization_) 来减少内存占用，这样大规模 LLMs 可以使用资源有限的设置，也会减少推理延迟。

### 5.4.1 Background for Quantization

-  在神经网络压缩中，_量化 quantization_ 通常指代从 _浮点数 floating-point numbers_ 到 integers 的映射过程，特别是 8-bit 整数量化 (即 _INT8 quantization_) 。对于神经网络模型，普遍有两类数据可以被量化，即 _权重 weights (model parameters)_ 和 _激活函数 activations (hidden activations)_，原始都是以浮点数表示的。为了说明模型量化的基本思想，我们引入一个既简单又受欢迎的量化函数 $x_q = R(x / S) - Z$ , 表示将 浮点数 $x$ 转换成一个量化后的值 $x_q$。$S$ 和 $Z$ 指代 _换算系数 scaling factor_ (包括决定 剪切范围 clipping range 的两个参数 $\alpha$ 和 $\beta$ ) 和 _零点因子 zero-point factor_ (决定量化是对称还是非对称)， $R(\cdot)$ 指代 _舍入运算 rounding operation_  是将缩放后的浮点值映射成整数。

-  作为逆过程，_反量化 dequantization_ 将量化后的值还原成原始值: $\tilde x = S \cdot (x_q + Z)$。 _量化误差 quantization error_ 指原始值 $x$ 和复原值 $\tilde x$ 之间的差值。参数 $\alpha$ 和 $\beta$ 的范围对量化性能有值巨大影响，通常需要根据真实数据分布来 _校准 calibrated_ ，离线是 _static_ 方法，运行时是 _动态 dynamic_ 方法。

### 5.4.2 Quantization Methods for LLMs

-  有两种主要模型量化方法：_quantization-aware training QAT_ 需要附加全模型再训练和 _post-training quantization PTQ_ 无需模型再训练。与小尺寸 LM 相比，LLMs 需要考虑两个主要差异：首先 LLMs 包含巨多参数，因此 PTQ 比 QAT 的计算花费更低；第二，LLMs 展示出非常不同的激活模型 即 _大异常值特征 large outlier features_，因此很难量化 LLMs，特别是隐藏层激活函数。

#### 训练后量化 Post-Training Quantization PTQ

##### 混合精度分解 Mixed-precision decomposition

-  当模型达到 6.7B 参数及以上，隐藏激活会出现极大的数值，叫做 _异常值涌现 the emergence of outliers_ 。这些异常值主要分布在 Transformer layers 的一些特定的特征空间。根据此项发现，一种 vector-wise 量化方法，叫做 _LLM.int8()_ 被提出来，它在矩阵乘法中将 带有异常值的特征维度与其它维度区分开。然后分别用 16-bit floating numbers 和 8-bit inegers 和这两部分相乘，这样以高精度还原这些异常值。

##### 细颗粒度量化 Fine-grained quantization

-  对于 Transformer 模型，weight 和 activations 通常以 _张量 tensor_ 形式。一种直接方法就是对整体张量 (即 per-tensor quantization) 使用粗颗粒度量化参数。然后这会导致重建结果的不准确。因此，提出了细颗粒度量化来减少量化误差。 ZeroQuant 使用一种 token-wise 量化方法即动态校准压缩激活。同时，对于 weights ，由于容易被量化，对其使用一种 group-wise quantization。实践中，模型量化通常使用一组大小为 128.

##### Balancing the quantization difficulty

-  考虑到 weights 比 activations 更容易被 quantized， SmoothQuant 提出将困难从 activations 迁移到 weights。特别是用一个线性层使用 scaling transformation 来平衡 weights 和 activations 的难度 $Y = (X diag(s)^{-1}) \cdot (diag(s)W)$。
-  通过引入一个 _数学等价变换 mathmatically equivalent transformation_，上面公式通过 scaling factor $s$ 来控制量化难度。为了设置 $s$，它引入了迁移强度参数 $\alpha$ 来平衡难度，每个 $s_j = \max(\text x_j)^{\alpha} / \max(\text w_j)^{1-\alpha}$ 都由迁移强度来决定。

##### 分层量化 Layerwise quantization

-  这种方法可以找到最优将分层重建最小化的量化权重损失: $\arg \min_{\hat {\text W}} ||\text {WX} - \hat {\text W} \text X||^2_2$。为了高效优化这个目标函数，GPTQ 通过固定所有行的权重量化顺序提升了原始的 _最优脑量化 optimal brain quantization OBQ_ 方法。此外，有了特别设计的方法，即 lazy batch-updates 和 Cholesky reformulation，GPTQ 可用于在 3 或 4 bit 精度量化大模型 如 175B OPT。最近，AWQ 进一步简化了优化形式，结合了激活感知的权重缩放，这类似于 SmoothQuant 的思想：与异常值激活相对应的权重，更要被精确量化。它不会直接优化重建损失，而是在校准数据中运行超参数搜索来达到损失最小化。以上方法会是多种策略联合使用来提升量化性能。为了达到高效实现，量化方法也取决了硬件或是系统层级的支持，如高效GPU kernels 或硬件友好的组分区。

#### Other Quantization Methods

##### 高效微调增强量化 Efficient fine-tuning enhanced quantization

-  对于训练后量化，_直接低比特量化 direct low-bit quantization_ 如 INT4 quantization 往往会导致性能的严重下降。为了克服挑战，QLoRA 在量化模型中混合了附加的小型可调适配器 (16-bit precision)，来达到一种高效高精度的模型微调。 它包含了 LoRA 的优点和量化方法。这项实验表明使用 QLoRA，4-bit 量化模型能够达到 全16-bit微调后的性能。

##### Quantization-aware training QAT for LLMs

-  最近的一项研究通过应用 _无数据蒸馏 data-free distilation_ 方法来压缩权重，激活以及键值缓存，探索了 QAT 方法的效果。通过引导基于 LLaMA 的延伸实验，在权重和键值缓存 4-bit quantization 上表现出理想的结果，但是 4-bit 激活量化还有待更多探索。

### 5.4.3 实证分析和研究 Empirical Analysis and Findings

-  量化目前已成为减少 LLMs 部署中内存占用和延迟的常用技术。特别重要的是要了解什么精度级别 (如 INT8 或 INT4) 可以应用在量化 LLMs 的不同部分 (如 权重或激活)，同时保持高精度。

-  关于不同因素 (如模型大小和敏感性) 在后训练量化方法中的影响有了一项非常全面的评估。还有一项研究检视了 k-bit 量化的伸缩定律在推理方面的表现。同样，之前工作 如 LLM.int8()、GPTQ、QLoRA 和 GLM 还广泛地检查了量化方法在各种设置中的性能。下面，我们总结了几项这些研究中的重要发现，这有助于帮助大家简要理解而无需深入。

 -  _INT8 weight quantization can often yield very good results on LLMs, while the performance of lower precision weight quantization depends on specific methods_。大多数情况，INT8 权重向量可以被有效的应用于在不降低性能情况下减少内存占用。同时，对于 INT4 (或 INT3) 权重量化，现有方法依赖于特定策略来减少性能降级，如 layerwise method, activation-aware scaling 和 low-rank adapter tuning。有趣的是，LLMs 比小模型对于 low-bit 权重量化更不敏感。实践中，在相同内存占用情况下，建议使用有着更低量化精度的大模型，而不是有着高量化精度的小模型。比如，4-bit 60GB LLM 比 8-bit 30GB LLM 好得多。

-  _Activations are more difficult to be quantized than weights_。已经证实 Transformer LM 超过 6.7B 就会出现极大的异常值，这时 LLMs 难被量化的最基础难题。提出了各种各样的方法，如 mixed-precision decomposituion, fine-grained quantization 和 difficulty migration 去减轻异常值的影响。由于极大异常值主要存在于 LLMs 的激活，小模型对于激活量化更抵触。实践中，高质量 INT8 激活量化仍是个困难的任务，尽管很多方法可以达到满意的结果。此外，低精度激活量化仍未被成功探索过，即使是 QAT。

- _Efficient fine-tuning enhanced quantization is a good option to enhance the performance of quantized LLMs_。量化中的高效微调方法有两部分的好处。第一，它能通过更新高精度适配器来增强适宜的容量，这样直接弥补了 low-bit 量化带来的性能衰退。第二，只需微调小型适配器，它就能轻量灵活地支持特定任务或特定目标的微调 LLMs，如 指令微调或聊天指向的微调。总体来说，它是有效性和训练成本之间的一种很好的权衡，提供了一种提高量化 LLMs 的性能的有前途的方法。

### 5.4.4 Open-source Libraries and Quantized LLMs

#### Quantization Libraries

-  三种主要的 LLMs 量化库，包括：

##### Bitsandbytes

-  INT8 quantization

##### GPTQ-for-LLaMA

- 提供基于 GPTQ 算法的对 各种大小 LLaMA 模型的 4-bit quantization

##### AutoGPTQ

-  基于 GPTQ 开发的量化包，支持 LLMs 的 INT4 quantization

-  llama.cpp 使得在 MacBook 上运行 quantized LLaMA 模型成为可能。它支持 INT4、INT5、INT8 quantization，使用 C/C++ 开发。支持类 LLaMA 模型，如 Alpaca 和 Vicuna

#### Quantized LLMs

-  与原始模型相比，量化的模型占用内存更少，推理速度更快。GPTQ 广泛用于量化生成式语言模型，导致了 LLaMA 和 OPT 的各种量化变种。此外，也用于量化指令微调后的模型，如 Vicuna 和
-  WizardLM。由于量化 LLMs 的数量过于多，我们无法一一包括。

---

# 6 利用 UTILIZATION

-  预训练和自适应微调过后，使用 LLMs 的主要方式就是维解决各种任务而设计合适的 prompting strategies。典型的提示方法叫 _上下文学习 in-context learning_，它用自然语言文本确切地构造任务描述 和/或 证明。

## 6.1 In-Context Learning

-  GPT-3 首次提出了 ICL

### 6.1.1 提示公式化 Prompting Formulation