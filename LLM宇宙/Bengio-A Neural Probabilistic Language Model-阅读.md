# 一种神经概率语言模型 A Neural Probabilistic Language Model

> Abstract
> A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. Traditional but very successful approaches based on n-grams obtain generalization by concatenating very short overlapping sequences seen in the training set. We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences. The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations. Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence. Training such large models (with millions of parameters) within a reasonable time is itself a significant challenge. We report on experiments using neural networks for the probability function, showing on two text corpora that the proposed approach significantly improves on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts.
>  Keywords: Statistical language modeling, artificial neural networks, distributed representation, curse of dimensionality

## Abstract

-  统计语言模型的目标就是学习语言中单词序列的联合概率分布。难度本质上就是 _维度灾难 curse of dimensionality_ ：模型测试的单词序列与训练时往往大不相同。

## 1. 介绍

-  造成语言建模的其他学习问题困难的根本原因是 _维度灾难_。