---
layout: post
title: "基于判别式的分类" 
---


# 1、判别式函数

对于分类任务，我们定义一组判别式函数$g_i(\boldsymbol{x}), i=1,2,...,K$，并且如果$g_j(\boldsymbol{x}) = \max_{i=1}^K g_i(\boldsymbol{x})$，我们就选择$C_j$。

前面我们讨论的贝叶斯分类器中，首先估计先验概率$p(C_i)$和类似然$p(\boldsymbol{x}\mid C_i)$，再使用贝叶斯规则计算后验密度。然后，我们使用后验密度定义判别式函数，即

$$g_i(\boldsymbol{x}) = \log p(C_i\mid \boldsymbol{x})$$

这称作基于似然的分类（likelihood-based classification，属于“生成式模型”，因为它是基于先验和似然来计算联合分布）。

当然我们还可以绕过似然或后验概率的估计，直接为判别式假定模型，如使用参数$$\boldsymbol{w}_i$$来定义

$$g_i(\boldsymbol{x}\mid \boldsymbol{w}_i)$$

然后学习优化模型参数$$\boldsymbol{w}$$，最小化训练集上的分类误差（一般使用梯度下降法）。这称为基于判别式的分类（discriminant-based classifaction，属于“判别式模型”）。


# 2、线性判别式

如果类密度$p(\boldsymbol{x}\mid C_i)$是高斯的，并且所有类的实例分布具有共同的协方差矩阵，则判别式函数是线性的：


$$g_i(\boldsymbol{x}) = \boldsymbol{w}_i^T\boldsymbol{x} +  w_{i0} = \sum_{j=1}^d w_{ij}x_j + w_{i0}$$

其中参数可以用下式解析的计算：

$$\boldsymbol{w}_i = \Sigma^{-1}\boldsymbol{\mu}_i$$

$$w_{i0} = -\frac{1}{2}\boldsymbol{\mu}_i^T\Sigma^{-1}\boldsymbol{\mu}_i + \log p(C_i)$$

给定训练集后，我们可以首先计算$\Sigma$、$$\boldsymbol{\mu}_i$$、$$p(C_i)$$的估计，然后计算线性判别式的参数并得到判别式模型。


# 3、逻辑斯谛判别式

我们首先考虑二分类的特殊情况。

在分类时，如果

$$\begin{cases}p(C_1 \mid \boldsymbol{x})>0.5 \\ \frac{p(C_1 \mid \boldsymbol{x})}{1-p(C_1 \mid \boldsymbol{x})}>1 \\ \log \frac{p(C_1 \mid \boldsymbol{x})}{1-p(C_1 \mid \boldsymbol{x})} >0   \end{cases}$$

则会选择$C_1$，否则选择$C_2$。

$\frac{p(C_1 \mid \boldsymbol{x})}{1-p(C_1 \mid \boldsymbol{x})}$称为“几率”（odds），反映了$\boldsymbol{x}$作为$$C_1$$类的相对可能性。对几率取对数，即$$\log\frac{p(C_1 \mid \boldsymbol{x})}{1-p(C_1 \mid \boldsymbol{x})}$$称为“对数几率”（log odds)，又称“分对数”（logit）。

在两个共享相同的协方差矩阵的正态类的情况下，对数几率是线性的：

$$logit(p(C_1 \mid \boldsymbol{x})) = \log\frac{p(C_1 \mid \boldsymbol{x})}{1-p(C_1 \mid \boldsymbol{x})} = \boldsymbol{w}^T\boldsymbol{x}+w_{0}$$

对数几率的逆是逻辑斯蒂（logistic）函数，又称S形（Sigmoid）函数：

$$p(C_1 \mid \boldsymbol{x}) = sigmoid(\boldsymbol{w}^T\boldsymbol{x}+w_{0}) = \frac{1}{1+e^{-(\boldsymbol{w}^T\boldsymbol{x}+w_0)}} = \frac{e^{\boldsymbol{w}^T\boldsymbol{x}+w_0}}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+w_0}}$$

于是可得

$$p(C_2 \mid \boldsymbol{x}) = \frac{1}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+w_0}}$$

给定训练数据集$$D = \{(\boldsymbol{x}^{(1)},y^{(1)}), (\boldsymbol{x}^{(2)},y^{(2)}), ..., (\boldsymbol{x}^{(N)},y^{(N)})\}$$，可以利用“最大似然法”来估计参数$\boldsymbol{w}$、$w_0$。对数似然函数为：

$$\mathcal{L}(\boldsymbol{w}, w_0) = \sum_{s=1}^N \log p(y^{(s)} \mid \boldsymbol{x}^{(s)}; \boldsymbol{w}, w_0)$$

该参数估计过程又叫逻辑斯谛回归。

以上讨论的都是二分类问题，逻辑斯谛判别式也可用于多分类问题。设数据样本共可能取$K$个类，则定义对数几率：

$$\log\frac{p(C_k \mid \boldsymbol{x})}{p(C_K \mid \boldsymbol{x})} = \boldsymbol{w}_k^T\boldsymbol{x}+w_{k0} ,\ k=1,2,...,K-1$$

于是可得到：

$$p(C_k \mid \boldsymbol{x}) = \frac{e^{\boldsymbol{w}_k^T\boldsymbol{x}+w_{k0}}}{1+\sum_{k'=1}^{K-1}e^{\boldsymbol{w}_{k'}^T\boldsymbol{x}+w_{k'0}}} ,\ k=1,2,...,K-1$$

$$p(C_K \mid \boldsymbol{x}) = \frac{1}{1+\sum_{k'=1}^{K-1}e^{\boldsymbol{w}_{k'}^T\boldsymbol{x}+w_{k'0}}}$$

其参数估计方法与二分类逻辑斯谛回归类似。




