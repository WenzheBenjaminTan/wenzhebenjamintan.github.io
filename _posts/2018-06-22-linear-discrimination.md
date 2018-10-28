---
layout: post
title: "线性判别式" 
---

# 1、广义线性模型

考虑单调可微函数$g(\cdot)$，令$$y = g^{-1}(\boldsymbol{w}^T\boldsymbol{x}+b)$$，这样得到的模型称为“广义线性模型”（generalized linear model）,其中函数$g$称为“联系函数”（link function）。

对数线性回归是广义线性模型在$$g(\cdot) = \ln(\cdot)$$时的特性，即：

$$y = e^{\boldsymbol{w}^T\boldsymbol{x}+b}$$


# 2、逻辑回归

广义线性模型不但可以用于回归任务，也可用于分类任务。考虑二分类问题，其输出标记$$y\in \{0,1\}$$，而线性回归模型产生的预测值$$z = \boldsymbol{w}^T\boldsymbol{x}+b$$是实值，于是，我们将实值$z$转换为$0/1$值。最理想的是“单位阶跃函数”（unit-step function）：

$$y = \begin{cases} 0 & z < 0 \\ 0.5 & z = 0 \\ 1 & z > 1 \end{cases}$$

这里$y$表示样本$\boldsymbol{x}$作为正例的可能性，因此，若预测值$z$大于零就判为正例，小于零则为反例，预测值为临界值零则可任意判别。

但因为单位阶跃函数不连续，不能直接用作$g^{-1}(\cdot)$，而“Sigmoid函数”是一个常用的替代函数：

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

于是可得到:

$$y = \frac{1}{1+e^{-(\boldsymbol{w}^T\boldsymbol{x}+b)}}$$

确切地说，$y$应该表示为$$p(Y=1 \mid \boldsymbol{x})$$，进而可得：

$$\ln\frac{p(Y=1 \mid \boldsymbol{x})}{1-p(Y=1 \mid \boldsymbol{x})} = \ln\frac{p(Y=1 \mid \boldsymbol{x})}{p(Y=0 \mid \boldsymbol{x})} = \boldsymbol{w}^T\boldsymbol{x}+b$$

$\frac{p(Y=1 \mid \boldsymbol{x})}{p(Y=0 \mid \boldsymbol{x})}$称为“几率”（odds），反映了$\boldsymbol{x}$作为正例的相对可能性。对几率取对数，即$$\ln\frac{p(Y=1 \mid \boldsymbol{x})}{p(Y=0 \mid \boldsymbol{x})}$$称为“对数几率”（log odds，又称logit）。

我们进一步列出似然概率：

$$p(Y=1 \mid \boldsymbol{x}; \boldsymbol{w}, b) = \frac{e^{\boldsymbol{w}^T\boldsymbol{x}+b}}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+b}}$$

$$p(Y=0 \mid \boldsymbol{x}; \boldsymbol{w}, b) = \frac{1}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+b}}$$

给定训练数据集$$D = \{(\boldsymbol{x}_1,y_1), (\boldsymbol{x}_2,y_2), ..., (\boldsymbol{x}_N,y_N)\}$$，可以利用“最大似然法”来估计参数$\boldsymbol{w}$、$b$。对数似然函数为：

$$LL(\boldsymbol{w}, b) = \sum_{i=1}^N \ln (y_i \mid \boldsymbol{x}_i; \boldsymbol{w}, b)$$

以上讨论的都是二分类问题，逻辑回归模型也可用于多分类问题。设离散型随机变量$Y$的取值集合为：$$\{1,2,...,K\}$$，则定义对数几率：

$$\ln\frac{p(Y=k \mid \boldsymbol{x})}{p(Y=K \mid \boldsymbol{x})} = \boldsymbol{w}_k^T\boldsymbol{x}+b_k ,\ k=1,2,...,K-1$$

于是可得到：

$$p(Y=k \mid \boldsymbol{x}; \boldsymbol{W}, \boldsymbol{b}) = \frac{e^{\boldsymbol{w}_k^T\boldsymbol{x}+b_k}}{1+\sum_{k'=1}^{K-1}e^{\boldsymbol{w}_{k'}^T\boldsymbol{x}+b_{k'}}} ,\ k=1,2,...,K-1$$

$$p(Y=K \mid \boldsymbol{x}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{1+\sum_{k'=1}^{K-1}e^{\boldsymbol{w}_{k'}^T\boldsymbol{x}+b_{k'}}}$$

其参数估计方法类似二分类逻辑回归模型。


# 3、线性判别分析

线性判别分析（Linear Discriminant Analysis, 简称LDA）的思想非常朴素：训练时，设法将训练样例投影到一条直线上，使得同类样例的投影点尽可能接近，异类样例的投影点尽可能远离；预测时，将预测样本投影到同样的这条直线上，根据投影点的位置来确定它的类别。

考虑二分类问题，训练数据集为$$D = \{(\boldsymbol{x}_1,y_1), (\boldsymbol{x}_2,y_2), ..., (\boldsymbol{x}_N,y_N)\}, \boldsymbol{x}_i \in \mathcal{X} \subseteq \mathbb{R}^d, y_i \in \mathcal{Y} = \{0,1\}, i =1,2,...,N$$。

进一步，令$X^{(k)}$、$\boldsymbol{\mu}^{(k)}$、$$\boldsymbol{\Sigma}^{(k)}$$分别表示第$$k\in \{0,1\}$$类样例的集合、均值向量（维度为$d$）、协方差矩阵（维度为$d\times d$）。

假定要学习的直线为$$y=\boldsymbol{w}^T\boldsymbol{x}$$（这里省略了偏置$b$，因为考察的是样本点在该直线上的投影，可以令该直线总是经过原点，即$b=0$），则两类样本的中心在直线上的投影分别为$$\boldsymbol{w}^T\boldsymbol{\mu}^{(0)}$$和$$\boldsymbol{w}^T\boldsymbol{\mu}^{(1)}$$，投影的方差分别为$$\boldsymbol{w}^T\boldsymbol{\Sigma}^{(0)}\boldsymbol{w}$$和$$\boldsymbol{w}^T\boldsymbol{\Sigma}^{(1)}\boldsymbol{w}$$

欲使同类样例的投影点尽可能接近，可以让同类样例投影点的方差尽可能小，即$$\boldsymbol{w}^T\boldsymbol{\Sigma}^{(0)}\boldsymbol{w} + \boldsymbol{w}^T\boldsymbol{\Sigma}^{(1)}\boldsymbol{w}$$；而欲使异类样例的投影点尽可能远离，可以让两个类中心之间的距离尽可能大，即$$\|\boldsymbol{w}^T\boldsymbol{\mu}^{(0)} - \boldsymbol{w}^T\boldsymbol{\mu}^{(1)}\|_2^2$$尽可能大。同时考虑二者，可得到最大化目标为：

$$J = \frac{\|\boldsymbol{w}^T\boldsymbol{\mu}^{(0)} - \boldsymbol{w}^T\boldsymbol{\mu}^{(1)}\|_2^2}{\boldsymbol{w}^T\boldsymbol{\Sigma}^{(0)}\boldsymbol{w} + \boldsymbol{w}^T\boldsymbol{\Sigma}^{(1)}\boldsymbol{w}} = \frac{\boldsymbol{w}^T(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})^T\boldsymbol{w}}{\boldsymbol{w}^T(\boldsymbol{\Sigma}^{(0)} + \boldsymbol{\Sigma}^{(1)})\boldsymbol{w}}$$

定义“类内散度矩阵”（within-class scatter matrix）：

$$\boldsymbol{S}_w = \boldsymbol{\Sigma}^{(0)} + \boldsymbol{\Sigma}^{(1)}$$

以及“类间散度矩阵”（between-class scatter matrix）：

$$\boldsymbol{S}_b = (\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})^T$$

它是向量$$\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)}$$与自身的外积。

则LDA的最大化目标为：

$$J = \frac{\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}}{\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w}}$$

$J$也称为$$\boldsymbol{S}_b$$和$$\boldsymbol{S}_w$$的“广义瑞利商”（generalized Rayleigh quotient）。

由于$J$的分子、分母都是关于$\boldsymbol{w}$的二次项，因此上式的解与$\boldsymbol{w}$的长度无关，只与其方向有关。不失一般性，令$$\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w} = 1$$，则优化问题可等价为：

$$\begin{align} \min_{\boldsymbol{w}} & -\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w} \\
s.t.\ & \boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w} = 1
\end{align}$$

应用拉格朗日乘子法，可得：

$$\boldsymbol{S}_b\boldsymbol{w} = \lambda \boldsymbol{S}_w\boldsymbol{w}$$

令$$\lambda_{\boldsymbol{w}} = (\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})^T\boldsymbol{w}$$，则可得：

$$\boldsymbol{S}_b\boldsymbol{w} = (\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})^T\boldsymbol{w} = \lambda_{\boldsymbol{w}}(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)}) = \lambda \boldsymbol{S}_w\boldsymbol{w}$$

由于该问题与$\boldsymbol{w}$的长度无关，可令$$\lambda_{\boldsymbol{w}} = \lambda$$，于是得：

$$\boldsymbol{w} = \boldsymbol{S}_w^{-1}(\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^{(1)})$$

上述讨论的是而分类问题，LDA也可以推广到多分类问题中。假设存在$K$个类，集合$X^{(k)}$中样例数为$$n_k$$，则显然是有：$$\sum_{k=1}^K n_k = N$$。

于是，可得类内散度矩阵为：

$$\boldsymbol{S}_w = \sum_{k=1}^K\boldsymbol{\Sigma}^{(k)}$$

由于不止两个类中心点，不能简单套用二分类LDA的做法，用两个中心点的距离来度量类间散度矩阵。我们可以考虑用每一类样本集的中心点距总的中心点的距离作为度量。考虑到每一类样本集的大小可能不同（密度分布不均），故我们对这个距离加以权重。因此，定义类间散度矩阵为：

$$\boldsymbol{S}_b = \sum_{k=1}^K n_k(\boldsymbol{\mu}^{(k)}-\boldsymbol{\mu})(\boldsymbol{\mu}^{(k)}-\boldsymbol{\mu})^T$$

其中$$\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \boldsymbol{x}_i$$表示所有样本上的均值向量。

设$\boldsymbol{W} \in \mathbb{R}^{d\times (K-1)}$是投影矩阵，则常用的一种优化目标是：

$$\max_{\boldsymbol{W}} J = \frac{tr(\boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W})}{tr(\boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W})}$$

其中$$tr(\cdot)$$表示矩阵的迹（trace），即矩阵对角线的元素之和，它是一个矩阵不变量，也等于所有特征值之和（还有一个常用的矩阵不变量是矩阵的行列式，它等于所有特征值之积）。
