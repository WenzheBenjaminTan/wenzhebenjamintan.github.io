---
layout: post
title: "信息论" 
---

# 1、基本原理

信息论的基本原理是：从不太可能发生的事件中能学到更多的有用信息。主要包含以下三层意思：

1）发生可能性较大的事件包含较少的信息；

2）发生可能性较小的事件包含较多的信息；

3）独立事件包含额外的信息。

# 2、自信息（self information）

对于事件$X = x$，定义该事件的自信息为：$$I(X = x) = -\log p(x)$$。可以看出，该事件的可能性越小（不确定性越大），自信息越大。

# 3、熵（entropy）

自信息仅仅是处理单个事件的输出，如果是计算某类事件的自信息期望，它就是熵（定义了该类事件的不确定性）：

$$Ent(X) = \mathbb{E}_{x\sim p(x)}[I(X = x)] = \mathbb{E}_{x\sim p(x)}[-\log p(x)] = -\sum_xp(x)\log p(x)$$

熵刻画了按照真实分布$p$来识别一个样本所需要的编码长度期望（即平均编码长度）。如，含有4个字母$$\{A,B,C,D\}$$，真实分布$$p=\{\frac{1}{2},\frac{1}{4},\frac{1}{4},0\}$$，则识别样本需要的平均编码长度为1.5。

对于离散型随机变量$X$，假设其取值集合大小为$K$，则容易证明：$$0\leq Ent(X) \leq \log K$$。

# 4、条件熵（conditional entropy）

对于随机变量$X$和$Y$，条件熵$$Ent(Y\mid X)$$定义为给定$X$条件下$Y$的条件概率分布的熵对$X$的期望：

$$Ent(Y\mid X) = \mathbb{E}_{x\sim p(x)}[Ent(Y\mid X = x)] = -\sum_x\sum_y p(x,y)\log p(y\mid x)$$

容易证明：$$Ent(X,Y) = Ent(X) + Ent(Y\mid X)$$

即描述$X$和$Y$所需要的信息是描述$X$所需要的信息加上给定$X$条件下描述$Y$所需要的额外信息的和。


# 5、相对熵（relative entropy）

相对熵又被称为KL散度（Kullback-Leibler divergence）或信息散度（information divergence），是两个概率分布间差异的非对称性度量。

假设两个概率分布分别为$p$和$q$，则KL散度定义为：

$$KL (p\| q) = \mathbb{E}_{x\sim p(x)}\left[\log\frac{p(x)}{q(x)}\right] = \sum_xp(x)\log\frac{p(x)}{q(x)}$$

KL散度具有如下两个性质：

1）KL散度非负：当它为0时，当且仅当$p$和$q$是同一个分布（对于离散型随机变量），或者两个分布几乎处处相等（对于连续型随机变量）；

2）KL散度不对称：$$KL (p\| q) \neq KL (q\| p)$$。

# 6、交叉熵（cross entropy）

交叉熵用于刻画使用错误分布$q$来表示真实分布$p$中的样本的平均编码长度：

$$Ent(p,q) = \mathbb{E}_{x\sim p(x)}\left[-\log q(x)\right] = -\sum_xp(x)\log q(x) = Ent(p) + KL (p\| q)$$

从上式可以看出$$KL (p\| q)$$刻画了使用错误分布$q$编码真实分布$p$带来的平均编码长度的增量。





