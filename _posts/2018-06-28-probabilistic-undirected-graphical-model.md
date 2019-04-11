---
layout: post
title: "概率无向图模型" 
---

# 1、模型定义

机器学习最重要的任务，是根据一些已观察到的证据（例如训练样本）来对感兴趣的未知变量（例如类别标记）进行估计。概率模型（probabilistic model）提供了一种描述框架，将学习任务归结于计算变量的概率分布。在概率模型中，利用已知变量估计未知变量的分布称为“推断”（inference），其核心是如何基于可观测变量推测出未知变量的条件分布。具体来说，假定可观测变量集合为$X$，所关心的变量集合为$Y$，其他变量集合为$O$，“生成式”（generative）模型考虑联合分布$p(Y,O,X)$，而“判别式”（discriminative）模型考虑条件分布$p(Y,O\mid X)$。给定一组观测变量值，推断就是要由$p(Y,O,X)$或$p(Y,O\mid X)$得到条件概率分布$p(Y\mid X)$。

属性变量之间往往存在复杂的联系，为了便于研究高效的推断和学习算法，需有一套简洁紧凑地表达变量间关系的工具。

概率无向图模型（probabilistic undirected graphical model）是一类用无向图来表达变量相关关系的概率模型。它以“变量关系图”为表示工具，最常见的使用一个节点表示一个或一组随机变量，节点之间的边表示变量间的概率相关关系。


# 2、马尔可夫随机场

马尔可夫随机场（Markov Random Field, MRF）是典型的生成式无向图模型。图中每个节点表示一个或一组变量，节点之间的边表示两个变量之间的依赖关系。由于没有方向，因此没有边的头尾之分，我们不提及父节点或子女节点，而是讨论团（clique）。对于图中节点的一个子集，如果其中任意两节点之间都有边连接，则称该节点子集为一个团。若在一个团中加入另外任何一个节点都不再形成团，则称该团为“极大团”（maximal clique）；换言之，极大团就是不能再被其他团所包含的团。显然，每个节点至少出现在一个极大团中。

在无向图中我们定义了势函数（potential function）$$\psi_{Q}(\boldsymbol{x}_Q)$$，其中$$\boldsymbol{x}_Q$$是团$Q$中变量的集合。并且，我们定义所有变量的联合分布为图中极大团的势函数乘积：

$$\begin{equation}p(\boldsymbol{x}) = \frac{1}{Z}\prod_Q \psi_{Q}(\boldsymbol{x}_Q)\label{2}\end{equation}$$

其中$$Z = \sum_{\boldsymbol{x}}\prod_Q \psi_{Q}(\boldsymbol{x}_Q)$$是规范化常数，以确保$$\sum_{\boldsymbol{x}}p(\boldsymbol{x})=1$$。在实际应用中精确计算$Z$通常很困难，但许多任务往往不需要获得$Z$的精确值。像上式这样，通过归一化团势能乘积定义的分布也被称作吉布斯分布（Gibbs distribution）。

在马尔可夫随机场中，可借助“分离”的概念来获得“条件独立性”。若从节点集$A$中的节点到$B$中的节点都必须经过节点集$C$中的节点，则称节点集$A$和$B$被节点集$C$分离，$C$称为“分离集”（separating set）。

对马尔可夫随机场，有“全局马尔可夫性”（global Markov preperty）：给定两个变量子集的分离集，则这两个变量子集条件独立。

也就是说，若令上例中$A$、$B$和$C$对应的变量集分别为$$\boldsymbol{x}_A$$、$$\boldsymbol{x}_B$$和$$\boldsymbol{x}_C$$，则$$\boldsymbol{x}_A$$和$$\boldsymbol{x}_B$$在给定$$\boldsymbol{x}_C$$的条件下独立，记为$$\boldsymbol{x}_A\bot \boldsymbol{x}_B \mid \boldsymbol{x}_C$$。

势函数$$\psi_{Q}(\boldsymbol{x}_Q)$$的作用是定量刻画变量集$$\boldsymbol{x}_Q$$中变量之间的关系，它应该是非负函数，且在所偏好的变量取值上有较大的函数值。

为了满足非负性，指数函数常被用于定义势函数，即

$$\psi_{Q}(\boldsymbol{x}_Q) = e^{-H_Q(\boldsymbol{x}_Q)}$$

其中$$H_Q(\boldsymbol{x}_Q)$$是一个定义在变量$$\boldsymbol{x}_Q$$上的实值函数，常见形式为：

$$H_Q(\boldsymbol{x}_Q) = \sum_{u,v\in Q,u\neq v}\alpha_{uv}x_ux_v + \sum_{v\in Q}\beta_vx_v$$

其中$$\alpha_{uv}$$和$$\beta_v$$是参数。上式中的第二项仅考虑单节点，第一项则考虑每一对节点的关系。

此时，我们可以将得到某一个状态$\boldsymbol{x}$的概率写为：

$$p(\boldsymbol{x}) = \frac{1}{Z} e^{-\sum_{Q}H_Q(\boldsymbol{x}_Q)} = \frac{1}{Z} e^{-E(\boldsymbol{x})}$$

服从这种形式的$E(\boldsymbol{x})$称作能量函数（energy function），相应的模型称作基于能量的模型，又称为玻尔兹曼机（Boltzmann Machine），$\boldsymbol{x}$的分布称作玻尔兹曼分布（Boltzmann distribution）。


# 3、条件随机场

条件随机场（Conditional Random Field, CRF）是一种判别式无向图模型。前面提到过，生成式模型是直接对联合分布进行建模，而判别式模型则是对条件分布进行建模。

条件随机场试图对多个变量在给定观测值后的条件概率进行建模。具体来说，若令$$\boldsymbol{x} = \{x_1,x_2,...,x_n\}$$为观测序列，$$\boldsymbol{y} = \{y_1,y_2,...,y_n\}$$为与之相应的标记序列，则条件随机场的目标是构建条件概率模型$$p(\boldsymbol{y}\mid \boldsymbol{x})$$。需注意的是，标记变量$\boldsymbol{y}$可以是结构型变量，即其分量之间具有某种相关性。

令$$G=< V,E > $$表示节点与标记变量$\boldsymbol{y}$中元素一一对应的无向图，$$y_v$$表示与节点$v$对应的标记变量，$n(v)$表示节点$v$的邻接节点集合，若图$G$的每个变量$$y_v$$都满足马尔可夫性，即

$$p(y_v\mid \boldsymbol{x}, \boldsymbol{y}_{V\backslash\{v\}}) = p(y_v\mid \boldsymbol{x}, \boldsymbol{y}_{n(v)})$$

则$$(\boldsymbol{y},\boldsymbol{x})$$构成一个条件随机场。

理论上来说，图$G$可具有任意结构，只要能标记变量之间的条件独立性关系即可。但在现实应用中，尤其是对标记序列建模时，最常用的仍是链式结构，即“链式条件随机场”（chain-structured CRF）。下面我们主要讨论这种条件随机场。

与马尔可夫随机场定义联合概率的方式类似，条件随机场使用势函数和图结构上的团来定义条件概率$$p(\boldsymbol{y}\mid \boldsymbol{x})$$。在链式条件随机场中，通过选用指数势函数并引入特征函数（feature function），条件概率被定义为：

$$\begin{equation}p(\boldsymbol{y}\mid\boldsymbol{x}) = \frac{1}{Z}\exp\left(\sum_j\sum_{i=1}^{n-1}\lambda_jt_j(y_{i+1},y_i,\boldsymbol{x},i) + \sum_k\sum_{i=1}^n\mu_ks_k(y_i,\boldsymbol{x},i)\right)\label{3}\end{equation}$$

其中$$t_j(y_{i+1},y_i,\boldsymbol{x},i)$$是定义在观测序列的两个相邻标记位置上的转移特征函数（transition feature function），用于刻画相邻标记变量之间的相关关系以及观测序列对它们的影响，$$s_k(y_i,\boldsymbol{x},i)$$是定义在观测序列上的标记位置$i$上的状态特征函数（status feature function），用于刻画观测序列对标记变量的影响，$$\lambda_j$$和$$\mu_k$$是参数，$$Z$$为规范化常数。

对比式\eqref{3}和\eqref{2}可看出，条件随机场和马尔可夫随机场均使用团上的势函数定义概率，两者在形式上没有显著区别；但条件随机场处理的是条件概率，而马尔可夫随机场处理的是联合概率。



# 4、信念传播算法

信念传播（belief propagation）算法是一种精确推断变量边际概率的方法。其中从变量$$x_i$$向$$x_j$$传播的信念$$m_{ij}(x_j)$$定义为：

$$m_{ij}(x_j) = \sum_{x_i}\psi(x_i,x_j)\prod_{k\in n(i)\backslash j} m_{ki}(x_i)$$

不难看出，每次信念传播过程仅与变量$$x_i$$及其邻接节点直接相关，换言之，信念传播相关的计算被限制在图的局部进行。

在信念传播算法中，一个节点仅在接收到来自其他所有节点的信念后才能向另一个节点发送信念，其节点的边际分布正比于它所接收到所有信念的乘积，即

$$\begin{equation}p(x_i) \propto \prod_{k\in n(i)}  m_{ki}(x_i) \label{4}\end{equation}$$

若图结构中没有环，则信念传播算法经过两个步骤即可完成所有信念传播，进而能计算所有变量上的边际分布：

1）指定一个根节点，从所有叶节点开始向根节点传播信念，直到根节点收到所有邻接节点的信念；

2）从根节点开始向叶节点传播信念，直到所有叶节点均收到信念。

此时图的所有边上都有方向不同的两条信念传播，基于这些信念和式\eqref{4}即可获得所有变量的边际概率。


