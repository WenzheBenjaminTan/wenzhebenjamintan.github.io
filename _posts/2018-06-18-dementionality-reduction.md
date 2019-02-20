---
layout: post
title: "降维" 
---

# 1、引言

在大多数学习算法中，复杂度依赖于输入的维度$d$和数据样本的规模$N$。为了减少存储空间和计算时间，我们对降低输入的维度感兴趣。降低维度的主要方法有两种：特征选择和特征提取。

在特征选择（feature selection）中，我们感兴趣的是从$d$维中找出为我们提供最多信息的$k$个维，并且丢弃其他的$d-k$个维。

在特征提取（feature extraction）中，我们感兴趣的是将原来的$d$个维进行重新组合，得到新的$k$个维的集合。特征提取可以是监督的，也可以是无监督的，这取决于该方法是否使用输出信息。最广泛使用的特征提取方法是主成分分析（principal components analysis, PCA）和线性判别分析（linear discriminant analysis, LDA）。它们都是线性投影方法，分别是无监督和监督的。

# 2、特征选择

特征选择的一个典型代表就是子集选择（subset selection）。对于发现的特征集中，我们选择一个最佳子集，使得其包含的维数最少，且对正确率（需要引入一个误差函数）的贡献最大。

$d$个变量有$2^d$个可能的子集，除非$d$很小，否则我们很难对所有子集进行检验。如果使用启发式方法，在合理的（多项式）时间内得到一个合理的（但不是最优的）解是可能的。

一般采用两种方法：前向选择（forward selection）和后向选择（backward selection）。在前向选择中，我们从空集开始，逐个添加变量，每次添加一个降低误差最多的变量，直到进一步的添加不回降低误差（或降低很少）。在后向选择中，我们从所有变量开始，逐个排除它们，每次排除一个降低误差最多（或提高很少）的变量，直到进一步的排除不会显著提高误差。在这两种情况下，误差检测都应该在不同于训练集的验证集上进行，因为我们想要检验泛化准确率。使用更多的特征，我们一般会有更低的训练误差，但是不一定有更低的验证误差。

我们用$F$表示输入维的特征$$x_i (i=1,2,...,d)$$ 的集合，$E(F)$ 表示当我们只使用$F$中的输入维时，在验证样本上出现的误差。依赖于应用（回归或分类），误差或者是均方误差，或者是误分类错误率。

在顺序前向选择（sequential forward selection）中，我们从$$F = \varnothing$$开始。每一步中，我们针对所有可能的$$x_i$$，训练我们的模型并在验证集上计算$$E(F\cup x_i)$$。然后，我们选择导致最小误差的输入$$x_j$$：

$$j = arg\min_i E(F\cup x_i)$$

如果$$E(F\cup x_j) < E(F)$$，则将$$x_j$$添加到$F$中，否则停止。如果误差降低太小，我们甚至可以决定提前停止；这里存在一个事先定义的阈值，依赖于应用约束以及错误和复杂度的折中。

这样的过程也许开销很大，因为将$d$维减少到$k$维，我们需要训练和测试系统$$d + (d-1) + (d-2) + \cdots + (d-k)$$次，其复杂度为$$O(d^2)$$。这是一个局部搜索过程，并不能保证找到最佳子集。例如，$$x_i$$和$$x_j$$本身可能都不好，但是合起来却可能会把误差降低很多，但是该算法是逐个增加特征，因此也许不能发现$$x_i$$和$$x_j$$的并。以更多计算为代价，一次增加$m$个而不是一个特征是可能的。

在顺序后向选择（sequential backward selection）中，我们从包括所有特征的$F$开始，并且执行类似的过程，我们每次去掉一个特征，并且是去掉导致误差最小的那个$$x_j$$：

$$j = arg\min_i E(F\setminus x_i)$$

如果$$E(F\setminus x_j) < E(F)$$，则将$$x_j$$从$F$中去掉，否则停止。为了尽可能降低复杂度，如果误差增加不大，我们可能也会决定去掉该特征。

前向搜索的所有可能变体对于后向搜索都是可行的。后向搜索与前向搜索具有相同的复杂度，但是训练具有较多特征的系统且我们预料有许多无用特征时，前向搜索一般更可取。

子集选择是监督的，因为需要输出来被回归器或分类器用作计算误差，它可以用于任何回归和分类方法。在多元正态分类的特殊情况下，如果原来的$d$维类密度是多元正态的，则其任意子集也是多元正态的，仍然可以使用参数分类，并具有用$k\times k$维协方差矩阵代替$d\times d$维协方差矩阵的优点。

在像人脸识别这样的应用中，特征选择不是降维的好方法，因为个体像素值本身并不携带很多识别信息；携带脸部识别信息的是许多像素值的组合。后续要介绍的特征提取方法是一个更好的选择。


# 3、特征提取

## 3.1 主成分分析

向量$\boldsymbol{x}$在方向$\boldsymbol{w}$上的投影为：

$$z = \boldsymbol{w}^T\boldsymbol{x}$$

在投影方法中，我们希望找到一个从原$d$维输入空间到新的$k (k < d)$维空间的、具有最小信息损失的映射。

我们假设$$\boldsymbol{x}\sim \mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$$，其中$\Sigma$是$\boldsymbol{x}$的协方差矩阵。主成分是这样的$$\boldsymbol{w}_1$$，样本投影到$$\boldsymbol{w}_1$$上之后差别变得最明显（即方差最大）。为了得到唯一解，我们要求$$\|\boldsymbol{w}_1\| = 1$$。因为$$z_1 = \boldsymbol{w}_1^T\boldsymbol{x}$$，并且$$Cov(\boldsymbol{x}) = \Sigma$$，则

$$Var(z_1) = \boldsymbol{w}_1^T\Sigma\boldsymbol{w}_1$$

这变成一个优化问题，即寻找$$\boldsymbol{w}_1$$，使得$$Var(z_1)$$受限于约束$$\boldsymbol{w}_1^T\boldsymbol{w}_1 = 1$$最大化。其拉格朗日形式为：

$$\max_{\boldsymbol{w}_1} \boldsymbol{w}_1^T\Sigma\boldsymbol{w}_1 - \alpha(\boldsymbol{w}_1^T\boldsymbol{w}_1 - 1)$$

对目标函数关于$$\boldsymbol{w}_1$$求导并令其等于0，于是得到一阶必要条件：

$$2\Sigma\boldsymbol{w}_1 - 2\alpha\boldsymbol{w}_1 = 0$$

因此

$$\Sigma\boldsymbol{w}_1 = \alpha\boldsymbol{w}_1$$

当且仅当$$\boldsymbol{w}_1$$是$$\Sigma$$的特征向量，$\alpha$是对应的特征值时上式成立。进而我们有：

$$\boldsymbol{w}_1^T\Sigma\boldsymbol{w}_1 = \alpha \boldsymbol{w}_1^T\boldsymbol{w}_1 = \alpha$$

为了使方差最大，我们选择具有最大特征值的特征向量。

因此，主成分是输入样本的协方差矩阵具有最大特征值$$\lambda_1 = \alpha$$的特征向量。

第二个主成分$$\boldsymbol{w}_2$$也应该最大化方差，具有单位长度，并且与$$\boldsymbol{w}_1$$正交。后者的要求是使得投影后$$z_2 = \boldsymbol{w}_2^T\boldsymbol{x}$$与$z_1$不相关。

对于第二个主成分，我们可以得到优化问题：

$$\max_{\boldsymbol{w}_2} \boldsymbol{w}_2^T\Sigma\boldsymbol{w}_2 - \alpha(\boldsymbol{w}_2^T\boldsymbol{w}_2 - 1) - \beta(\boldsymbol{w}_2^T\boldsymbol{w}_1 - 0)$$

一阶必要条件为：

$$\begin{equation}\label{1}
2\Sigma\boldsymbol{w}_2 - 2\alpha\boldsymbol{w}_2 - \beta\boldsymbol{w}_1 = 0
\end{equation}$$

用$$\boldsymbol{w}_1^T$$左乘，可得：

$$2\boldsymbol{w}_1^T\Sigma\boldsymbol{w}_2 - 2\alpha\boldsymbol{w}_1^T\boldsymbol{w}_2 - \beta\boldsymbol{w}_1^T\boldsymbol{w}_1 = 0$$

注意到$$\boldsymbol{w}_1^T\boldsymbol{w}_2 = 0$$，$$\boldsymbol{w}_1^T\boldsymbol{w}_1 = 1$$。$$\boldsymbol{w}_1^T\Sigma\boldsymbol{w}_2$$是标量，于是有

$$\boldsymbol{w}_1^T\Sigma\boldsymbol{w}_2 = \boldsymbol{w}_2^T\Sigma\boldsymbol{w}_1 = \lambda_1\boldsymbol{w}_2^T\boldsymbol{w}_1 = 0$$

进而可得$$\beta = 0$$，并且式\eqref{1}可以简化为：

$$\Sigma\boldsymbol{w}_2 = \alpha\boldsymbol{w}_2$$

这表明$$\boldsymbol{w}_2$$也是$\Sigma$的特征向量，对应第二大特征值$$\lambda_2 = \alpha$$。类似地，我们可以证明其他维被具有递减的特征值的特征向量给出。

因为$\Sigma$是对称的，因此对于两个不同的特征值，特征向量是正交的。如果$\Sigma$是正定的（即对于所有非空$\boldsymbol{x}$，有$$\boldsymbol{x}^T\Sigma\boldsymbol{x} > 0$$），则它所有的特征值都为正。如果$\Sigma$是奇异的，则它的秩（有效维数）为$k$，并且$k < d$，$$\lambda_i (i=k+1,...,d)$$均为0（$$\lambda_i$$为递减序）。$k$个具有非零特征值的特征向量是约化空间的维。第一个特征向量$$\boldsymbol{w}_1$$贡献了方差的最大部分，第二个贡献了第二大部分，依次类推。

我们定义

$$\boldsymbol{z} = \boldsymbol{W}^T(\boldsymbol{x} - \boldsymbol{m})$$

其中$$\boldsymbol{W}$$的$k$列是$\boldsymbol{S}$（$\Sigma$的估计）的$k$个主特征向量。我们在$\boldsymbol{x}$投影前减去其均值$\boldsymbol{m}$（$\boldsymbol{\mu}$的估计），将数据在原点中心化。线性投影后，我们得到$k$维空间，它的维是特征向量，并且在这些新维上的方差等于特征值。$$Cov(\boldsymbol{z}) = \boldsymbol{W}^T\boldsymbol{S}\boldsymbol{W} = \boldsymbol{D}$$，是一个对角阵，对角元素分别是$k$个递减的特征值；也就是说，不同的$$z_i$$之间是不相关的。

因为$$\boldsymbol{W}\boldsymbol{W}^T = \boldsymbol{I}_d$$，因此可以得到

$$\boldsymbol{S} = \boldsymbol{W}\boldsymbol{D}\boldsymbol{W}^T$$

这称为$\boldsymbol{S}$的谱分解（spectral decomposition）。

我们已经知道，如果$$\boldsymbol{x}\sim \mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$$，则投影后$$\boldsymbol{W}^T\boldsymbol{x}\sim \mathcal{N}_k(\boldsymbol{W}^T\boldsymbol{\mu}, \boldsymbol{W}^T\Sigma\boldsymbol{W})$$，即将$d$元正态的样本投影到$k$元正态上。实例$$\boldsymbol{x}^{(i)}$$投影到$\boldsymbol{z}$-空间为：

$$\boldsymbol{z}^{(i)} = \boldsymbol{W}^T(\boldsymbol{x}^{(i)} - \boldsymbol{\mu})$$

它可以逆投影到原来的空间：

$$\widehat{\boldsymbol{x}}^{(i)} = \boldsymbol{W}\boldsymbol{z}^{(i)} + \boldsymbol{\mu}$$

$$\widehat{\boldsymbol{x}}^{(i)}$$是$$\boldsymbol{x}^{(i)}$$从它在$\boldsymbol{z}$-空间中的表示的重构。我们定义重构误差（reconstruction error）为实例与它的从低维空间重构之间的距离：

$$\sum_i \|\widehat{\boldsymbol{x}}^{(i)} - \boldsymbol{x}^{(i)}\|^2$$

重构误差取决于我们考虑了多少个主成分。

PCA并不利用输出信息，因此它是非监督的。


## 3.2 线性判别分析

线性判别分析（Linear Discriminant Analysis, LDA）是一种用于分类问题的维度归约的监督方法。

首先考虑二分类问题，训练数据集为：

$$D = \{(\boldsymbol{x}^{(1)},y^{(1)}), (\boldsymbol{x}^{(2)},y^{(2)}), ..., (\boldsymbol{x}^{(N)},y^{(N)})\}, \boldsymbol{x}^{(i)} \in \mathcal{X} \subseteq \mathbb{R}^d, y^{(i)} \in \mathcal{Y} = \{0,1\}, i =1,2,...,N$$

进一步，令$$X_c$$、$$\boldsymbol{m}_c$$、$$\boldsymbol{S}_c$$分别表示第$$c\in \{0,1\}$$类样例的集合、均值向量（维度为$d$）、协方差矩阵估计（维度为$d\times d$）。

LDA的思想非常朴素：训练时，设法将训练样例投影到一条直线上，使得同类样例的投影点尽可能接近，异类样例的投影点尽可能远离；预测时，将预测样本投影到同样的这条直线上，根据投影点的位置来确定它的类别。


假定要学习的直线为$$y=\boldsymbol{w}^T\boldsymbol{x}$$（这里省略了偏置$b$，因为考察的是样本点在该直线上的投影，可以令该直线总是经过原点，即$b=0$），则两类样本的中心在直线上的投影分别为$$\boldsymbol{w}^T\boldsymbol{m}_0$$和$$\boldsymbol{w}^T\boldsymbol{m}_1$$，投影的方差估计分别为$$\boldsymbol{w}^T\boldsymbol{S}_0\boldsymbol{w}$$和$$\boldsymbol{w}^T\boldsymbol{S}_1\boldsymbol{w}$$。该投影过程实际上是从$d$维到1维的维度归约。

欲使同类样例的投影点尽可能接近，可以让同类样例投影点的方差，即$$\boldsymbol{w}^T\boldsymbol{S}_0\boldsymbol{w} + \boldsymbol{w}^T\boldsymbol{S}_1\boldsymbol{w}$$尽可能小；而欲使异类样例的投影点尽可能远离，可以让两个类中心之间的距离，即$$\|\boldsymbol{w}^T\boldsymbol{m}_0 - \boldsymbol{w}^T\boldsymbol{m}_1\|_2^2$$尽可能大。同时考虑二者，可得到最大化目标为：

$$J(\boldsymbol{w}) = \frac{\|\boldsymbol{w}^T\boldsymbol{m}_0 - \boldsymbol{w}^T\boldsymbol{m}_1\|_2^2}{\boldsymbol{w}^T\boldsymbol{S}_0\boldsymbol{w} + \boldsymbol{w}^T\boldsymbol{S}_1\boldsymbol{w}} = \frac{\boldsymbol{w}^T(\boldsymbol{m}_0 - \boldsymbol{m}_1)(\boldsymbol{m}_0 - \boldsymbol{m}_1)^T\boldsymbol{w}}{\boldsymbol{w}^T(\boldsymbol{S}_0 + \boldsymbol{S}_1)\boldsymbol{w}}$$

定义“类内散度矩阵”（within-class scatter matrix）：

$$\boldsymbol{S}_w = \boldsymbol{S}_0 + \boldsymbol{S}_1$$

以及“类间散度矩阵”（between-class scatter matrix）：

$$\boldsymbol{S}_b = (\boldsymbol{m}_0 - \boldsymbol{m}_1)(\boldsymbol{m}_0 - \boldsymbol{m}_1)^T$$

它是向量$$\boldsymbol{m}_0 - \boldsymbol{m}_1$$与自身的外积。

则LDA的最大化目标为：

$$J(\boldsymbol{w}) = \frac{\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}}{\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w}}$$

$J$也称为$$\boldsymbol{S}_b$$和$$\boldsymbol{S}_w$$的“广义瑞利商”（generalized Rayleigh quotient）。

由于$J$的分子、分母都是关于$\boldsymbol{w}$的二次项，因此上式的解与$\boldsymbol{w}$的长度无关，只与其方向有关。不失一般性，令$$\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w} = 1$$，则优化问题可等价为：

$$\begin{align} \min_{\boldsymbol{w}} & -\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w} \\
s.t.\ & \boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w} = 1
\end{align}$$

应用拉格朗日乘子法，可得：

$$\boldsymbol{S}_b\boldsymbol{w} = \lambda \boldsymbol{S}_w\boldsymbol{w}$$

令$$\lambda_{\boldsymbol{w}} = (\boldsymbol{m}_0 - \boldsymbol{m}_1)^T\boldsymbol{w}$$，则可得：

$$\boldsymbol{S}_b\boldsymbol{w} = (\boldsymbol{m}_0 - \boldsymbol{m}_1)(\boldsymbol{m}_0 - \boldsymbol{m}_1)^T\boldsymbol{w} = \lambda_{\boldsymbol{w}}(\boldsymbol{m}_0 - \boldsymbol{m}_1) = \lambda \boldsymbol{S}_w\boldsymbol{w}$$

由于该问题与$\boldsymbol{w}$的长度无关，于是可得：

$$\boldsymbol{w} = \boldsymbol{S}_w^{-1}(\boldsymbol{m}_0 - \boldsymbol{m}_1)$$

我们已经将样本从$d$维降到了1维，之后可以使用任何分类方法来进行分类。

上述讨论的是二分类问题，LDA也可以推广到多分类问题中。假设存在$K$个类，集合$X_c$中样例数为$$n_c$$，则显然是有：$$\sum_{c=1}^K n_c = N$$。

于是，可得类内散度矩阵为：

$$\boldsymbol{S}_w = \sum_{c=1}^K\boldsymbol{S}_c$$

由于不止两个类中心点，不能简单套用二分类LDA的做法，用两个中心点的距离来度量类间散度矩阵。我们可以考虑用每一类样本集的中心点距总的中心点的距离作为度量。考虑到每一类样本集的大小可能不同（密度分布不均），故我们对这个距离加以权重。因此，定义类间散度矩阵为：

$$\boldsymbol{S}_b = \sum_{c=1}^K n_c(\boldsymbol{m}_c-\boldsymbol{m})(\boldsymbol{m}_c-\boldsymbol{\mu})^T$$

其中$$\boldsymbol{m} = \frac{1}{K}\sum_{c=1}^K \boldsymbol{m}_c$$表示总均值。

不失一般性，我们设$\boldsymbol{W} \in \mathbb{R}^{d\times k}$是投影矩阵，即将$d$维实例投影到$k$维。投影后的类间散布矩阵是$$\boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W}$$，类内散布矩阵是$$\boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W}$$，它们都是$k\times k$矩阵。我们希望第一个散布大；也就是说，在投影之后的$k$维空间，我们希望类均值互相之间尽可能远离。我们希望第二个散布大；也就是说，在投影之后的$k$维空间，我们希望来自同一个类的样本尽可能接近它们的均值。对于一个散布（或协方差）矩阵，散布的一个度量是矩阵的行列式（另一种度量是矩阵的迹）。行列式是特征值的乘积，而特征值给出沿着它的特征向量方向的方差。因此，常用的一种优化目标是：

$$\max_{\boldsymbol{W}} J(\boldsymbol{W}) = \frac{|\boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W}|}{|\boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W}|}$$

它的解是$$\boldsymbol{S}_w^{-1}\boldsymbol{S}_b$$的按特征值递减排序的特征向量构成的矩阵。

因为$$\boldsymbol{S}_b$$是$K$个秩为1的矩阵$$(\boldsymbol{m}_c-\boldsymbol{m})(\boldsymbol{m}_c-\boldsymbol{m})^T$$之和，并且它们之中只有$K-1$个是独立的，所以$$\boldsymbol{S}_b$$具有最大可能秩为$K-1$。通常情况下，可以取$k = K-1$。这样，我们定义新的、较低的$K-1$维空间，然后在那里构造判别式。当然，可以用任意的分类方法来估计判别式。

我们还看到，为了使用LDA，$$\boldsymbol{S}_w$$应该是可逆的。如果不可逆，我们可以先用PCA消除奇异性，然后再把LDA应用于其结果。
