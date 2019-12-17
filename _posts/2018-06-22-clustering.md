---
layout: post
title: "聚类" 
---

# 1、混合密度模型

混合密度（mixture density）记作

$$p(\boldsymbol{x}) = \sum_{i=1}^K p(\boldsymbol{x}\mid \mathcal{G}_i)p(\mathcal{G}_i)$$

其中$$\mathcal{G}_i$$是混合分支（mixture component），也称为分组（group）或簇（cluster）。$$p(\boldsymbol{x}\mid \mathcal{G}_i)$$是支密度（component density），而$$p(\mathcal{G}_i)$$是混合比例（mixture proportion）。如果支密度服从高斯分布，则该模型称为高斯混合模型（Gaussian Mixture Model）。分支数$K$是超级参数，应当预先指定。

给定实例集和$K$，聚类任务要做的是：第一，估计给定实例所属的分支标号（或属于每个分支的概率）；第二，估计各支密度和混合比例。对于第一个任务中涉及的变量称为聚类问题提前假定的隐藏变量，即每个实例都属于一个特定的分支。

一般来说，完成了第一个任务后，第二个任务是好完成的（或者说第二个问题是比较一般性的），所以对聚类任务来说最主要是完成第一个任务，不过求解过程一般是在第一个任务和第二个任务之间反复迭代。


# 2、$k$-均值聚类

假定我们有数据集$$D = \{\boldsymbol{x}^{(i)}\}_{i=1}^N$$。而且我们以某种方法得到了$k$个参考向量（reference vector）$$\boldsymbol{m}_j, j = 1,...,k$$。给定样本$$\boldsymbol{x}^{(i)}$$，我们可以得到它的簇为：

$$j = arg\min_{\widetilde{j}} \|\boldsymbol{x}^{(i)} - \boldsymbol{m}_{\widetilde{j}}\|$$

当$$\boldsymbol{x}^{(i)}$$用$$\boldsymbol{m}_j$$表示时，相当于对原来的样本进行了重构（reconstruction），存在一个正比于距离$$\|\boldsymbol{x}^{(i)} - \boldsymbol{m}_j\|$$的误差。我们定义总重构误差为：

$$\begin{equation}\label{1}e(\{\boldsymbol{m}_j\}_{j=1}^k\mid D) = \sum_i\sum_j b_j^{(i)}\|\boldsymbol{x}^{(i)} - \boldsymbol{m}_j\|^2\end{equation}$$

其中

$$\begin{equation}\label{2} b_j^{(i)} = \begin{cases}1 & j = arg\min\limits_{\widetilde{j}} \|\boldsymbol{x}^{(i)} - \boldsymbol{m}_{\widetilde{j}}\| \\ 0 & others \end{cases}
\end{equation}$$

最好的参考向量是最小化总重构误差的参考向量。因为$$b_j^{(i)}$$也依赖于$$\boldsymbol{m}_j$$，我们不能解析地求解这个优化问题。对此，我们有一个称作$k$-均值聚类（k-means clustering）的迭代过程：首先，我们以随机初始化的$$\boldsymbol{m}_j$$开始；然后，在每次迭代中，我们先对每个$$\boldsymbol{x}^{(i)}$$，使用式\eqref{2}计算估计标号（estimated labels）$$b_j^{(i)}$$；一旦我们有了这些标号，就可以最小化式\eqref{1}，取它关于$$\boldsymbol{m}_j$$的导数并置0，我们得到

$$\boldsymbol{m}_j = \frac{\sum_ib_j^{(i)}\boldsymbol{x}^{(i)}}{\sum_ib_j^{(i)}}$$

即参考向量被设置为它所代表的所有实例的均值。这是一个迭代过程，因为一旦我们计算了新的$$\boldsymbol{m}_j$$，$$b_j^{(i)}$$也会改变并且需要重新计算，这反过来又影响$$\boldsymbol{m}_j$$。这个两步过程一直重复，直到$$\boldsymbol{m}_j$$收敛。

$k$-均值聚类的一个主要缺点是它是一个局部搜索过程，并且最终的$$\boldsymbol{m}_j$$高度依赖于初始的$$\boldsymbol{m}_j$$。对于初始化，存在各种不同的方法，比如：

1）可简单地随机选择$k$个实例作为初始的$$\boldsymbol{m}_j$$；

2）可以计算所有实例的均值，并将一些小随机向量加到均值上，得到$k$个初始的$$\boldsymbol{m}_j$$；

3）可以计算主成分，将它的值域划分成$k$个相等的区间，将数据划分成$k$个分组，然后取这些分组的均值作为初始$$\boldsymbol{m}_j$$。


# 3、期望最大化算法

在$k$-均值聚类中，我们把聚类看作是寻找最小化总重构误差的参考向量问题。在本节中，我们的方法是尝试直接寻找最大化样本似然的支密度和混合比例参数。使用混合密度模型，样本$$D = \{\boldsymbol{x}^{(i)}\}_{i=1}^N$$的对数似然为：

$$\begin{align} \mathcal{L}(\Theta\mid D) & = \log\prod_i p(\boldsymbol{x}^{(i)}\mid \Theta) \\
					& = \sum_i \log \sum_{j=1}^K p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j)p(\mathcal{G}_j)\end{align}$$

其中$\Theta$是包含支密度$$p(\boldsymbol{x}\mid \mathcal{G}_j)$$和混合比例$$p(\mathcal{G}_j)$$的有效统计量。不过，我们不能解析地求解该参数，而需要借助于迭代优化。

期望最大化（Expectation-Maximization，EM）算法用于包含隐藏变量的参数最大似然估计。要解决的问题涉及两组随机变量，其中一组$D$是可观测的，另一组$Z$是隐藏的。算法的目标是找到参数向量$\Theta$，它最大化可观测变量$D$的对数似然$$\mathcal{L}(\Theta\mid D)$$。由于$$\mathcal{L}(\Theta\mid D)$$无法直接求解，我们对$D$关联附加的隐藏变量（hidden variable）$Z$，并使用二者表示潜在的模型，最大化$D$和$Z$联合分布的对数似然（又称完全对数似然）$$\mathcal{L}_c(\Theta\mid D,Z)$$。显然，$$\mathcal{L}_c(\Theta\mid D,Z)$$也不能直接求解，但我们可以根据$D$和当前估计的参数值$$\Theta_l$$（其中$l$是当前迭代次数）先得到隐变量$Z$的概率分布$p(Z\mid D,\Theta_l)$，进而求得$Z$的期望$E[Z\mid D,\Theta_l]$，这是算法的期望（E）步；然后在最大化（M）步，我们寻找新的参数值$$\Theta_{l+1}$$，它最大化完全对数似然$$\mathcal{L}_c(\Theta\mid D,E[Z\mid D,\Theta_l])$$。于是可得求解步骤：

1）E步：$$E[Z\mid D,\Theta_l] = \sum_z p(Z=z\mid D,\Theta_l)$$；

2）M步：$$\Theta_{l+1} = arg\max\limits_{\Theta}\mathcal{L}_c(\Theta\mid D,E[Z\mid D,\Theta_l])$$。

在聚类问题中，隐藏的变量是数据的分支。我们定义一个指示变量（indicator variable）向量$$\boldsymbol{z}^{(i)} = \{z_1^{(i)}, z_2^{(i)},..., z_k^{(i)}\}$$，其中如果$$\boldsymbol{x}^{(i)}$$属于分支$$\mathcal{G}_j$$，则$$z_j^{(i)} = 1$$，否则$$z_j^{(i)} = 0$$。

1）E步：

$$\begin{align} E(z_j^{(i)} \mid D,\Theta_l) & =  E(z_j^{(i)} \mid \boldsymbol{x}^{(i)},\Theta_l) \text{（样本是独立同分布的）} \\
						& = p(z_j^{(i)} = 1 \mid \boldsymbol{x}^{(i)},\Theta_l) \text{（$z_j^{(i)}$是0/1随机变量）} \\
						& = p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)
\end{align}$$

我们可以看到，隐藏变量的期望值是一个概率值，在0和1之间，与$k$-均值聚类的0/1“硬”标记不同，它是"软"标记。



2）M步：

对于独立同分布的样本集，我们得到完全对数似然

$$\begin{align} \mathcal{L}_c(\Theta\mid D,Z) & = \log \prod_i p(\boldsymbol{x}^{(i)},\boldsymbol{z}^{(i)}\mid \Theta) \\
						& = \sum_i\log p(\boldsymbol{x}^{(i)},\boldsymbol{z}^{(i)}\mid \Theta) \\
						& = \sum_i \log p(\boldsymbol{z}^{(i)}\mid \Theta) + \log p(\boldsymbol{x}^{(i)}\mid \boldsymbol{z}^{(i)},\Theta) \\
						& = \sum_i \sum_j z_j^{(i)}[\log p(\mathcal{G}_j\mid \Theta) + \log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta)] 
\end{align}$$

因此可得

$$\begin{align} \mathcal{L}_c(\Theta\mid D,E[Z\mid D,\Theta_l]) &= \sum_i \sum_j E(z_j^{(i)} \mid D,\Theta_l)[\log p(\mathcal{G}_j\mid \Theta) + \log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta)] \\ 						
							&= \sum_i \sum_j p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)[\log p(\mathcal{G}_j\mid \Theta) + \log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta)]

\end{align}$$

进而

$$\begin{align} \Theta_{l+1} & = arg\max\limits_{\Theta}\mathcal{L}_c(\Theta\mid D,E[Z\mid D,\Theta_l]) \\
				& = arg\max\limits_{\Theta} \sum_i \sum_j p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)[\log p(\mathcal{G}_j\mid \Theta) + \log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta)]
\end{align}$$

对于$\Theta$，它由两个互相独立的部分构成：一部分是跟混合比例有关的，我们可以定义为$$\pi_j = p(\mathcal{G}_j\mid \Theta)$$；另一部分是跟支密度相关的，我们定义它为$\Phi$，于是$$\log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta) = \log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Phi)$$。

对于$$\pi_j$$，我们需要添加约束$$\sum_j\pi_j =1$$，然后利用拉格朗日方法求解：

$$\nabla_{\boldsymbol{\pi}} \sum_i\sum_j p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)\log \pi_j - \lambda(\sum_j\pi_j - 1) = 0$$

得到

$$\pi_j = \frac{\sum_i p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)}{N}$$

对于$\Phi$，我们求解：

$$\nabla_{\Phi} \sum_i \sum_j p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l)\log p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Phi) = 0$$

如果我们假设分支密度为高斯分布，即$$p(\boldsymbol{x}\mid \mathcal{G}_j, \Phi) \sim \mathcal{N}(\boldsymbol{m}_j, \boldsymbol{S}_j)$$，则对于E步，我们可以计算：

$$\begin{align} E(z_j^{(i)} \mid D,\Theta_l) & = p(\mathcal{G}_j \mid \boldsymbol{x}^{(i)},\Theta_l) \\
						& = \frac{p(\mathcal{G}_j\mid \Theta_l)p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta_l)}{p(\boldsymbol{x}^{(i)}\mid \Theta_l)} \text{（贝叶斯法则）} \\
						& = \frac{p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_j, \Theta_l)p(\mathcal{G}_j\mid \Theta_l)}{\sum_{\widetilde{j}} p(\boldsymbol{x}^{(i)}\mid \mathcal{G}_{\widetilde{j}}, \Theta_l)p(\mathcal{G}_{\widetilde{j}}\mid \Theta_l)} \\
						& =  \frac{\pi_j|\boldsymbol{S}_j|^{-\frac{1}{2}}\exp[-\frac{1}{2}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)^T\boldsymbol{S}_j^{-1}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)]}{\sum_{\widetilde{j}} \pi_{\widetilde{j}}|\boldsymbol{S}_{\widetilde{j}}|^{-\frac{1}{2}}\exp[-\frac{1}{2}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_{\widetilde{j}})^T\boldsymbol{S}_{\widetilde{j}}^{-1}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_{\widetilde{j}})]} \\
						& \equiv h_j^{(i)} 
\end{align}$$

在M步求解可得：

$$\pi_j = \frac{\sum_i h_j^{(i)}}{N}$$

$$\boldsymbol{m}_j = \frac{\sum_i h_j^{(i)}\boldsymbol{x}^{(i)}}{\sum_i h_j^{(i)}}$$

$$\boldsymbol{S}_j = \frac{\sum_i h_j^{(i)}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)^T}{\sum_i h_j^{(i)}}$$

可以看出，EM方法的执行过程如下：在E步，给定对分支的认知，我们估计这些标号；而在M步，给定E步估计的标号，我们更新我们对分支的认知。这两步与$k$-均值方法的两步相同：$$b_j^{(i)}$$的计算（E步）和$$\boldsymbol{m}_j$$的重新估计（M步）。只是在EM方法中，估计的软标号$$h_j^{(i)}$$取代了硬标号$$b_j^{(i)}$$。

EM方法一般用$k$-均值方法来进行初始化。在若干次$k$-均值迭代后，我们得到中心$$\boldsymbol{m}_j$$的估计，并且使用被每个中心涵盖的实例，我们估计$$\boldsymbol{S}_j$$并通过$$\frac{\sum_i b_j^{(i)}}{N}$$的到$$\pi_j$$。

在采用EM方法时，我们可以进一步化简假设，如可以假设所有实例共享协方差，即$$p(\boldsymbol{x}\mid \mathcal{G}_j, \Phi) \sim \mathcal{N}(\boldsymbol{m}_j, \boldsymbol{S})$$，此时可得到优化问题

$$\min_{\boldsymbol{m}_j,\boldsymbol{S}}\sum_i\sum_j h_j^{(i)}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)^T\boldsymbol{S}^{-1}(\boldsymbol{x}^{(i)}-\boldsymbol{m}_j)$$

我们可以进一步假设$$p(\boldsymbol{x}\mid \mathcal{G}_j, \Phi) \sim \mathcal{N}(\boldsymbol{m}_j, \boldsymbol{I})$$，这时优化问题化为

$$\min_{\boldsymbol{m}_j}\sum_i\sum_j h_j^{(i)}\|\boldsymbol{x}^{(i)}-\boldsymbol{m}_j\|^2$$

这是我们在$k$-均值聚类中定义的总重构误差，现在不同的是

$$h_j^{(i)} = \frac{\exp(-\frac{1}{2}\|\boldsymbol{x}^{(i)}-\boldsymbol{m}_j\|^2)}{\sum_{\widetilde{j}} \exp(-\frac{1}{2}\|\boldsymbol{x}^{(i)}-\boldsymbol{m}_{\widetilde{j}}\|^2)}$$

是0和1之间的概率，它以一定的概率将输入实例指定到所有簇中。因此，当使用$$h_j^{(i)}$$而不是$$b_j^{(i)}$$，实例对所有分支参数的更新都有贡献。这样，我们可以把$k$-均值聚类看作EM聚类的特例，假定实例输入变量是独立的、均具有单位方差，并且标号是“硬”的。






