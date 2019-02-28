---
layout: post
title: "参数估计与贝叶斯学习" 
---

# 1、参数估计

给定了概率密度族，概率密度模型就定义在若干参数（例如均值、方差），即样本的分布的有效统计量（statistics）上。因此，概率密度模型的训练过程就是参数估计（parameter estimation）的过程。

在机器学习中，密度估计可通过参数估计来直接获得$p(\boldsymbol{x})$；分类任务可通过参数估计来获得类条件（似然）密度$p(\boldsymbol{x}\mid C)$（见后面的贝叶斯分类器，它是“生成式模型”）；回归任务则一般直接估计密度$p(y\mid \boldsymbol{x})$（属于“判别式模型”）。


对于参数估计，统计学界的两个学派分别提供了不同的解决方案：频率主义学派（Frequentist）认为参数虽然未知，但却是客观存在的固定值，因此，可通过优化似然函数等准则来确定参数值；贝叶斯学派（Bayesian）则认为参数是未观察到的随机变量，其本身也可有分布，因此，可假定参数服从一个先验分布，然后基于观测到的数据来计算参数的后验分布。

频率主义学派的参数估计主要包括最大似然估计，贝叶斯学派的参数估计主要包括最大后验估计和贝叶斯估计。

## 1.2 最大似然估计

假定我们有一个独立同分布（iid）样本集$$D = \{\boldsymbol{x}^{(i)}\}_{i=1}^N$$。我们假设$\boldsymbol{x}^{(i)}$是从某个定义在参数$\theta$上的已知概率密度族$p(\boldsymbol{x}\mid\theta)$中抽取的实例：

$$\boldsymbol{x}^{(i)} \sim p(\boldsymbol{x}\mid\theta)$$

我们希望找出这样的$\theta$，使得$\boldsymbol{x}^{(i)}$尽可能像是从$p(\boldsymbol{x}\mid\theta)$抽取出来的。因为$\boldsymbol{x}^{(i)}$是相互独立的，给定参数$\theta$，样本集$D$的似然（likelihood）是所有样本似然的乘积：

$$L(D | \theta) \equiv p(D | \theta) = \prod_{i=1}^Np(\boldsymbol{x}^{(i)} | \theta)$$

在最大似然估计（maximum likelihood estimation）中，我们感兴趣的是找到这样的$\theta$，使得$$L(D \mid \theta)$$最大。

通常情况下，我们可以最大化该似然的对数。通过log运算可以把乘积转换为求和，并不改变它取最大值时的解，并且当概率密度函数包含指数项时可以进一步简化计算量。对数似然（log likelihood）定义为：

$$\mathcal{L}(\theta|D) \equiv \log L(D| \theta) = \sum_{i=1}^N\log p(\boldsymbol{x}^{(i)} | \theta)$$

此时参数$\theta$的最大似然估计为：

$$\theta_{ML} = arg \max_{\theta} \mathcal{L}(\theta|D)$$

### 1.2.1 高斯分布示例

设$X$是均值为$\mu$、方差为$\sigma^2$的高斯分布，记作$\mathcal{N}(\mu, \sigma^2)$，它的密度函数为：

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2}), \ -\infty < x < \infty$$

给定样本集$$D = \{x^{(i)}\}_{i=1}^N$$，其中$$x^{(i)} \sim \mathcal{N}(\mu, \sigma^2)$$，高斯样本的对数似然为：

$$\mathcal{L}(\mu,\sigma | D) = -\frac{N}{2}\log(2\pi) - N\log \sigma - \frac{\sum_i(x^{(i)}-\mu)^2}{2\sigma^2}$$

通过对该对数似然函数求偏导并置零，可得均值和方差的最大似然估计分别为：

$$m = \frac{\sum_ix^{(i)}}{N}$$

$$e^2 = \frac{\sum_i(x^{(i)}-m)^2}{N}$$

### 1.2.2 估计性能的评价

令$D$ 是取自参数$\theta$指定分布的样本集，并令$d=d(D)$是$\theta$的一个估计。为了评估该估计的性能，我们可以度量它与$\theta$有多大的不同，即$(d(D)-\theta)^2$。但是它是一个随机变量（依赖于样本集），我们需要对它在可能的$D$上取均值，进而考虑$r(d,\theta)$，它是估计$d$的均方误差（mean square error），定义为：

$$r(d,\theta) = \mathbb{E}[(d(D)-\theta)^2]$$

均方误差可以进一步写为：

$$\begin{align}
r(d,\theta) & = E[(d-\theta)^2] \\
	& = E[(d-E[d]+E[d]-\theta)^2] \\
	& = E[(d-E[d])^2 + (E[d]-\theta)^2 + 2(E[d]-\theta)(d-E[d])] \\
	& = E[(d-E[d])^2] + (E[d]-\theta)^2 \\
\end{align}$$

在上式中，第一项是方差（variance），度量在平均情况下估计$d$在期望周围的分散程度（从一个数据集到另一个数据集）；第二项是偏倚（bias）的平方，度量估计期望值偏离正确值$\theta$的程度，具体定义为：

$$b_{\theta}(d) = E[d(D)] - \theta$$

于是，我们把估计的均方误差写成方差和偏倚的平方之和：

$$r(d,\theta) = Var(d) + (b_{\theta}(d))^2$$

特别地，如果$b_{\theta}(d) = 0$，则称$d$是$\theta$的无偏估计（unbiased estimator）；如果当$N\rightarrow \infty$时，$b_{\theta}(d) \rightarrow 0$，则称$d$是$\theta$的渐近无偏估计（asymptotically unbiased estimator）；如果当$N\rightarrow \infty$时，$b_{\theta}(d) \rightarrow 0$ 且 $Var(d) \rightarrow 0$，则称$d$是$\theta$的一致估计（consistent estimator）。

例如，如果$x^{(i)}$是从均值$\mu$的密度抽取出来的，则样本平均值$m$是均值$\mu$的一个无偏估计，因为：

$$E(m) = E[\frac{\sum_i x^{(i)}}{N}] = \frac{1}{N}\sum_i E[x^{(i)}] = \frac{N\mu}{N} = \mu$$

这就意味着虽然在一个特定样本上，$m$可能与$\mu$不同，但是如果我们取许多这样的样本集$D_i$，并且估计许多$m_i = m(D_i)$，随着样本集的增加，它们的平均值将逼近$\mu$。

同时，$m$也是$\mu$的一个一致估计，即当$N\rightarrow \infty$时，$Var(m) \rightarrow 0$：

$$Var(m) = Var(\frac{\sum_i x^{(i)}}{N}) = \frac{1}{N^2}\sum_i Var(x^{(i)}) = \frac{N\sigma^2}{N^2} = \frac{\sigma^2}{N}$$

随着样本集中的样本点数$N$增大，$m$对$\mu$的偏离将变小。

接下来我们检查$\sigma^2$的最大似然估计$e^2$：

$$e^2 = \frac{\sum_i(x^{(i)}-m)^2}{N} = \frac{\sum_i(x^{(i)})^2-Nm^2}{N}$$

$$E[e^2] = \frac{\sum_i E[(x^{(i)})^2]-N\cdot E[m^2]}{N}$$

给定$Var[X] = E[X^2] - E[X]^2$，可以得到$E[X^2] = Var[X] + E[X]^2$，于是有：

$$E[(x^{(i)})^2] = \sigma^2 + \mu^2$$

$$E[m^2] = \frac{\sigma^2}{N} + \mu^2$$

进而可得到：

$$E[e^2] = \frac{N(\sigma^2 + \mu^2) - N(\frac{\sigma^2}{N}+\mu^2)}{N} = (\frac{N-1}{N})\sigma^2 \neq \sigma^2$$

上式说明$e^2$是$\sigma^2$的有偏估计，而$$(\frac{N}{N-1})e^2$$是一个无偏估计。然而，当$N$很大时，二者差别可以忽略。这是一个渐近无偏估计的例子，它的偏倚随着$N$趋向无穷而趋向于0。

## 1.3 最大后验估计与贝叶斯估计

有时，在看到样本集之前，我们（或应用领域专家）可能会有一些关于参数$\theta$可能取值的先验（prior）信息。这些信息是非常有用的，应当利用起来，**尤其是当样本数较小时**。

这些先验信息不会告诉我们参数的确切值（否则我们就不需要样本集了），并且我们通过把$\theta$看作是一个随机变量并为它定义先验密度$p(\theta)$来对它们的不确定性进行建模。

先验密度（prior density）$p(\theta)$告诉我们在看到样本集之前$\theta$的可能取值。我们把它和样本数据告诉我们的（即似然密度$$p(D\mid \theta)$$）信息结合起来，利用贝叶斯规则，得到$\theta$的后验密度（posterior density），它告诉我们看到样本集之后$\theta$的可能取值：

$$p(\theta| D) = \frac{p(D|\theta)p(\theta)}{p(D)} = \frac{p(D |\theta)p(\theta)}{\int p(D|\theta')p(\theta')d\theta'}$$

为了明确表示样本集$D$中有$n$个样本，可以将其标记为$$D_n$$，易得似然概率：

$$p(D_n | \theta) = \prod_{i=1}^n p(\boldsymbol{x}^{(i)} | \theta)$$

因此：

$$p(D_n | \theta) = p(\boldsymbol{x}^{(n)} | \theta)p(D_{n-1} | \theta)$$

进而容易得到：

$$p(\theta | D_n) = \frac{p(\boldsymbol{x}^{(n)} | \theta)p(D_{n-1} | \theta)p(\theta)}{\int p(\boldsymbol{x}^{(n)} | \theta')p(D_{n-1} | \theta')p(\theta')d\theta'} = \frac{p(\boldsymbol{x}^{(n)} | \theta)p(\theta |D_{n-1})}{\int p(\boldsymbol{x}^{(n)} | \theta')p(\theta' | D_{n-1} )d\theta'}$$

当没有观测样本时，定义$$p(\theta \mid D_0) = p(\theta)$$，为参数$$\theta$$的先验估计。然后让样本集合依次进入上述公式，就可以得到一系列的概率密度函数：$$p(\theta \mid D_0)$$、$$p(\theta \mid D_1)$$、$$p(\theta \mid D_2)$$、……、$$p(\theta \mid D_n)$$，这一过程称为**参数估计贝叶斯递归法**。这是一个在线学习过程，它和随机梯度下降法有很多相似之处。


最大后验估计（maximum a posteriori, MAP）是将后验概率取最大值时的$\theta$（称为后验概率分布下的众数）作为估计：

$$\theta_{MAP} = arg \max_{\theta} p(\theta | D)$$

如果我们没有更重要的理由偏爱$\theta$的某些值，则先验概率是扁平的，后验将与似然$p(D\mid\theta)$有同样的形式，因此MAP估计将等价于最大似然估计。

另外一种方法是贝叶斯估计（Bayes' estimator），它被定义为后验概率的期望值：

$$\theta_{Bayes} = E[\theta | D] = \int \theta p(\theta | D)d\theta$$

如果后验概率$ p(\theta \mid D)$满足高斯分布，则期望值就是众数，则$\theta_{Bayes} = \theta_{MAP}$。

举个例子，我们假设$$x^{(i)}\sim \mathcal{N}(\theta,\sigma_0^2)$$并且先验$\theta\sim \mathcal{N}(\mu,\sigma^2)$，其中$\mu$、$\sigma$和$\sigma_0^2$已知，可得：

$$p(D|\theta) = \frac{1}{(2\pi)^{N/2}\sigma_0^N}\exp(-\frac{\sum_i(x^{(i)}-\theta)^2}{2\sigma_0^2})$$

$$p(\theta) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\theta-\mu)^2}{2\sigma^2})$$

可以证明$p(\theta\mid D)$是正态的，满足：

$$E[\theta | D] = \frac{N/\sigma_0^2}{N/\sigma_0^2 + 1/\sigma^2}m + \frac{1/\sigma^2}{N/\sigma_0^2 + 1/\sigma^2}\mu$$

因此，贝叶斯估计是先验均值$\mu$和样本均值$m$的加权平均值，权重与它们的方差成反比。当$\sigma^2$较小时，即当我们关于$\sigma$的取值具有较少的先验不确定性时，或者当$N$较小时，我们的先验猜测$\mu$具有较好效果；利用样本提供的更多信息，随着样本规模$N$的增加，贝叶斯估计逼近样本的平均值。

# 2、贝叶斯密度估计

在使用贝叶斯方法获得$\theta$的后验密度$p(\theta\mid D)$后，为了估计$\boldsymbol{x}$上的密度，我们可以进一步获得：

$$\begin{align}
p(\boldsymbol{x}| D) &= \int p(\boldsymbol{x},\theta\mid D)d\theta \\
		& = \int p(\boldsymbol{x}\mid\theta, D)p(\theta\mid D)d\theta \\
		& = \int p(\boldsymbol{x}\mid\theta)p(\theta\mid D)d\theta
\end{align}$$

上式中$$p(\boldsymbol{x}\mid\theta, D) = p(\boldsymbol{x}\mid\theta)$$满足是因为只要我们知道了有效统计量$\theta$，我们就知道了关于分布的一切。

这称为贝叶斯密度估计，实际上我们是在使用所有$\theta$值的预测上取平均，用它们的概率加权。


# 3、参数估计角度的线性回归

线性回归模型是输入变量$\boldsymbol{x}$的基函数的线性组合，在数学上其形式如下：

$$\widehat{y}(\boldsymbol{x},\boldsymbol{w}) = w_0 + \sum_{j=1}^M w_j\phi_j(\boldsymbol{x})$$

这里$$\phi_j(\boldsymbol{x})$$就是前面提到的基函数，总共的基函数数目为$M$，如果定义$$\phi_0(\boldsymbol{x})=1$$的话，那么上面式子可以简单表示为：

$$\widehat{y}(\boldsymbol{x},\boldsymbol{w}) = \boldsymbol{w}^T\boldsymbol{\phi}(\boldsymbol{x})$$

其中，$$\boldsymbol{w} = (w_0,w_1,w_2,...,w_M)^T$$，$$\boldsymbol{\phi} = (\phi_0,\phi_1,\phi_2,...,\phi_M)^T$$。


## 3.1 最大似然线性回归

我们定义$$p(y\mid \boldsymbol{x},\boldsymbol{w}) \sim \mathcal{N}(\widehat{y}(\boldsymbol{x},\boldsymbol{w}),\sigma^2)$$，并假设样本是独立同分布的，可得条件对数似然如下：

$$\sum_{i=1}^N\log p(y^{(i)} | \boldsymbol{x}^{(i)}, \boldsymbol{w}) = -\frac{N}{2}\log(2\pi) - N\log \sigma - \frac{\sum_{i=1}^N\|y^{(i)}-\widehat{y}^{(i)}\|^2}{2\sigma^2}$$

对比均方误差：

$$MSE = \frac{1}{N}\sum_{i=1}^N\|y^{(i)}-\widehat{y}^{(i)}\|^2$$

可以看出，最大化关于$\boldsymbol{w}$的对数似然和最小化均方误差会得到相同的参数估计$\boldsymbol{w}$，因此对数似然和最小二乘法是等价的。


## 3.2 贝叶斯线性回归

我们假设给定了一组$N$个训练样本$$(\boldsymbol{X},\boldsymbol{y})$$，利用贝叶斯定理，可得模型参数的后验分布：

$$p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) \propto p(\boldsymbol{y}\mid \boldsymbol{X},\boldsymbol{w})p(\boldsymbol{w})$$

可以定义$$p(\boldsymbol{y}\mid \boldsymbol{X},\boldsymbol{w}) \sim \mathcal{N}(\widehat{\boldsymbol{y}}(\boldsymbol{X},\boldsymbol{w}),\frac{1}{\beta}\boldsymbol{I})$$、$$p(\boldsymbol{w}) \sim \mathcal{N}(\boldsymbol{0},\frac{1}{\alpha}\boldsymbol{I})$$。其中$\beta^{-1}$和$\alpha^{-1}$分别对应样本集和权重的高斯分布的方差。

那么，线性模型的对数后验概率函数可表示为：

$$\log p(\boldsymbol{w} | \boldsymbol{X},\boldsymbol{y}) = -\frac{\beta}{2}\sum_{i=1}^N\|y^{(i)}-\widehat{y}^{(i)}\|^2 - \frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w} + const $$

因此，最大化后验概率和带权重衰减项$$\frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w}$$的最小二乘回归是等价的。

# 4、贝叶斯分类器

## 4.1 类后验概率的计算

考虑分类任务中的“生成式模型”，学习联合分布$p(\boldsymbol{x},C)$是极其困难的（基于有限训练集直接估计联合概率，在数据上会遭遇样本稀疏问题，在计算上会遭遇组合爆炸问题，属性数越多，问题越严重。），我们可以进一步基于贝叶斯定理将$p(\boldsymbol{x},C)$写成：

$$p(\boldsymbol{x},C) = p(C)p(\boldsymbol{x}|C)$$

于是可得

$$p(C | \boldsymbol{x}) = \frac{p(C)p(\boldsymbol{x}|C)}{\sum_{C'}p(C')p(\boldsymbol{x}|C')}$$


因此，学习联合分布$p(\boldsymbol{x},C)$可以转化为学习以下概率分布：

1）类先验概率分布：$p(C)$；

2）类条件（似然）概率分布：$p(\boldsymbol{x}\mid C)$。

可以看出类条件（似然）概率$p(\boldsymbol{x}\mid C)$是所有特征属性上的联合概率，仍然难以从有限的训练样本直接估计而得到。

以朴素贝叶斯分类器（naive Bayes classifier）为例，其采用了“属性条件独立性假设”（attribute conditional independence assumption）：在分类确定的条件下，假设所有特征属性相互独立。于是可得：

$$\begin{align} p(\boldsymbol{x}\mid C) = \prod_{i=1}^d p(x_i \mid C)
\end{align}$$ 

其中$d$为特征属性的数目，$$x_i$$为$\boldsymbol{x}$在第$i$个属性上的取值。

令$$D_C$$表示训练集$D$中第$C$类样本组成的集合，若有充足的独立同分布样本，则可容易地估计出类先验概率：

$$p(C) = \frac{|D_C|}{|D|}$$

对离散属性而言，令$$D_{C,x_i}$$表示$$D_C$$中在第$i$个属性上取值为$$x_i$$的样本组成的集合，则类条件（似然）概率可估计为：

$$p(x_i |C) = \frac{|D_{C,x_i}|}{|D_C|}$$

对连续属性而言，可考虑概率密度函数。假定通过参数估计得到$$p(x_i\mid C) \sim \mathcal{N}(\mu_{C,i},\sigma_{C,i}^2)$$，则有：

$$p(x_i | C) = \frac{1}{\sqrt{2\pi}\sigma_{C,i}}\exp(-\frac{(x_i-\mu_{C,i})^2}{2\sigma_{C,i}^2})$$


## 4.2 贝叶斯最优分类器

假设一共有$K$种可能的类别标记，即$\mathcal{Y} = \\{C_1,C_2,...,C_K\\}$，$\lambda_{ij}$是将一个真实标记为$$C_j$$的样本误分类为$$C_i$$所产生的损失。则在获得类后验概率$p(C\mid\boldsymbol{x})$后，可得到将样本$\boldsymbol{x}$分类为$$C_i$$所产生的期望损失（expected loss），又称“条件风险”（conditional risk）：

$$R(C_i | \boldsymbol{x}) = \sum_{j=1}^K\lambda_{ij}p(C_j | \boldsymbol{x})$$

我们的任务是寻找一个分类器$h: \mathcal{X} \rightarrow \mathcal{Y}$以最小化总体风险：

$$R(h) = E_{\boldsymbol{x}}[R(h(\boldsymbol{x})|\boldsymbol{x})]$$

显然，对每个样本$\boldsymbol{x}$，若$h$都能最小化条件风险$$R(h(\boldsymbol{x})\mid\boldsymbol{x})$$，则总体风险也将被最小化。因此，有：

$$h^*(\boldsymbol{x}) = arg\min_{C\in\mathcal{Y}}R(C | \boldsymbol{x})$$

此时，$$h^*$$称为贝叶斯最优分类器（Bayes optimal classifier），与之对应的总体风险$$R(h^*)$$称为贝叶斯风险（Bayes risk）。

如果我们的目标是最小化分类错误率，则误判损失$$\lambda_{ij}$$可写为0/1损失函数的形式：

$$\lambda_{ij} = \begin{cases} 0 & \text{if } i=j \\ 1 & \text{otherwise} \end{cases}$$

此时条件风险为：

$$R(C| \boldsymbol{x}) = 1- p(C|\boldsymbol{x})$$

于是，最小化分类错误率的贝叶斯最优分类器为：

$$h^*(\boldsymbol{x}) = arg\max_{C\in\mathcal{Y}}p(C | \boldsymbol{x})$$

即对每个样本$\boldsymbol{x}$，选择能使类后验概率$$p(C\mid\boldsymbol{x})$$最大的分类标记。




