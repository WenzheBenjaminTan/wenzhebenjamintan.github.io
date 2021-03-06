---
layout: post
title: "机器学习及其实验设计" 
---

# 1、无监督学习与监督学习

无监督学习算法用于学习含有很多特征（feature）的数据集（dataset），然后得到这个数据集上有用的结构性质。在机器学习中，我们通常要学习用于生成数据集的整个概率分布，显示地，比如密度估计（density estimation），或是隐式地，比如生成（generation，产生一些和训练数据相似的新样本）或去噪（denoising，根据损坏的样本预测原本的样本）。还有一些其他类型的无监督学习任务，例如降维（将样本的维度进行缩减）和聚类（将数据集分成相似样本的集合）。

监督学习算法也用于学习含有很多特征的数据集，不过数据集中的样本都有一个标签（label）或目标（target）。

大致来说，无监督学习涉及观察随机向量$\boldsymbol{x}$的若干样本，试图显式或隐式地学习出概率分布$p(\boldsymbol{x})$，或是跟该分布相关的一些性质；而监督学习则是观察随机向量$\boldsymbol{x}$及其目标值或向量$y$，然后从$\boldsymbol{x}$预测$y$，通常是估计 $$p(y \mid \boldsymbol{x})$$。

无监督学习和监督学习并不是严格定义的术语，它们之间的界限通常是模糊的。很多机器学习技术可以同时用于这两个任务。

例如，概率的链式法则表明对于随机变量$$\boldsymbol{x} \in \mathbb{R}^d$$，联合分布可以分解为：

$$p(\boldsymbol{x}) = \prod_{i=1}^d p(x_i | x_1,x_2,...,x_{i-1}) $$

该分解意味着我们可以将其拆分为$d$个监督学习问题，来解决表面上的无监督学习$p(\boldsymbol{x})$。

同样地，我们求解监督学习问题的$$p(y\mid \boldsymbol{x})$$时，也可以使用传统的无监督学习方法来学习联合分布$p(\boldsymbol{x},y)$，然后推断：

$$p(y | \boldsymbol{x}) = \frac{p(\boldsymbol{x},y)}{\sum_{\widehat{y}}p(\boldsymbol{x},\widehat{y})}$$

对于监督学习，当给定有限的训练样本集，如是通过直接建模$$p(y\mid \boldsymbol{x})$$来预测$y$，这样得到的是“判别式模型”（discriminative model）；当是先对联合分布$p(\boldsymbol{x},y)$建模，然后再由此获得$$p(y\mid \boldsymbol{x})$$时，这样得到的是“生成式模型”（generative models）。

尽管无监督学习和监督学习并非完全不相关，但它们确实有利于粗略区分我们研究机器学习时遇到的问题。传统上，人们将回归、分类或结构化输出问题称为监督学习，将支持其他任务的密度估计相关问题称为无监督学习。

学习范式的其他变种也是可能的。例如半监督学习中，一些样本有监督目标，但其他样本没有。

大部分机器学习算法都简单地被训练于一个固定的数据集上。数据集可以用很多不同方式来表示，但是在所有情况下，数据集都是样本的集合，而样本是特征的集合。有些机器学习算法并不是用于一个固定的数据集上。例如，强化学习算法会和环境交互，所以学习系统和它的训练过程会有反馈回路。


# 2、机器学习实验设计

## 2.1 符号定义

$a$ : 标量

$\boldsymbol{a}$ : 向量

$\boldsymbol{A}$ : 矩阵

$\mathcal{X}$ : 输入空间

$\mathcal{Y}$ : 输出空间

$D$ : 可观测数据集

$Z$ : 隐藏的数据集

$\boldsymbol{x}^{(i)}$ : 数据集的第$i$个样本输入，共有$N$个样本

$$x^{(i)}_j$$ : 样本输入$\boldsymbol{x}^{(i)}$的第$j$个分量，共有$d$个分量

$\boldsymbol{X}$ : 大小为 $N \times d$ 的输入矩阵（或称设计矩阵），其中行$\boldsymbol{X}_{i,:}$为样本输入$\boldsymbol{x}^{(i)}$

$y^{(i)}$ : 监督学习中与$\boldsymbol{x}^{(i)}$关联的标签输出

$$C_i$$ : 分类问题中的第$i$个类，共有$K$个类，此时$$\mathcal{Y} = \{C_1,C_2,...,C_K\}$$

$\boldsymbol{a}^{[l]}$：人工神经网络中，第$l$层的激活值向量

$\boldsymbol{w}^{< i >}$ : 第$i$轮迭代时参数$\boldsymbol{w}$的值

$\mathcal{H}$ : 假设空间

$h$ : 假设，学得的模型

$\boldsymbol{a} \bullet \boldsymbol{a}'$ : 向量$\boldsymbol{a}$ 与$\boldsymbol{a}'$ 的内积

$\boldsymbol{a} \times \boldsymbol{a}'$ : 向量$\boldsymbol{a}$ 与$\boldsymbol{a}'$ 的外积

$\boldsymbol{a} \odot \boldsymbol{a}'$ : 向量$\boldsymbol{a}$ 与$\boldsymbol{a}'$ 的点积 

$1(\cdot)$ : 指示函数，在$\cdot$为真和假时分别取值为1，0

$sign(\cdot)$ : 符号函数，在$\cdot$ <0，=0，>0 时分别取值为-1，0，1



## 2.2 训练和测试

在监督学习中，一般用 $$(\boldsymbol{x}^{(i)}, y^{(i)})$$ 表示数据集（data set） $D$中的第 $i$ 个样本（sample），其中$$\boldsymbol{x}^{(i)}\in \mathcal{X}$$、$$y^{(i)}\in \mathcal{Y}$$。在无监督学习中，则$$D = \{\boldsymbol{x}^{(i)}\}_{i=1}^N$$。

从数据中学得模型的过程称为“学习”（learning）或“训练”（training），这个过程中通过执行某个学习算法（learning algorithm）来完成。训练过程中使用的数据称为“训练数据”（training data），其中的每个样本称为“训练样本”（training sample），训练样本组成的集合称为“训练集”（training set）。学得模型对应了关于数据的某种潜在的规律，因此亦称“假设”（hypothesis）；这种潜在规律自身，则称为“真相”（ground-truth），学习过程就是为了找出或逼近真相。有时候也将学得模型称为“学习器”（learner），可看作学习算法在给定训练数据和假设空间后的实例化。

机器学习的目标是使学得模型能很好地适用于“新样本”，而不是仅仅在训练样本上工作得很好。学得模型适用于新样本的能力，称为“泛化”（generalization）能力。具有强泛化能力的模型能很好地适用于整个样本空间。于是，尽管训练集通常只是样本空间的一个很小的采样，我们仍希望它能很好地反映出样本空间的特性，否则就很难期望在训练集上学得的模型能在整个样本空间上都工作得很好。通常假设样本空间中全体样本服从一个未知“分布”（distribution）$p$，我们获得的每个样本都是独立地从这个分布上采样获得的，即“独立同分布”。一般而言，训练样本越多，我们得到的关于$p$的信息越多，这样就越有可能通过学习获得具有强泛化能力的模型。

在学得模型后，使其进行预测的过程称为“测试”（testing），被预测的样本称为“测试样本”（test sample），测试样本组成的集合称为“测试集”（test set）。在测试集上可以估计学得模型的泛化能力，得到其期望误差。测试集应当是之前从未使用过的，且应该足够大以使误差估计有意义。

## 2.3 验证和交叉验证

大多数学习算法都有超参数（hyperparameter）（如批次大小、训练迭代的总次数、学习率、正则化参数、神经网络的层数、决策树的数量或树的深度、聚类的簇数等），可以用于控制算法行为或限制假设空间。超参数的值不是通过学习算法本身学习出来的。为了优化超参数，我们需要一个训练过程中观测不到的“验证集”（validation set）。

前面我们讨论过和训练数据相同分布的样本组成的测试集，它可以用来估计学习过程完成之后的学习器的泛化误差。但是，测试样本不能以任何形式参与到模型的确定过程中，包括设定超参数。因此，测试集中的样本不能用于验证集，验证集只能从训练数据中构建。

此外，对于同一个算法，一轮验证是不够的，因为训练集和验证集都可能较小并且可能包含异常实例，如噪声或离群点，可能会误导我们。为了平均这种随机性，我们可以使用一定数目的训练集和验证集对来对同一个学习算法进行多轮验证，基于验证误差的分布来评估学习算法的性能，从而优化超参数。

为了能够重复验证，我们的第一需求是（在留下一部分作为测试集后，如不需要测试则无需保留）从数据集$D$中获得一定数目的训练集和验证集对。如果数据集$D$足够大，我们可以随机地将其分为$K$个部分（$K$通常为10或30），然后将每一部分随机地分为两半，一半用于训练，另一半用于验证。验证误差可以估计为$K$次计算后的平均验证误差。然而，数据集很少有如此之大以允许我们这么做。因此，我们应该在小数据集上尽力而为，其方法是以不同划分来重复使用相同数据，这称为“交叉验证”（cross-validation）。交叉验证的潜在问题是验证误差是相互依赖的，因为这些不同划分共享了数据。

最常见的是$K$-折交叉验证方法，将数据集分成$K$个不重合的子集。在第$i$次验证时，数据的第$i$个子集用于验证集，其他的数据用于训练集。需要特别说明的是，随着计算成本的降低，多次运行$K$-折交叉验证已经成为可能（例如，$10\times 10$折），并且在验证误差的平均值上取平均，以便得到更可靠的误差估计。

在优化完学习算法的超参数后，可以将所有训练集和验证集合并为最终的训练集用来训练最终的模型，然后通过测试集来评估最终的泛化误差。需要注意的是，我们经常是把学得模型在实际使用中遇到的数据集合称为测试集，因此，测试过程也是模型的实际使用过程。


## 2.4 假设空间

我们希望学得模型是在测试集上泛化能力表现很好的学习器。为了达到这个目的，应该从训练样本中尽可能学出适用于所有潜在样本的“普遍规律”，这样才能在遇到新样本时做出正确的判别。然而，当学习器把训练样本学得“太好”了的时候，很可能已经把训练样本自身的一些特点当作了潜在样本都会具有的一般性质，这样就会导致泛化性能下降。机器学习主要面临两大挑战：欠拟合（underfitting）和过拟合（overfitting）。欠拟合是指模型不能在训练集上获得足够低的误差，而过拟合是指训练误差和测试误差之间的差距太大。

通过调整模型的容量（capacity），我们可以控制模型是否偏向于过拟合或者欠拟合。通俗来讲，模型的容量是指其拟合各种函数的能力。容量低的模型可能很难拟合训练集，而容量高的模型可能会过拟合，因为记住了不适用于测试集的训练集性质。

一种控制学习算法容量的方法是选择“假设空间”（hypothesis space），即学习算法可以选择为解决方案的函数的集合。例如，线性回归算法将关于其输入的所有线性函数作为假设空间。而广义线性回归的假设空间还包括非线性函数，而非仅有线性函数，这样做就相当于增加了学得模型的容量。

## 2.5 归纳偏好

通过学习得到的模型对应了假设空间中的一个假设。可能会有多个假设的训练误差都是一样的，但与它们对应的模型在面临新样本的时候，却会产生不同的输出，这时应该采用哪一个假设呢？这时就需要依靠“归纳偏好”（inductive bias）来发挥作用了。具体来说，归纳偏好是指机器学习算法在学习过程中对某种假设类型的偏好。

“奥卡姆剃刀”（Occam's razor）原则是一种常用的归纳偏好原则，即“若有多个假设与观察一致，则选择最简单的那个”。

事实上，归纳偏好对应了学习算法本身所做出的关于“什么样的模型更好”的假设。在具体的现实问题中，这个假设是否成立，即算法的归纳偏好是否与问题本身匹配，大多数时候直接决定了算法能否取得好的性能。

## 2.6 “没有免费午餐”定理

假设样本空间$\mathcal{X}$和假设空间$\mathcal{H}$都是离散的。令$$p(h\mid D,\mathcal{L}_a)$$代表算法$$\mathcal{L}_a$$基于训练集$D$产生假设$h$的概率，再令$f$代表我们希望学习的真相。$$\mathcal{L}_a$$的“训练集外误差”，即$$\mathcal{L}_a$$在训练集之外的所有样本上的误差为：

$$E_{ote}(\mathcal{L}_a \mid D,f) = \sum_{h} \sum_{\boldsymbol{x} \in \mathcal{X} - D} p(\boldsymbol{x})1(h(\boldsymbol{x}) \neq f(\boldsymbol{x}))p(h\mid D,\mathcal{L}_a)$$

考虑二分类问题，$f$可以是任何$$\mathcal{X}\rightarrow \{0,1\}$$的函数，函数空间大小为$2^{\mid\mathcal{X}\mid}$。对所有可能的$f$按均匀分布对训练集外误差求和，可得：

$$\begin{align}\sum_f E_{ote}(\mathcal{L}_a \mid D,f) & = \sum_f \sum_{h} \sum_{\boldsymbol{x} \in \mathcal{X} - D} p(\boldsymbol{x})1(h(\boldsymbol{x}) \neq f(\boldsymbol{x}))p(h\mid D,\mathcal{L}_a) \\
	& = \sum_{\boldsymbol{x} \in \mathcal{X} - D}  p(\boldsymbol{x}) \sum_{h} p(h\mid D,\mathcal{L}_a) \sum_f 1(h(\boldsymbol{x}) \neq f(\boldsymbol{x})) \\
	& = 2^{\mid\mathcal{X}\mid - 1} \sum_{\boldsymbol{x} \in \mathcal{X} - D}  p(\boldsymbol{x}) \\
\end{align}$$

由此可看出，总误差竟然与学习算法无关，对任意两个学习算法$$\mathcal{L}_a$$和$$\mathcal{L}_b$$，我们都有

$$ \sum_f E_{ote}(\mathcal{L}_a \mid D,f) = \sum_f E_{ote}(\mathcal{L}_b \mid D,f) $$

也就是说，无论学习算法$$\mathcal{L}_a$$多聪明，学习算法$$\mathcal{L}_b$$多笨拙，对于“真相”出现的机会相同情况下，它们的期望性能都是相等的。这就是“没有免费午餐”（no free lunch）定理。

需要注意的是，NFL定理有一个重要前提：所有“真相”出现的机会相同，比如上面的例子中我们假设了$f$的均匀分布。而实际情形往往不是如此的。比如“奥卡姆剃刀”原则实际上是暗示简单模型代表的“真相”出现的概率会更大一些。

所以，NFL定理最重要的寓意，是让我们清楚地认识到，脱离具体的问题（“真相”出现概率的偏好），空泛地谈论“什么学习算法更好”毫无意义，因为若考虑了所有潜在的问题（所有“真相”出现的机会相同），则所有学习算法都一样好。要谈论算法的相对优劣，必须要针对具体的学习问题（对“真相”的概率分布进行假设）。在某些问题上表现好的学习算法，在另一些问题上却可能不尽如人意，学习算法自身的归纳偏好与问题是否匹配，往往会起到决定性的作用。



