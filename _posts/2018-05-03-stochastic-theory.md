---
layout: post
title: "随机理论基础" 
---
# 1、概率论

## 1.1 基本概念

概率论中一个基本概念是随机试验。一个过程，若它的结果预先无法确定，则称之为随机试验，简称为**试验**（experiment）。所有试验可能结果组成的集合，称为**样本空间**（sample space），记作$\Omega$。$\Omega$中的元素称为**样本点**（sample point），用$\omega$表示。由$\Omega$的若干个样本点构成的集合，即$\Omega$的子集（可以是$\Omega$本身），称作**事件**（event），常用大写字母（如$A$、$B$、$C$等）表示。由$\Omega$中的若干子集构成的集合称为集类，常用花写字母（如$\mathcal{A}$、$\mathcal{B}$、$\mathcal{F}$等）表示。

由于并不是所有的$\Omega$的子集都能方便的定义概率。一般我们只限制在满足一定条件的集类上研究概率性质，为此我们引入$\beta$代数概念:

**定义：** 设$\mathcal{F}$为由$\Omega$的某些子集构成的非空集类，若满足：

1）$\Omega \in \mathcal{F}$；

2) 若$A \in \mathcal{F}$，则$A^C \in \mathcal{F}$；

3）若$A_n \in \mathcal{F} (n \in \mathbb{N})$，则$\bigcup_{n=1}^\infty A_n \in \mathcal{F}$。

则称$\mathcal{F}$为**$\beta$代数**，称$(\Omega,\mathcal{F})$为**可测空间**。

**定义：** 设$(\Omega,\mathcal{F})$为可测空间，$P$是定义在$\mathcal{F}$上的集函数，若满足：

1）（非负性）$P(A) \geq 0, \forall A \in \mathcal{F}$；

2）（规一性）$P(\Omega) = 1$；

3）（可数可加性）若$A_n \in \mathcal{F} (n \in \mathbb{N})$，且$A_i \cap A_j = \varnothing, \forall i \neq j$，有

$$P(\bigcup_{n=1}^\infty A_n) = \sum_{n=1}^\infty P(A_n)$$

则称$P$为可测空间$(\Omega,\mathcal{F})$上的一个概率函数，简称**概率**（probability），称$(\Omega,\mathcal{F},P)$为**概率空间**（probability space），称$\mathcal{F}$为**事件域**，若事件$A \in \mathcal{F}$，称$P(A)$为事件$A$的概率。

**定义：** 如果事件$\\{A_1, A_2, \cdots\\}$ 两两不交，并且$\bigcup_{n=1}^\infty A_n = \Omega$，则称集合$\\{A_1, A_2, \cdots\\}$为$\Omega$的一个**划分**（partition）。

## 1.2 条件概率

**定义：** 设$A$、$B$为$\Omega$中的事件，且$P(B) > 0$，则在事件$B$发生的条件下事件$A$发生的条件概率（conditional probability of $A$ and $B$）记作$P(A\|B)$，表示为

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

注意，计算条件概率时，样本空间由初始的$\Omega$变成了$B$，所有后续事件发生的概率也都根据它们和$B$之间的关系有所调整。

**定义：** 称事件$A$、$B$是独立的，如果

$$P(A \cap B) = P(A)P(B)$$

**Bayes定理：** 设$\\{A_1, A_2, \cdots\\}$为样本空间的一个划分，$B$为任意事件，则对$i=1,2,\cdots$，有

$$P(A_i | B) = \frac{P(A_i)P(B | A_i)}{\sum_{j=1}^{\infty}P(B | A_j)P(A_j)}$$

上式又叫贝叶斯公式，其中$P(A_j)$表示可能导致结果$B$的原因$A_j$的**先验概率**（prior probability），$P(B \| A_j)$表示由原因$A_j$导致结果$B$发生的**似然概率**（likelihood probability），$$\sum_{j=1}^{\infty}P(B \mid A_j)P(A_j)=P(B)$$是观测到结果$B$发生的**边缘概率**（marginal probability），又称**证据**（evidence），而$P(A_i \| B)$表示结果$B$发生后可能是原因$A_j$导致的**后验概率**（posterior probability）。


# 2、随机变量

**定义：** 设$(\Omega,\mathcal{F},P)$为一个概率空间，$X(\omega)$是定义在样本空间$\Omega$上的单值实函数，如果对$\forall a \in \mathbb{R}$，有$\\{\omega \| X(\omega) \leq a\\} \in \mathcal{F}$，则称$X(\omega)$为概率空间$(\Omega,\mathcal{F},P)$上的**随机变量**（random variable）。

## 2.1 分布函数

**定义：** 设$X$为概率空间$(\Omega,\mathcal{F},P)$上的随机变量，其累积分布函数（cumulative distribution function，简记为cdf，简称**分布函数**）记作$F_X(x)$，表示为

$$F_X(x) = P(X \leq x), \forall x \in \mathbb{R}$$

**定理：** 函数$F(x)$是一个累积分布函数，当且仅当它同时满足下列三个条件：

1）$\lim_{x\rightarrow -\infty}F(x)=0$且$\lim_{x\rightarrow +\infty}F(x)=1$；

2）$F(x)$是$x$的单调非减函数；

3）$F(x)$右连续，即，对任意$x_0$，有$\lim_{x\rightarrow x_0^+}F(x)=F(x_0)$。

**定义：** 设$X$为一随机变量，如果$F_X(x)$是$x$的连续函数，则称$X$是**连续的**（continuous）；如果$F_X(x)$是$x$的阶梯函数，则称$X$是**离散的**（discrete）。

## 2.2 概率质量函数和概率密度函数

**定义：** 设$X$为概率空间$(\Omega,\mathcal{F},P)$上的离散随机变量，其**概率质量函数**（probability mass function，简记为pmf）为

$$f_X(x) = P(X = x), \forall x \in \mathbb{R}$$

**定义：** 设$X$为概率空间$(\Omega,\mathcal{F},P)$上的连续随机变量，其**概率密度函数**（probability density function，简记为pdf）是满足下式的函数：

$$F_X(x) = \int_{-\infty}^{x}f_X(t)dt, \forall x \in \mathbb{R}$$

**定理：** 函数$f_X(x)$是随机变量$X$的概率质量函数或概率密度函数，当且仅当它同时满足下列两个条件：

1）对任意$x$，都有$f_X(x) \geq 0$；

2）$\sum_{x}f_X(x) = 1$（概率质量函数）或者$\int_{-\infty}^{\infty}f_X(x) = 1$（概率密度函数）。

## 2.3 期望和方差

**定义：** 设$X$为一随机变量，$F_X(x)$为其分布函数，若$\int_{-\infty}^{\infty}\|x\|dF_X(x)$存在，则称

$$EX = \int_{-\infty}^{\infty}xdF_X(x)$$

为随机变量$X$的数学期望，简称**期望**。

当$X$为离散的，且有概率质量函数$f_X(x)$，则

$$EX = \sum_{x}xf_X(x)$$

即$EX$是$X$所有可能取值的加权平均。

当$X$为连续的，且有概率密度函数$f_X(x)$，则

$$EX = \int_{-\infty}^{\infty}xf_X(x)dx$$

数学期望存在以下两个主要性质：

1）若$C_i(i=1,2,...,n)$为常数，$X_i(i=1,2,...,n)$为随机变量，则

$$E(\sum_{i=1}^{n}C_iX_i)=(\sum_{i=1}^{n}C_iEX_i$$

2）设$X$为一随机变量，$F_X(x)$为其分布函数，$g(x)$为$x$的函数，若$E(g(X))$存在，则

$$E(g(X))=\int_{-\infty}^{\infty}g(x)dF_X(x)$$

**定义：** 令$DX\triangleq E(X-EX)^2 = EX^2-(EX)^2$，称$DX$为随机变量$X$的**方差**（variance，有时记$DX=VarX=\sigma_X^2$，$\sigma_X$称作$X$的标准差（standard deviation））。

$DX$刻画了$X$取值的分散程度。

## 2.4 矩和矩母函数

**定义：** 记

$$E(X^k) = \int_{-\infty}^{\infty}x^kdF_X(x)$$

为随机变量$X$的**$k$阶矩**（$k$-th moment），其中$k \in \mathbb{N}$。

**定义：** 随机变量$X$的**矩母函数**（moment generating function，简记为mgf），定义为

$$M_X(t)\triangleq E(e^{tX}) =  \int_{-\infty}^{\infty}e^{tx}dF_X(x)$$

如上式右边积分存在。

显然，如果$X$的$k$阶矩存在，则

$$E(X^k) = M_X^{(k)}(0)$$

矩母函数因此得名。

可以证明，随机变量$X$的矩母函数与分布函数是一一对应的（因为存在一个Laplace变换关系）。

## 2.5 常用随机变量的分布

## 2.5.1 离散型随机变量

1）几何分布（取得第一次成功需要$k$次试验）

设$0 < p < 1$，若$X$的分布律为

$$P(X=k) = (1-p)^{k-1}p, k=1,2,...$$

则称$X$是参数为$p$的**几何分布**（geometric distribution），简记为$X \sim Geo(p)$。

此时

$$EX=\frac{1}{p}, DX=\frac{1-p}{p^2}$$

2）二项分布（$n$次试验中有$k$次成功的概率）

设$0 < p < 1, n \in \mathbb{N}^*$，若$X$的分布律为

$$P(X=k) = C_n^kp^k(1-p)^{n-k}, k=0,1,2,...,n$$

则称$X$是参数为$(n,p)$的**二项分布**（binomial distribution），简记为$X \sim B(n,p)$。

此时

$$EX=np, DX=np(1-p)$$

3）泊松分布（在给定区间内发生$k$次事件的概率）

设$\lambda>0$，表示**给定区间内**事件的平均发生次数，若$X$的分布律为

$$P(X=k) = \frac{\lambda^k}{k!}e^{-\lambda}, k=0,1,2,...$$

则称$X$是参数为$\lambda$的**泊松分布**（Poisson distribution），简记为$X \sim Po(\lambda)$。

泊松分布是二项分布$n$很大而$p$很小时的一种极限形式。

此时

$$EX=\lambda, DX=\lambda$$

容易证明，如果$X \sim Po(\lambda_X)$且$Y \sim Po(\lambda_Y)$，则

$$X+Y \sim Po(\lambda_X + \lambda_Y)$$

### 2.5.2 连续型随机变量

1）均匀分布

设$a < b$，若$X$的pdf为

$$f(x)=\begin{cases}
\frac{1}{b-a} & a < x < b \\
0 & others
\end{cases}$$

则称$X$是在区间$(a,b)$上的**均匀分布**（uniform distribution），简记为$X \sim U(a,b)$。

此时

$$EX=\frac{a+b}{2}, DX=\frac{(b-a)^2}{12}$$

2）正态分布

设$\mu \in \mathbb{R}, \sigma > 0$，若$X$的pdf为

$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

则称$X$是参数为$(\mu,\sigma^2)$的**正态分布**（normal distribution），简记为$X \sim N(\mu,\sigma^2)$。

此时

$$EX=\mu, DX=\sigma^2$$

3）指数分布（在给定区间内两个相邻事件发生的间隔时间）

设$\lambda>0$，表示**单位时间内**事件的平均发生次数，若$X$的pdf为

$$f(x)=\begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}$$

则称$X$是参数为$\lambda$的**指数分布**（exponential distribution），简记为$X \sim E(\lambda)$。

此时

$$EX=\frac{1}{\lambda}, DX=\frac{1}{\lambda^2}$$

### 2.5.3 随机变量之间的一些近似规则

二项分布、泊松分布、正态分布三者之间存在一些近似关系，可以按如下规则进行近似处理：

1）当$n > 50, p < 0.1$时，$B(n,p)$可以用$Po(np)$来近似；

2）当$np > 5, n(1-p) > 5$时，$B(n,p)$可以用$N(np,np(1-p))$来近似；

3）当$\lambda > 15$时，$Po(\lambda)$可以用$N(\lambda,\lambda)$来近似。

## 2.6 多维随机变量

二维随机变量$(X,Y)$的**联合分布**（简记为joint cdf）定义为

$$F_{X,Y}(x,y) = P(X \leq x, Y \leq y)$$

$X$和$Y$的**边缘分布**为

$$F_X(x)=P(X \leq x)=\lim_{y\rightarrow \infty}F_{X,Y}(x,y)=F_{X,Y}(x,+\infty)$$

$$F_Y(x)=P(Y \leq y)=\lim_{x\rightarrow \infty}F_{X,Y}(x,y)=F_{X,Y}(+\infty,y)$$

**定义：** 称随机变量$X$和$Y$**相互独立**，若对$\forall(x,y) \in \mathbb{R}^2$，有

$$F_{X,Y}(x,y)=F_X(x)F_Y(y)$$

对于离散型二维随机变量$(X,Y)$，其**联合概率质量函数**（简记为joint pmf）为

$$f_{X,Y}(x,y)=P(X=x,Y=y)$$

对于连续型二维随机变量$(X,Y)$若存在一非负函数$f_{X,Y}(x,y)$，对$\forall (x,y) \in \mathbb{R}^2$，有

$$F_{X,Y}(x,y) = \int_{-\infty}^x\int_{-\infty}^yf_{X,Y}(u,v)dudv$$

则称$f_{X,Y}(x,y)$为$(X,Y)$的**联合概率密度函数**（简记为joint pdf）。

**定义：** 随机变量$X$和$Y$之间的**协方差**（covariance）定义为

$$Cov(X,Y) \triangleq E((X-EX)(Y-EY)) = E(XY) - EXEY$$

协方差$Cov(X,Y)$用于表示$X$和$Y$之间的线性相关关系。如果$X$与$Y$相互独立，则必然有$Cov(X,Y) \neq 0$，但是$Cov(X,Y) = 0$只代表$X$与$Y$不线性相关，并不代表它们独立。

**定义：** 若$0 < DX = \sigma_X^2 < \infty, 0 < DY = \sigma_Y^2 < \infty$，则称

$$\rho(X,Y) = \frac{Cov(X,Y)}{\sigma_X\sigma_Y}$$

为$X$和$Y$之间的$$相关系数$$（correlation coefficient）。

相关系数$\rho(X,Y)$用于表示$X$和$Y$之间的线性相关的程度，若$\rho(X,Y) = 0$，则$X$和$Y$之间不线性相关，简称**不相关**。

协方差和相关系数主要存在如下性质：

1）方差的计算可以通过下式：

$$D(\sum_{i=1}^n a_iX_i) = \sum_{i=1}^n a_i^2DX_i + 2\sum_{i<j}a_ia_jCov(X_i,X_j)$$

2）若$X_1,X_2,...,X_n$两两不相关，则

$$D(\sum_{i=1}^n X_i) = \sum_{i=1}^n DX_i$$

3）Schwarz不等式。若$X$和$Y$的二阶矩存在，则

$$|E(XY)|^2 \leq E(X^2)E(Y^2)$$

4）$\rho(X,Y) = \pm 1$，当且仅当

$$P \left( \frac{Y-EY}{\sqrt{DY}} = \pm \frac{X-EX}{\sqrt{DX}} \right) = 1$$

# 3、随机过程

设对每一个参数$t \in T$，$X(t,\omega)$是一个随机变量，我们称随机变量族$X_T = \\{X(t,\omega) \| t\in T\\}$为一个**随机过程**（stochastic process）。其中，$T\subset \mathbb{R}$，称为该随机过程的**时间集**（time set）。

容易看出，$X(\cdot,\cdot)$是定义在$T\times \Omega$上的二元单值实函数。固定$t \in T$，$X(t,\cdot)$是定义在样本空间$\Omega$上的函数，即为一个随机变量。$X(\cdot,\cdot)$ 有时也记为$X_t(\omega)$，简记为$X(t)$或$X_t$。

时间集$T$常用的有三种：

1）$T_1 = \mathbb{N} = \\{0,1,2,...\\}$；

2）$T_2 = \mathbb{Z} = \\{...,-2,-1,0,1,2,...\\}$；

3）$T_3 = [a,b]$，其中$a$可以取$-\infty$或0，$b$可以取$+\infty$。

当$T$取可数集（T1或T2）时，通常称$X_T$为随机序列。

$X_t(t\in T)$可能取值的全体之集称为该随机过程的**状态空间**，记作$S$，$S$中的元素称作状态。

To fully characterize a stochastic process $\\{X(t)\\}$, we specify the joint cdf that describes the interdependence of all random variables that define the process. Thus, we define a random vector

$$\mathbf{X} = [X(t_0),X(t_1),...,X(t_n)]$$

which can take values

$$\mathbf{x} = [x_0,x_1,...,x_n]$$

Then, for all possible $(x_1,x_2,...,x_n)$ and $(t_0,t_1,...,t_n)$, it is neccessary to specify a joint cdf:

$$F_X(x_0,...,x_n;t_0,...,t_n) = P(X(t_0)\leq x_0,...,X(t_n)\leq x_n)$$

## 3.1 平稳过程

Stationary is the property of a system to mantain its dynamic behavior invariant to time shifts. The same idea applies to stochastic processes. Formally, $\\{X(t)\\}$ is said to be a stationary process if and only if

$$\begin{equation}\label{1}
F_X(x_0,...,x_n;t_0+\tau,...,t_n+\tau) = F_X(x_0,...,x_n;t_0,...,t_n), \forall \tau \in \mathbb{R}
\end{equation}$$

The property defined through $\eqref{1}$ is somtimes referred to as strict-sence stationarity, to differentiate it from a weaker requirement called wide-sense stationarity. $\\{X(t)\\}$ is said to be a wide-sense stationary process if and only if

$$E(X(t)) = C \text{ and } E(X(t)X(t+\tau))=g(\tau), \forall \tau \in \mathbb{R}$$

where $C$ is a constant independent of $t$, and $g(\tau)$ is a function of $\tau$ but not of $t$. Thus, wide-sense staionarity requires the first two moments only (as opposed to all moments) to be independent of $t$.

## 3.2 独立过程

The simplest possible stochastic process we can consider is just a sequence of random variables $$\{X_0,X_1,...,X_n\}$$ which are mutually independent. We say that $$\{X_k\} (k=0,1,...,n)$$ is an independent process if and only if

$$F_X(x_0,...,x_n;t_0,...,t_n) = F_{X_0}(x_0;t_0)...F_{X_n}(x_n;t_n)$$


## 3.3 马尔可夫过程

Suppose we observe a chain from time $$t_0$$ up to time $$t_k$$, and let $$t_0\leq t_1\leq \cdots\leq t_{k-1}\leq t_k$$. Let us think of the observed value $$x_k$$ at $$t_k$$ as the "present state" of the chain, and of $$\{x_0,x_1,...,x_{k-1}\}$$ as its observed "past history". Then, $$\{X_{k+1},X_{k+2},...\}$$, for time instants $$t_{k+1}\leq t_{k+2}\leq \cdots$$, represents the unknown "future". In an independent chain, this future is completely independent of the past. This is a very strong property, which is partly relaxed in a Markov chain. In this type of chain, the future is conditionally independent of the past history, given the present state. In other words, the entire past history is summarized in the present state. This fact is often referred to as the memoryless property, since we need no memory of the past history (only the present) to probabilistically predict the future.

Formally, we define $\\{X(t)\\}$ to be a Markov process if 

$$\begin{equation}\label{2}
P(X(t_{k+1})\leq x_{k+1} | X(t_k)=x_k,X(t_{k-1})=x_{k-1},...,X(t_0)=x_0) \\= P(X(t_{k+1})\leq x_{k+1} | X(t_k)=x_k)
\end{equation}$$

for any $$t_0\leq t_1\leq \cdots\leq t_k\leq t_{k+1}$$. This memoryless property is also referred to as the Markov property.

In the case of a discrete-time Markov chain, state transitions are constrained to occur at time instants $0,1,2,...,k,...$, and $\eqref{2}$ may be written as

$$P(X_{k+1} = x_{k+1} | X_k=x_k,X_{k-1}=x_{k-1},...,X_0=x_0) = P(X_{k+1} = x_{k+1} | X_k=x_k)$$

The Markov property has two aspects:

__(M1)__ All past state information is irrelevant (no state memory needed);

__(M2)__ How long the process has been in the current state is irrelevant (no state age memory needed).

__(M2)__ in particular is responsible for imposing serious constraints on the nature of the random variable that specifies the time interval between consecutive state transitions (the interevent times must follow exponential distribution).

## 3.4 半马尔可夫过程

A semi-markove process is an extension of a Markov process where constraint __(M2)__ is relaxed. As a result, the interevent times are no longer constrained to be exponentially distributed. A state transition may now occur at any time, and interevent times can have arbitrary probability distributions. However, when an event does occur, the process behaves like a normal Markov chain and obeys __(M1)__: The probability of making a transition to any new state depends only on the current value of the state (and not on any past states).

## 3.5 更新过程

A renewal process is a chain $\\{N(t)\\}$ with state space $\\{0,1,2,...\\}$ whose purpose is to count state transitions. The time intervals defined by successive state transitions are assumed to be independent and characterized by a common distribution. This is the only constraint imposed on the process, as this distribution may be arbitrary. Normally, we set an initial state $N(0) = 0$. And it is clear that in such a counting process

$$N(0) \leq N(t_1) \leq \cdots \leq N(t_k)$$

for any $$0 \leq t_1 \leq \cdots \leq t_k$$.
