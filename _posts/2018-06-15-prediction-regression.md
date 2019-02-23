---
layout: post
title: "时序预测和函数回归" 
---
# 1、基本概念

时序预测是指基于历史数据对变量未来的取值进行定量估计，仅考虑时间的因素。函数回归是指确定一个结果变量与某些因素变量（可以是时间）之间的因果关系模型（即一个函数），从而可以根据因素变量的变化预测结果变量的变化，既可以定量估计取值又可以确定变化规律。

# 2、时序预测

变量随时间的变动主要由四个部分组成：长期趋势、循环变动（季节）、不规则变动、随机变动。随机变动影响可以通过平均的方法（如移动平均和指数平滑）减弱，长期趋势可以通过趋势校正的方法减弱，循环变动可以通过季节校正的方法减弱，不规则变动则比较难处理。

## 2.1 移动平均预测方法

定义$n$为移动平均基数，移动平均预测方法假定最新的$n$个观察项对于变量的估计是同等重要的。因此，在当前期间$t$，若最新的$n$个期间的数据为$$y_{t-n+1},y_{t-n+2},...,y_t$$，则期间$t+1$的估计值可计算如下：

$$y_{t+1}^* = \frac{y_{t-n+1}+y_{t-n+2}+\cdots+y_t}{n}$$

对于如何选择移动平均基数$n$，没有严格规定。若变量随时间的变化保持为合理的常数，则建议采用较大的$n$；否则，若数据变化较大，则$n$的值应该小。实际中，$n$的取值通常在2到10之间。

## 2.2 指数平滑预测方法

指数平滑预测方法考虑所有的观察数据，且对最新的观察数据给出更大的权重。定义$\alpha (0 < \alpha < 1)$为指数平滑系数，并假设过去$t$个期间的变量观察值为$$y_1,y_2,...,y_t$$，则$t+1$期间的估计值计算为：

$$y_{t+1}^* = \alpha y_t + \alpha(1-\alpha)y_{t-1} + \alpha(1-\alpha)^2y_{t-2} + \cdots $$

上述公式可以简化为：

$$\begin{align} y_{t+1}^* & = \alpha y_t + (1-\alpha)(\alpha y_{t-1} + \alpha(1-\alpha)y_{t-2} + \alpha(1-\alpha)^2y_{t-3} + \cdots \\
& = \alpha y_t + (1-\alpha)y_t^*
\end{align}
$$

按照这种方法，$$y_{t+1}^*$$可以递归地从$$y_t^*$$算出。递归公式的开始不用估计$t=1$时的$$y_1^*$$，可以让$t=2$的估计等于$t=1$时的实际数据值，即$$y_2^* = y_1$$。实际上，开始计算时可以用任何合理的方法，例如可以将开始时的估计$$y_0^*$$作为某个“合适的”多个期间的平均值。

指数平滑系数$\alpha$的选取对估计未来的预测值非常重要。$\alpha$的值大意味着最近的观察数据具有更大的权重。实际中，$\alpha$取值在0.01和0.3之间。

## 2.3 带趋势和季节校正的指数平滑预测方法

为了消除时间序列数据中长期趋势和季节变动的影响，可以在指数平滑预测方法中加入趋势和季节校正。设$T_t$表示第$t$期的趋势校正值，$I_t$为第$t$期的季节乘积数，$L$为季节周期，$S_t$为第$t$期的初始预测值，则递归过程可以表示为：

$$
\begin{align}
S_{t+1} = \alpha \frac{y_t}{I_{t-L}} + (1-\alpha)(S_t + T_t) \\
T_{t+1} = \beta (S_{t+1} - S_t) + (1-\beta)T_t \\
I_t = \gamma\frac{y_t}{S_t} + (1-\gamma)I_{t-L} \\
y_{t+1}^* = (S_{t+1}+T_{t+1})\cdot I_{t+1-L}
\end{align}
$$

其中$\alpha$、$\beta$、$\gamma$分别为预测值、趋势校正值、季节乘积数的指数平滑系数，取值可以相同，也可以不同。确定它们的原则是使预测值与实际值之间的均方差最小，一般需要借助历史数据来进行逐步逼近，也可根据经验选定，一般来说，取值范围为0.01到0.3之间。

递归的边界条件需给出初始预测值$S_1$，初始变化趋势$T_1$，以及初始季节乘积数$I_i(i=1-L,2-L,...,0)$。


# 3、函数回归

## 3.1 回归模型

### 3.1.1 一维回归模型

一维回归模型可以一般性地表示为：

$$f_{\boldsymbol{\theta}}(x) = \sum_{j=1}^b\theta_j\phi_j(x) = \boldsymbol{\theta}^T\boldsymbol{\phi}(x)$$

上式中，$$\phi_j(x)$$是基函数向量$$\boldsymbol{\phi}(x) = [\phi_1(x),\phi_2(x),...,\phi_b(x)]^T$$的第$j$个因子，$$\theta_j$$是参数向量$$\boldsymbol{\theta} = [\theta_1,\theta_2,...,\theta_b]^T$$的第$j$个因子。如果把基函数向量变成多项式的形式，即

$$\boldsymbol{\phi}(x) = [1,x,x^2,...,x^{b-1}]^T$$

就得到了多项式回归模型。

最简单的回归模型假定，相关变量是独立变量的线性函数：

$$f_{\boldsymbol{\theta}}(x) = a + bx$$

即$\boldsymbol{\phi}(x) = [1,x]^T$，$\boldsymbol{\theta} = [a,b]^T$。



### 3.1.2 多维回归模型

1）线性形式

在一维回归模型中，将一维的输入$x$扩展为$d$维的向量形式$\mathbf{x} = [x_1,x_2,...,x_d]^T$，则可得到

$$f_{\boldsymbol{\theta}}(\mathbf{x}) = \sum_{j=1}^b\theta_j\phi_j(\mathbf{x}) = \boldsymbol{\theta}^T\boldsymbol{\phi}(\mathbf{x})$$

上面的模型形式称为线性形式。

2）层级形式

更一般地，可将多维模型表示为层级形式：

$$f_{\boldsymbol{\theta}}(\mathbf{x}) = \sum_{j=1}^b\alpha_j\phi(\mathbf{x};\boldsymbol{\beta}_j)$$

上式中，$\phi(\mathbf{x};\boldsymbol{\beta})$是含有参数向量$\boldsymbol{\beta}$的基函数。

当基函数为$S$型函数，即

$$\phi(\mathbf{x};\boldsymbol{\beta}) = \frac{1}{1+\exp(-\mathbf{x}^T\boldsymbol{\omega}-\gamma)}, \boldsymbol{\beta} = (\boldsymbol{\omega}^T,\gamma)^T$$

其模仿的是人类脑细胞的输入输出函数，因此该模型也称为人工神经网络模型。实际上，该模型为一个典型三层结构的人工神经网络模型。需要注意的是，人工神经网络模型中，参数$\boldsymbol{\theta}$和函数$f_\boldsymbol{\theta}$并不是一一对应的，因此参数学习过程也比较困难。

## 3.2 回归方法

### 3.2.1 最小二乘法（Least Squares，LS）

最小二乘法一般用于线性形式的模型。回归损失函数定义为观察值与估计值之差的平方和（二次损失函数）：

$$C(\boldsymbol{\theta}) =\frac{1}{2N} \sum_{i=1}^N(\boldsymbol{\theta}^T\boldsymbol{\phi}(\mathbf{x}_i)-y_i)^2$$

目的是找到一组参数$\boldsymbol{\theta}$，使得损失函数最小。根据极小值必要条件，可得：

$$\nabla_{\boldsymbol{\theta}} C(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N(\boldsymbol{\theta}^T\boldsymbol{\phi}(\mathbf{x}_i)-y_i)\boldsymbol{\phi}(\mathbf{x}_i) =\mathbf{0}$$

可得最小二乘解为：

$$\boldsymbol{\theta} = \left[\sum_{i=1}^N\boldsymbol{\phi}(\mathbf{x}_i)\boldsymbol{\phi}^T(\mathbf{x}_i)\right]^{-1}\left[\sum_{i=1}^N\boldsymbol{\phi}(\mathbf{x}_i)y_i\right]$$

上式中，$$\sum_{i=1}^N\boldsymbol{\phi}(\mathbf{x}_i)\boldsymbol{\phi}^T(\mathbf{x}_i)$$称为信息矩阵，这关系到参数估计的程度，信息矩阵的逆矩阵$$\left[\sum_{i=1}^N\boldsymbol{\phi}(\mathbf{x}_i)\boldsymbol{\phi}^T(\mathbf{x}_i)\right]^{-1}$$是协方差矩阵，与参数估计的方差成正比，这一些文献中，协方差矩阵表示为：

$$\mathbf{P} = \left[\sum_{i=1}^N\boldsymbol{\phi}(\mathbf{x}_i)\boldsymbol{\phi}^T(\mathbf{x}_i)\right]^{-1}$$

如果该系统可辨识，则协方差矩阵肯定存在，且信息矩阵和协方差矩阵都是正定对称矩阵。


### 3.2.2 梯度下降法（Gradient Descent, GD）

考虑一般回归模型形式，损失函数可以通用性地表示为：

$$C(\boldsymbol{\theta}) =\frac{1}{N} \sum_{i=1}^NJ(f_{\boldsymbol{\theta}}(\mathbf{x}_i), y_i)$$

因为目标是使得损失函数最小，可以采用梯度下降的方式来进行参数更新迭代。参数的梯度方向可以表示为：

$$\nabla_{\boldsymbol{\theta}} C(\boldsymbol{\theta}) =\frac{1}{N} \sum_{i=1}^N\frac{\partial J(f_{\boldsymbol{\theta}}(\mathbf{x}_i), y_i)}{\partial f_{\boldsymbol{\theta}}(\mathbf{x}_i)}\frac{\partial f_{\boldsymbol{\theta}}(\mathbf{x}_i)}{\partial \boldsymbol{\theta}}$$

参数更新公式为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} C(\boldsymbol{\theta}) = \boldsymbol{\theta} - \frac{\eta}{N} \sum_{i=1}^N\frac{\partial J(f_{\boldsymbol{\theta}}(\mathbf{x}_i), y_i)}{\partial f_{\boldsymbol{\theta}}(\mathbf{x}_i)}\frac{\partial f_{\boldsymbol{\theta}}(\mathbf{x}_i)}{\partial \boldsymbol{\theta}}$$


其中$\eta$为更新步长，又称学习速率。在机器学习中的几乎所有算法都采用这种结构，朝着损失函数梯度下降的方向来不断更新参数。

1）批量梯度下降法（Batch Gradient Descent，BGD）

批量梯度下降法是梯度下降法最原始的形式，即每一步梯度迭代都使用所有样本（总共N个样本）来进行更新，其优点是能得到局部最优解，但缺点是当样本数量很多时，训练过程缓慢。

2）随机梯度下降法（Stochastic Gradient Descent，SGD）

随机梯度下降法的具体思路是在每一步梯度迭代时都使用一个随机样本来进行更新。在样本量极其大的情况下，可能不用使用所有样本就可以获得一个代价值在可接受范围内的模型。因此随机梯度法的优点是训练速度快，但准确度会下降，即获取的最优解不一定是局部最优解（当迭代次数足够多时，它会趋于局部最优解）。

3）小批量梯度下降法（Mini-Batch Gradient Descent，MBGD）

小批量梯度下降法的思路是在每一步梯度迭代时使用一个小批量样本（大于1小于N的数量）来进行更新。该方法克服了上述两种方法的缺点，又同时兼顾两种方法的优点。


#### 3.2.2.1 梯度下降法的一些问题

虽然梯度下降法效果很好，并广泛使用，但是也存在着问题和挑战：

1）选择一个合理的学习速率很难。如果学习速率过小，则收敛速度很慢；如果学习速率过大，那么会阻碍收敛，即在极值点附近会振荡。通常采用的方法是学习速率调度（leanring rate schedules），即在每次更新过程中对学习速率进行调整。一般使用某种事先设定的策略或者每次迭代中衰减一个较小的值。无论哪种调整方法，都需要事先进行固定设置，这便无法自适应每次学习的数据集特点。

2）模型所有的参数每次更新都是使用相同的学习速率。如果数据特征是稀疏的或者每个特征有着不同的取值统计分布，那么便不能在每次更新中每个参数都使用相同的学习速率，那些很少出现的特征应该使用一个相对较大的学习速率。

3）对于非凸目标函数，容易陷入那些次优的局部极值点中，如在人工神经网络的学习过程中。而更严重的问题还不是局部极值点，而是鞍点（不是极值点的驻点）。

#### 3.2.2.2 梯度下降法的改进

我们假设梯度下降法要优化的参数为$\theta$，损失函数在$\theta$处的导数为$$\nabla_{\theta}$$，则一般的梯度下降法表示为：

$$\theta += - learning\_rate * \nabla_{\theta}$$

这种更新可能会让学习过程比较曲折，一些学者提出了各种加速的更新方法。

1）Momentum更新方法

Momentum更新方法的思想是对梯度方向进行累积，对那些当前梯度方向与上一次梯度方向相同的参数，进行动量加强，而对于那些梯度方向与上一次梯度方向不同的参数，进行动量削减。更新过程表示为：

$$accumulation = momentum\_coefficient * accumulation + \nabla_{\theta}$$

$$\theta += - learning\_rate * accumulation$$

2) AdaGrad更新方法

这种方法是在学习率上面动手脚，使得每个参数更新都会有自己与众不同的学习率。其更新过程如下：

$$cache += {\nabla_{\theta}}^2$$

$$\theta += - learning\_rate * \nabla_{\theta} / \sqrt{cache}$$

可以看出实际上是增加了一个附加变量$cache$来缩放梯度，并且不停地增加这一附加变量。因为对每个参数计算其相应梯度的平方和，并将其平方根去除学习速率，所以可以对每个参数自适应不同的学习速率：对稀疏特征，得到更大的学习更新，对非稀疏特征，得到较小的学习更新，因此该改进尤其适合处理稀疏特征数据。

3）RMSProp更新方法

RMSProp更新方法是由AdaGrad演化过来的。因为在学习过程中，我们需要持续的活力来不断更新数据，而不是衰退到停止。因此可以修改更新过程为：

$$cache = decay\_rate*cache + (1-decay\_rate)*\nabla_{\theta}^2$$

$$\theta += - learning\_rate * \nabla_{\theta} / \sqrt{cahche}$$

其仍然保持了AdaGrad对更新步长的补偿效果，但是不会再发生更新停止的情况。

4）Adam更新方法

Adam可以认为是AdaGrad方法和Momentum方法的结合，可称之为“极品”。其更新过程如下：

$$m = beta1 * m + (1-beta1) * \nabla_{\theta}$$

$$v = beta2*v + (1-beta2)*\nabla_{\theta}^2$$

$$\theta += - learning\_rate * m / \sqrt{v}$$


### 3.2.3 自然梯度下降法（Natural Gradient Descent, NGD）

梯度下降法只考虑了在梯度方向对参数进行更新，并没有考虑模型层面的更新程度，当采用随机梯度下降法或小批量梯度下降法时损失函数可能会出现较大的波动，甚至发散。而自然梯度下降法则能够抵抗这种波动性，是一种稳定的优化方法。

在自然梯度法中，我们假定模型的输出$$f_{\boldsymbol{\theta}}(\mathbf{x})$$是一个概率分布，于是两个模型之间的差异可以用KL散度来表示：

$$KL(f_{\boldsymbol{\theta}_1}\| f_{\boldsymbol{\theta}_2}) = \sum_{\boldsymbol{x}} f_{\boldsymbol{\theta}_1}(\mathbf{x})\log\frac{f_{\boldsymbol{\theta}_1}(\mathbf{x})}{f_{\boldsymbol{\theta}_2}(\mathbf{x})}$$

按照梯度下降法的思想，我们定义每一轮的迭代优化都要解决这样一个子问题：

$$\begin{align} &\min_{\Delta\boldsymbol{\theta}} C(\boldsymbol{\theta} + \Delta\boldsymbol{\theta}) \\
    s.t.  \ 	& \|\Delta\boldsymbol{\theta}\| < \epsilon
\end{align}$$

将损失函数进行一阶泰勒展开，问题就变为：

$$\begin{align} &\min_{\Delta\boldsymbol{\theta}} C(\boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}}C(\boldsymbol{\theta}) \Delta\boldsymbol{\theta} \\
    s.t.  \ 	& \|\Delta\boldsymbol{\theta}\| < \epsilon
\end{align}$$

对目标函数求导，并对更新量做一定的限制，就可以得到梯度下降法。如果我们改为对模型的距离进行约束，就得到另一个优化问题：

$$\begin{align} &\min_{\Delta\boldsymbol{\theta}} C(\boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}}C(\boldsymbol{\theta}) \Delta\boldsymbol{\theta} \\
    s.t.  \ 	& KL(f_{\boldsymbol{\theta}}\| f_{\boldsymbol{\theta} + \Delta\boldsymbol{\theta}}) < \epsilon
\end{align}$$

这就是自然梯度法的优化形式。可以看出，有了模型层面的约束，每一轮迭代无论参数发生多大的变化，模型的变化都会限制在一定的范围内，因此无论我们使用什么样的模型，这个约束都会起效果，因此这个约束是具有普适性的，在任何模型上都能发挥同样稳定的效果。

