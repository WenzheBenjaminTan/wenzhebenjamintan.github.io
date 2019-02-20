---
layout: post
title: "优化理论2" 
---
# 1、无约束优化问题

无约束优化问题可以表示为如下形式：

$$
\begin{align}
	\min f(\mathbf{x}) \\
	s.t.\ \mathbf{x} \in \mathbb{R}^n 
\end{align}
$$

其中，要最小化的目标函数$f:\mathbb{R}^n\rightarrow \mathbb{R}$ 是一个多元实值函数，决策变量$$\mathbf{x} = (x_1,x_2,...,x_n)^T$$ 不受任何约束。

## 1.1 局部极小点的条件

**定理：局部极小点的一阶必要条件。** 多元实值函数$f$在$\mathbb{R}^n$上一阶连续可微，即$f \in \mathcal{C}^1$，如果$\mathbf{x}^*$是其局部极小点，则有：

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

**定理：局部极小点的二阶必要条件。** 多元实值函数$f$在$\mathbb{R}^n$上二阶连续可微，即$f \in \mathcal{C}^2$，如果$\mathbf{x}^*$是其局部极小点，则有：

$$\text{1、}\nabla f(\mathbf{x}^*) = \mathbf{0} $$

$$\text{2、}D^2f(\mathbf{x}^*) \geq 0 \text{（半正定）}$$

**定理：局部极小点的二阶充分条件。** 多元实值函数$f$在$\mathbb{R}^n$上二阶连续可微，即$f \in \mathcal{C}^2$，如果同时满足：

$$\text{1、}\nabla f(\mathbf{x}^*) = \mathbf{0} $$

$$\text{2、}D^2f(\mathbf{x}^*) > 0 \text{（正定）}$$

则$\mathbf{x}^*$是函数$f$的一个严格局部极小点。

## 1.2 梯度方法

多元实值函数$f$在$\mathbb{R}^n$上一阶连续可微，即$f \in \mathcal{C}^1$，梯度方向$\nabla f(\mathbf{x})$就是函数$f$在$\mathbf{x}$处增长最快的方向。反之，梯度负方向$-\nabla f(\mathbf{x})$就是函数$f$在$\mathbf{x}$处下降最快的方向。因此，如果需要搜索函数的极小点，梯度负方向是一个很好的搜索方向。

令$\mathbf{x}^{(0)}$作为初始搜索点，并沿着梯度负方向构造一个新的点$\mathbf{x}^{(0)}-\alpha\nabla f(\mathbf{x}^{(0)})$，由泰勒一阶展开可得

$$f(\mathbf{x}^{(0)}-\alpha\nabla f(\mathbf{x}^{(0)})) = f(\mathbf{x}^{(0)}) - \alpha \|\nabla f(\mathbf{x}^{(0)})\|^2 + o(\alpha)$$

因此，如果$\nabla f(\mathbf{x}^{(0)})\neq \mathbf{0}$，那么当$\alpha > 0$足够小时，有

$$f(\mathbf{x}^{(0)}-\alpha\nabla f(\mathbf{x}^{(0)})) < f(\mathbf{x}^{(0)})$$

成立。搜索方向$\mathbf{d}^{(k)} = -\nabla f(\mathbf{x}^{(0)})$这为极小点搜索提供了一个很好的启发。

给定一个搜索点$\mathbf{x}^{(k)}$，可以通过迭代的方式得到$\mathbf{x}^{(k+1)}$：

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k \mathbf{d}^{(k)}$$

其中$\alpha_k$是一个正实数，称为步长。

迭代的终止条件可以是

$$\frac{|f(\mathbf{x}^{(k+1)}) - f(\mathbf{x}^{(k)})|}{\max\{1,|f(\mathbf{x}^{(k)})|\}} < \epsilon$$

或

$$\frac{\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|}{\max\{1,\|\mathbf{x}^{(k)}\|\}} < \epsilon$$

其中$\epsilon > 0$是一个预设的阈值。


### 1.2.1 基本的梯度下降法

当步长始终是固定的，即$$\alpha_k = \alpha$$，则称该方法是基本的梯度下降法（或固定步长的梯度下降法）。

### 1.2.2 最速下降法

对于最速下降法（steepest descent），$\alpha_k$的值通过使函数

$$\phi_k(\alpha) = f(\mathbf{x}^{(k)} + \alpha \mathbf{d}^{(k)}) (\alpha > 0)$$

达到最小的方式来确定。根据微分链式法则，计算函数$\phi_k$的一阶导数：

$$\phi'_k(\alpha) = -(\nabla f(\mathbf{x}^{(k)}))^T\nabla f(\mathbf{x}^{(k)}-\alpha\nabla f(\mathbf{x}^{(k)}))$$

当$$\nabla f(\mathbf{x}^{(k)}) \neq \mathbf{0}$$时，令$\phi'_k(\alpha) = 0$可以求得$\phi_k(\alpha)$的极小点，即$\alpha_k$。


## 1.2 牛顿法

在确定搜索方向时，最速下降法只用到了目标函数的一阶导数。这种方式并非总是高效的。在某些情况下，如果能够在迭代过程中引入高阶导数，其效率可能优于最速下降法。牛顿法就是如此，它同时使用一阶和二阶导数来确定搜索方向。

### 1.2.1 基本的牛顿法

给定一个迭代点后，首先构造一个二次型函数，其与目标函数在该点处的一阶和二阶导数相等，以此可以作为目标函数的近似；接下来求该二次型函数的极小点，以此作为下一次迭代的起始点；重复以上过程，以求得目标函数的极小点。这就是基本的牛顿法的思路。如果目标函数本身就是二次型函数，那么构造的近似函数与目标函数是完全一致的。如果目标函数不是二次型函数，那么近似函数得到的极小点给出的是目标函数极小点的大体位置。当初始点与目标函数的极小点足够接近时，基本的牛顿法的效率要优于最速下降法。

多元实值函数$f$在$\mathbb{R}^n$上二阶连续可微，即$f \in \mathcal{C}^2$，将函数$f$在点$\mathbf{x}^{(k)}$处进行二次展开，并忽略二次以上的项，可得到二次型近似函数：

$$q(\mathbf{x}) = f(\mathbf{x}^{(k)}) + (\mathbf{x} - \mathbf{x}^{(k)})^T \nabla f(\mathbf{x}^{(k)}) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^{(k)})^T D^2f(\mathbf{x}^{(k)})(\mathbf{x} - \mathbf{x}^{(k)})$$

对函数$q$应用局部极小点的一阶必要条件，可得

$$\nabla q(\mathbf{x}) = \nabla f(\mathbf{x}^{(k)}) + D^2f(\mathbf{x}^{(k)})(\mathbf{x} - \mathbf{x}^{(k)}) = \mathbf{0}$$

如果$D^2f(\mathbf{x}^{(k)})$可逆，可求得函数$q$的局部极小点：

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - (D^2f(\mathbf{x}^{(k)}))^{-1}\nabla f(\mathbf{x}^{(k)})$$

这就是基本的牛顿法的迭代公式。

如果初始点$\mathbf{x}^{(0)}$离极小点$\mathbf{x}^*$较远，原来的牛顿法不一定具备下降特性，也就是说会出现$f(\mathbf{x}^{(k+1)}) \geq f(\mathbf{x}^{(k)})$
。

### 1.2.2 牛顿下山法

如果$\nabla f(\mathbf{x}^{(k)}) \neq \mathbf{0}$且$D^2f(\mathbf{x}^{(k)}) > 0$，那么从$\mathbf{x}^{(k)}$到$\mathbf{x}^{(k+1)}$的搜索方向$\mathbf{d}^{(k)} = - (D^2f(\mathbf{x}^{(k)}))^{-1}\nabla f(\mathbf{x}^{(k)})$肯定是一个下降方向。因此，可对牛顿法进行如下修正：

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k \mathbf{d}^{(k)}$$

$\alpha_k$的值通过使函数

$$\phi_k(\alpha) = f(\mathbf{x}^{(k)} + \alpha \mathbf{d}^{(k)}) (\alpha > 0)$$

达到最小的方式来确定。根据微分链式法则，计算函数$\phi_k$的一阶导数：

$$\phi'_k(\alpha) = -(D^2f(\mathbf{x}^{(k)}))^{-1}\nabla f(\mathbf{x}^{(k)})^T\nabla f(\mathbf{x}^{(k)} - \alpha(D^2f(\mathbf{x}^{(k)}))^{-1}\nabla f(\mathbf{x}^{(k)}))$$

当$\nabla f(\mathbf{x}^{(k)}) \neq \mathbf{0}$且$D^2f(\mathbf{x}^{(k)}) > 0$时，令$\phi'_k(\alpha) = 0$可以求得$\phi_k(\alpha)$的极小点，即$\alpha_k$。这就是牛顿下山法。


### 1.2.3 Levenberg-Marquardt（莱文贝格-马夸特）修正


如果海森（Hessian）矩阵$D^2f(\mathbf{x}^{(k)})$不正定，那么搜索方向$\mathbf{d}^{(k)} = - (D^2f(\mathbf{x}^{(k)}))^{-1}\nabla f(\mathbf{x}^{(k)})$可能不是一个下降方向，为了保证每次产生的方向是下降方向，可以对牛顿法进行Levenberg-Marquardt 修正如下：

$$\mathbf{d}^{(k)} = - (D^2f(\mathbf{x}^{(k)}) + \mu_k\mathbf{I})^{-1}\nabla f(\mathbf{x}^{(k)})$$

其中$\mu_k\geq 0$，当其足够大时，总能保证矩阵$$D^2f(\mathbf{x}^{(k)}) + \mu_k\mathbf{I}$$是正定的，因此$\mathbf{d}^{(k)}$是一个下降方向。

有意思的是，令$$\mu_k\rightarrow 0$$，牛顿法的Levenberg-Marquardt 修正就相当于原始的牛顿法；令$$\mu_k\rightarrow \infty$$，Levenberg-Marquardt 修正又会表现出步长较小时梯度方法的特性。

给定一个$$\mu_k$$，可以向牛顿下山法那样计算得到相应的$$\alpha_k$$，即$$\alpha_k = arg\min_{\alpha>0} f(\boldsymbol{x}^{(k)} + \alpha\boldsymbol{d}^{(k)})$$，于是可得新的迭代公式为：

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha_k (D^2f(\mathbf{x}^{(k)}) + \mu_k\mathbf{I})^{-1}\nabla f(\mathbf{x}^{(k)})$$

一开始可以为$\mu_k$选择较小的值（甚至可以等于0），然后逐渐缓慢增加，直到出现下降特性，即$f(\mathbf{x}^{(k+1)}) < f(\mathbf{x}^{(k)})$。

牛顿法中每轮迭代均需要涉及到海森矩阵及其求逆的计算，对于高维问题（$n$很大）计算复杂度相当高，这是牛顿法的缺陷之一。共轭方向法（conjugate direction method）和拟牛顿法（quasi-Newton method）可以改善这一问题，但收敛性会下降。

## 1.3 共轭方向法

一般情况下，共轭方向法的性能优于最速下降法，但不如牛顿法。

### 1.3.1 基本的共轭方向法

基本的共轭方向法针对$n$维二次型函数的最小化问题：

$$f(\boldsymbol{x}) = \frac{1}{2}\boldsymbol{x}^T\boldsymbol{Q}\boldsymbol{x} - \boldsymbol{x}^T \boldsymbol{b}$$

其中，$$\boldsymbol{Q} = \boldsymbol{Q}^T > 0$$，$\boldsymbol{x} \in \mathbb{R}^n$。如果$$\mathbb{R}^n$$中的两个非零方向$$\boldsymbol{d}^{(i)}$$和$$\boldsymbol{d}^{(j)}$$满足 ${\boldsymbol{d}^{(i)}}^T\boldsymbol{Q}\boldsymbol{d}^{(j)}=0$ ，则称它们关于$$\boldsymbol{Q}$$是共轭的。容易证明，如果方向$$\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ..., \boldsymbol{d}^{(k)} \in \mathbb{R}^n, k\leq n-1$$ 非零，且是关于$$\boldsymbol{Q}$$两两共轭的，那么它们是线性无关的。

值得注意的是，由于$$\boldsymbol{Q} > 0$$，因此函数有一个全局极小点，可通过求解$$\boldsymbol{Qx} = \boldsymbol{b}$$得到。

对于任意初始点$$\boldsymbol{x}^{(0)}$$，给定一组关于$$\boldsymbol{Q}$$共轭的方向$$\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ..., \boldsymbol{d}^{(n-1)}$$，都可以通过如下$n$次迭代后得到全局极小点：

$$\boldsymbol{g}^{(k)} = \nabla f(\boldsymbol{x}^{(k)}) = \boldsymbol{Q}\boldsymbol{x}^{(k)} - \boldsymbol{b}$$

$$\alpha_k = -\frac{(\boldsymbol{g}^{(k)})^T\boldsymbol{d}^{(k)}}{(\boldsymbol{d}^{(k)})^T\boldsymbol{Q}\boldsymbol{d}^{(k)}}$$

$$\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)} + \alpha_k \boldsymbol{d}^{(k)}$$

可以证明，对于任意$$0\leq k \leq n-1$$，$$\boldsymbol{g}^{(k+1)}$$正交于由$$\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ..., \boldsymbol{d}^{(k)} $$张成的子空间中的任意向量。进而可得

$$f(\boldsymbol{x}^{(k+1)}) = \min_{\alpha_0,...,\alpha_k} f(\boldsymbol{x}^{(0)} + \sum_{i=0}^k\alpha_i \boldsymbol{d}^{(i)})$$

换言之，如果记

$$\mathcal{V}_k = \boldsymbol{x}^{(0)} + span[\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ..., \boldsymbol{d}^{(k)}]$$

则有$$f(\boldsymbol{x}^{(k+1)}) = \min_{\boldsymbol{x}\in \mathcal{V}_k} f(\boldsymbol{x})$$。随着$k$的增大，$$span[\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ..., \boldsymbol{d}^{(k)}]$$不断扩张；当$k$足够大时，$$\boldsymbol{x}^*$$将位于$$\mathcal{V}_k$$中（最多需要$n$步）。

基本的共轭方向法效率很高，但是前提是必须能够给定一组$$\boldsymbol{Q}$$共轭方向。

### 1.3.2 共轭梯度法

共轭梯度法不需要预先给定$$\boldsymbol{Q}$$共轭方向，而是从初始方向（一般初始化为梯度下降方向）开始进行迭代，不断产生$$\boldsymbol{Q}$$共轭方向。在每次迭代中，利用上一搜索方向和当前的梯度下降方向之间的线性组合构造一个新方向，使其与前面已经产生的搜索方向组成$$\boldsymbol{Q}$$共轭方向。对于一个$n$维正定二次型函数，最多经过$n$次迭代，即可得到极小点。

共轭梯度法的步骤如下：

1）令$k=0$，选择初始值$\boldsymbol{x}^{(0)}$。

2）计算$$\boldsymbol{g}^{(0)} = \nabla f(\boldsymbol{x}^{(0)})$$，如果$$\boldsymbol{g}^{(0)} = 0$$，停止迭代；否则，令$$\boldsymbol{d}^{(0)}= -\boldsymbol{g}^{(0)} $$。

3）计算$$\alpha_k = arg\min_{\alpha>0} f(\boldsymbol{x}^{(k)} + \alpha\boldsymbol{d}^{(k)}) = -\frac{(\boldsymbol{g}^{(k)})^T\boldsymbol{d}^{(k)}}{(\boldsymbol{d}^{(k)})^T\boldsymbol{Q}\boldsymbol{d}^{(k)}}$$。

4）计算$$\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)} + \alpha \boldsymbol{d}^{(k)}$$；

5）计算$$\boldsymbol{g}^{(k+1)} = \nabla f(\boldsymbol{x}^{(k+1)})$$，如果$$\boldsymbol{g}^{(k+1)} = 0$$，停止迭代；否则，计算$$\beta_k = = \frac{(\boldsymbol{g}^{(k+1)})^T\boldsymbol{Q}\boldsymbol{d}^{(k)}}{(\boldsymbol{d}^{(k)})^T\boldsymbol{Q}\boldsymbol{d}^{(k)}}$$，令$$\boldsymbol{d}^{(k+1)}= -\boldsymbol{g}^{(k+1)} + \beta_k \boldsymbol{d}^{(k)} $$。

6）令$k=k+1$，回到第3步。

### 1.3.2 针对非二次型问题的共轭梯度法修正

共轭梯度法是共轭方向法的一种具体实现，能够保证在$n$步之内迭代求得$n$维正定二次型函数的极小点。如果将函数$$f(\boldsymbol{x}) = \frac{1}{2}\boldsymbol{x}^T\boldsymbol{Q}\boldsymbol{x} - \boldsymbol{x}^T \boldsymbol{b}$$ 视为某个目标函数在当前迭代点的泰勒展开式的近似，就可以将共轭梯度法推广应用到一般的非线性函数，其中$$\boldsymbol{Q}$$为目标函数在当前迭代点的黑塞矩阵。对于二次型函数，其黑塞矩阵是常数矩阵。但是，对于一般的非线性函数而言，每次迭代都必须重新计算黑塞矩阵，这需要非常大运算量。因此，有必要对共轭梯度法进行修改，消除每次迭代中求黑塞矩阵的环节，使得算法更加高效。

注意到$\boldsymbol{Q}$矩阵只出现在计算$$\alpha_k$$和$$\beta_k$$的环节。

而由

$$\alpha_k = arg\min_{\alpha>0} f(\boldsymbol{x}^{(k)} + \alpha\boldsymbol{d}^{(k)})$$

可知，$$\alpha_k$$可替换为一维搜索的结果。

因此，只需要考虑如何在$$\beta_k$$的计算过程中去掉$$\boldsymbol{Q}$$就可以。

有以下三种修正公式：

1）Hestenes-Stiefel公式

$$\beta_k = \frac{(\boldsymbol{g}^{(k+1)})^T[\boldsymbol{g}^{(k+1)} - \boldsymbol{g}^{(k)}]}{(\boldsymbol{d}^{(k)})^T[\boldsymbol{g}^{(k+1)} - \boldsymbol{g}^{(k)}]}$$

2）Polak-Ribiere公式

$$\beta_k = \frac{(\boldsymbol{g}^{(k+1)})^T[\boldsymbol{g}^{(k+1)} - \boldsymbol{g}^{(k)}]}{(\boldsymbol{g}^{(k)})^T \boldsymbol{g}^{(k)}}$$

3）Fletcher-Reeves公式

$$\beta_k = \frac{(\boldsymbol{g}^{(k+1)})^T\boldsymbol{g}^{(k+1)}}{(\boldsymbol{g}^{(k)})^T \boldsymbol{g}^{(k)}}$$

上述三个公式都是根据二次型目标函数推论出来的，因此对于二次型问题，这三个公式是等价的。但是，当目标函数为非二次型函数时，这三个公式并不一致。究竟哪个公式更好，取决于具体的目标函数。

对于非二次型问题，共轭梯度法通常不会在$n$步内收敛到极小点，随着迭代的进行，搜索方向也将不再是$$\boldsymbol{Q}$$共轭方向。常用的修正方法是每经过几次迭代后（如$n$或$n+1$次），重新将搜索方向初始化为目标函数的梯度下降方向，然后继续搜索直到满足停止规则。需要注意的是，共轭梯度法在搜索时有可能出现不是下降方向的情况，这时候可以强制改用梯度下降方向来代替。



## 1.4 拟牛顿法

牛顿法的一个缺陷是涉及海森矩阵及其求逆运算。为了避免这一缺陷，可以通过设计一个近似矩阵$$\boldsymbol{H}_k$$来代替$$(D^2f(\mathbf{x}^{(k)}))^{-1}$$，从而使得迭代公式变为：

$$\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)} - \alpha \boldsymbol{H}_k\boldsymbol{g}^{(k)}$$

$$\boldsymbol{H}_k$$随着迭代的进行不断更新，使其至少具备$$(D^2f(\mathbf{x}^{(k)}))^{-1}$$的部分性质，这就是拟牛顿法的基本思路。

在$\boldsymbol{x}^{(k)}$处对$f$进行一阶泰勒展开，并将$\boldsymbol{x}^{(k+1)}$代入，可得：

$$\begin{align}
f(\mathbf{x}^{(k+1)}) &= f(\mathbf{x}^{(k)}) +(\boldsymbol{g}^{(k)})^T(\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}) + o(\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|) \\
		&= f(\mathbf{x}^{(k)}) - \alpha (\boldsymbol{g}^{(k)})^T \boldsymbol{H}_k\boldsymbol{g}^{(k)} + o(\|\boldsymbol{H}_k\boldsymbol{g}^{(k)}\|\alpha)
\end{align}$$

当$\alpha$足够小时，上式等号右侧第2项主导了第3项。因此，当$\alpha$比较小时，为了保证函数$f$从$\boldsymbol{x}^{(k)}$到$\boldsymbol{x}^{(k+1)}$是下降的，必须有：

$$(\boldsymbol{g}^{(k)})^T \boldsymbol{H}_k\boldsymbol{g}^{(k)} > 0$$

要保证上式成立，最简单的方式就是保证$$\boldsymbol{H}_k$$始终是对称正定的。

对于每一步迭代，我们可以取$$\alpha_k = arg \min_{\alpha > 0} f(\boldsymbol{x}^{(k)} - \alpha \boldsymbol{H}_k\boldsymbol{g}^{(k)}$$，从而得到

$$\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)} - \alpha_k \boldsymbol{H}_k\boldsymbol{g}^{(k)}$$

对于拟牛顿法来说，当目标函数是二次型函数时，$$\boldsymbol{H}_1, \boldsymbol{H}_2, ...$$必须满足

$$\boldsymbol{H}_{k+1} \Delta \boldsymbol{g}^{(i)} = \Delta \boldsymbol{x}^{(i)}, 0\leq i \leq k$$

其中，$$\Delta \boldsymbol{x}^{(i)} = \boldsymbol{x}^{(i+1)} - \boldsymbol{x}^{(i)} = \alpha_i \boldsymbol{d}^{(i)}$$，$$\Delta \boldsymbol{g}^{(i)} = \boldsymbol{g}^{(i+1)} - \boldsymbol{g}^{(i)} = \boldsymbol{Q} \Delta \boldsymbol{x}^{(i)}$$（$$\boldsymbol{Q}$$为二次型函数的实际黑塞矩阵）。实际上，拟牛顿法也是一种共轭方向法，即$$\boldsymbol{d}^{(0)}, \boldsymbol{d}^{(1)}, \boldsymbol{d}^{(2)}, ...$$是$\boldsymbol{Q}$共轭的。



## 1.5 迭代方法的收敛性

迭代方法通过迭代过程产生一个迭代点序列，除了初始迭代点，每个迭代点都是从前一个迭代点中按照一定的方式衍生而来的。

如果对于任意起始点，算法都能够保证产生一组迭代点序列，最终收敛到满足局部极小点一阶必要条件的点，那么该算法就被认为是全局收敛的。如果要求初始点足够靠近极小点，算法产生的迭代点序列才能收敛到满足局部极小点一阶必要条件的点，那么算法就不是全局收敛的，而是局部收敛的。对于全局或局部收敛的算法，评价算法究竟有多快收敛到极小点的指标是收敛率。

最速下降法在最坏情况下的收敛阶数为1，而对于牛顿法，如果初始点的选择比较合适，收敛阶数至少为2。


# 2、有约束优化问题

## 2.1 仅含等式约束的优化问题

拉格朗日乘子（Lagrange multipliers）法是一种寻找多元函数在一组约束下的极值的方法。通过引入拉格朗日乘子，可以将有$n$个变量与$k$个约束条件的优化问题转化为具有$n+k$个变量的无约束优化问题求解。

假定$\mathbf{x}$为$n$维向量，欲寻找$\mathbf{x}$的某个取值$\mathbf{x}^*$，使得目标函数$f(\mathbf{x})$最小且同时满足$h(\mathbf{x}) = 0$的约束。从几何角度看，该问题的目标是在由方程$h(\mathbf{x}) = 0$确定的$n-1$维曲面上寻找能使目标函数$f(\mathbf{x})$最小的点。不难得出如下结论：

1）对于约束曲面上的任意点$\mathbf{x}$，该点的约束梯度$\nabla h(\mathbf{x})$正交于约束曲面；

2）在最优点$\mathbf{x}^*$，目标函数在该点的梯度$$\nabla f(\mathbf{x}^*)$$正交于约束曲面（目标函数等值线与约束曲面相切）。

由此可知，在最优点$\mathbf{x}^*$，$\nabla h(\mathbf{x})$和$$\nabla f(\mathbf{x})$$的方向要么相同要么相反，即存在实数$\lambda$使得

$$\begin{equation}\label{1} \nabla f(\mathbf{x}^*) + \lambda\nabla h(\mathbf{x}^*) = \mathbf{0}\end{equation}$$

其中$\lambda$就称为拉格朗日乘子。定义拉格朗日函数为：

$$L(\mathbf{x},\lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$$

不难发现，将其对$\mathbf{x}$求偏导并置零即得式 \eqref{1}；将其对$\lambda$求偏导并置零即得约束条件$h(\mathbf{x}) = 0$。于是，原约束优化问题可转化为对拉格朗日函数$L(\mathbf{x},\lambda)$的无约束优化问题。

## 2.2 含不等式约束的优化问题

考虑不等式约束$g(\mathbf{x}) \leq 0$。定义拉格朗日函数为：

$$L(\mathbf{x},\mu) = f(\mathbf{x}) + \mu g(\mathbf{x})$$

取得的最优点$$\mathbf{x}^*$$或在$g(\mathbf{x}) < 0$的区域中，或在边界$g(\mathbf{x}) = 0$上。对于$$g(\mathbf{x}^*) < 0$$的情形，约束$g(\mathbf{x}) \leq 0$不起作用，可直接通过条件$\nabla f(\mathbf{x}) = 0$来获得最优点，这等价于将$\mu$置零，然后让$L(\mathbf{x},\mu)$对$\mathbf{x}$求偏导并置零来得到最优点；对于$$g(\mathbf{x}^*) = 0$$的情形，类似于上面等式约束的分析，但需注意的是，如果可行域中是存在满足$g(\mathbf{x}) < 0$的点的，此时$$\nabla g(\mathbf{x}^*)$$和$$\nabla f(\mathbf{x}^*)$$的方向一定是相反的，即存在常数$\mu\geq 0$使得$$\nabla f(\mathbf{x}^*) + \mu\nabla g(\mathbf{x}^*) = \mathbf{0}$$。整合以上两种情形，必满足$\mu g(\mathbf{x}) = 0$。

因此，在约束$g(\mathbf{x}) \leq 0$下最小化$f(\mathbf{x})$，可转化为如下约束情况下最小化拉格朗日函数$L(\mathbf{x},\mu)$：

1）$g(\mathbf{x}) \leq 0$；

2）$\mu\geq 0$；

3）$\nabla f(\mathbf{x}) + \mu\nabla g(\mathbf{x}) = \mathbf{0}$；

4）$\mu g(\mathbf{x}) = 0$。


上面条件统称为Karush-Kuhn-Tucker（简称KKT）条件，其给出了某个点是约束问题局部极小点应该满足的一阶必要条件。

将上述做法推广到多个约束，考虑一般情况的数学规划问题：

$$
\begin{align}
\label{2} \min f(\mathbf{x}) \\
	s.t.\ & h_i(\mathbf{x}) = 0\ (i = 1,2,...,m) \\
	& g_j(\mathbf{x}) \leq 0\ (j = 1,2,...,p) \\
\end{align}
$$

其可行域中至少存在一个绝对可行点（即可以让所有不等式约束都不取等号的可行点），引入拉格朗日乘子$\boldsymbol{\lambda} = (\lambda_1,\lambda_2,...,\lambda_m)^T$和$\boldsymbol{\mu} = (\mu_1,\mu_2,...,\mu_p)^T$，需要最小化的拉格朗日函数为：

$$\begin{equation}\label{3}L(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^m\lambda_i h_i(\mathbf{x}) +\sum_{j=1}^p\mu_j g_j(\mathbf{x})\end{equation}$$

由各约束引入的KKT条件为：

1）$h_i(\mathbf{x}) = 0$ （$i=1,2,...,m$）；

2）$g_j(\mathbf{x}) \leq 0$ （$j=1,2,...,p$）；

3）$\mu_j\geq 0$ （$j=1,2,...,p$）；

4）$\nabla f(\mathbf{x}) + \sum_{i=1}^m\lambda_i\nabla h_i(\mathbf{x}) +\sum_{j=1}^p\mu_j\nabla g_j(\mathbf{x}) = \mathbf{0}$ ；

5）$\mu_j g_j(\mathbf{x}) = 0$ （$j=1,2,...,p$）。

## 2.3 拉格朗日对偶问题

一个优化问题可以从两个角度来考察，即“主问题”（primal problem）和“对偶问题”（dual problem）。对于主问题\eqref{2}，基于式\eqref{3}，其拉格朗日对偶函数（dual function）$\Gamma:\mathbb{R}^m\times\mathbb{R}^p\rightarrow\mathbb{R}$定义为：

$$
\begin{align}
\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu}) &= \inf_{\mathbf{x}\in \Omega}L(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\mu}) \\
		&= \inf_{\mathbf{x}\in \Omega}\left(f(\mathbf{x}) + \sum_{i=1}^m\lambda_i h_i(\mathbf{x}) +\sum_{j=1}^p\mu_j g_j(\mathbf{x})\right) \\
\end{align}
$$

其中$\Omega$为主问题的可行域。

假设主问题的最优解为$$\mathbf{x}^*$$，最优值为$$p^*$$，则对任意的$\boldsymbol{\lambda}$和$\boldsymbol{\mu}\succeq 0$都有

$$\sum_{i=1}^m\lambda_i h_i(\mathbf{x}^*) +\sum_{j=1}^p\mu_j g_j(\mathbf{x}^*) \leq 0$$

进而有

$$\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu}) = \inf_{\mathbf{x}\in \Omega}L(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\mu}) \leq L(\mathbf{x}^*,\boldsymbol{\lambda},\boldsymbol{\mu}) \leq f(\mathbf{x}^*) = p^*$$

即对偶函数给出了主问题最优值的下界。显然这个下界取决于$\boldsymbol{\lambda}$和$\boldsymbol{\mu}$的取值（常常通过将拉格朗日函数$L(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\mu})$对$\mathbf{x}$求偏导并置零来获得对偶函数的表达形式）。

于是，一个很自然的问题是：对偶函数能获得最大值是什么？这就引出了对偶问题：

$$\begin{equation}\label{4}\max_{\boldsymbol{\lambda},\boldsymbol{\mu}}\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu}) \text{ s.t. }\boldsymbol{\mu}\succeq 0\end{equation}$$

其中$\boldsymbol{\lambda}$和$\boldsymbol{\mu}$称为“对偶变量”（dual variable）。

考虑对偶问题\eqref{4} 的最优值$$d^*$$，显然有$$d^*\leq p^*$$，这称为弱对偶性（weak duality）成立；若$$d^* = p^*$$，则称为强对偶性（strong duality）成立，此时由对偶问题能获得主问题的最优下界。


# 3、凸优化理论

对于目标函数是凸函数、约束集是凸集的最小化问题称为凸优化问题或凸规划。线性规划、二次规划（目标函数为二次型函数、约束方程为线性方程）都可以归为凸规划。凸优化问题有很多特别的地方，常用到的有两点：

1）凸优化问题的局部极小点就是全局极小点；

2）极小点的一阶必要条件（包括无约束时$\nabla f(\mathbf{x})=0$和有约束时KKT条件）是凸优化问题的充分条件。

无论主问题的凸性如何，拉格朗日对偶问题始终是凸优化问题。

对于一般的优化问题，强对偶性一般不成立，但是，如果主问题是凸优化问题（即$f$和$g$是凸函数，$h$是仿射函数（affine function）），且至少存在一个绝对可行点，则对偶问题的强对偶性成立（这个条件就是传说中的Slater's condition）。即便主问题不是凸优化问题，但其对偶问题的解满足KKT条件，那么强对偶性也成立。

在强对偶性成立时，将拉格朗日函数分别对原变量和对偶变量求导，再分别置零，即可得到原变量与对偶变量的数值关系。于是，对偶问题解决了，主问题也就解决了。这是一般的凸优化问题求解思路。



