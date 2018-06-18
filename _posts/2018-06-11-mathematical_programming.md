---
layout: post
title: "数学规划" 
---
# 1、数学规划模型的一般形式

对于一个数学规划模型，可以表示为如下通用形式：

$$
\begin{align}
	\min f(\mathbf{x}) \\
	s.t.\ & h_i(\mathbf{x}) = 0\ (i = 1,2,...,m) \\
	& g_j(\mathbf{x}) \leq 0\ (j = 1,2,...,p) \\
	& \mathbf{x} \in \Omega 
\end{align}
$$

其中，$$\mathbf{x} = (x_1,x_2,...,x_n)^T$$ 称为决策变量；$f:\mathbb{R}^n\rightarrow \mathbb{R}$ 称为目标函数；$$h_i:\mathbb{R}^n\rightarrow \mathbb{R}$$ 和 $$g_j:\mathbb{R}^n\rightarrow \mathbb{R}$$ 称为约束函数，且$m\leq n$，它们构成的约束条件统称为函数约束；约束$\mathbf{x} \in \Omega$ 称为集合约束，比如，$\Omega$ 可以取值 $\Omega = \mathbb{R}^n$ 或 $\Omega = \mathbb{Z}^n$；$\min$ 表示对目标函数$f(\mathbf{x})$ 的优化结果是求其最小值；$s.t.$ 为subject to的缩写，是受约束于约束条件的意思。

对一个数学规划问题可以表述为：求满足所有约束条件的$$\mathbf{x}^*$$，使得$$f(\mathbf{x}^*)$$ 达到最优。其中$$\mathbf{x}^*$$称为该问题的最优解，$$f^* = f(\mathbf{x}^*)$$ 称为该问题的最优值。

决策变量、目标函数和约束条件是数学规划模型的三个要素，如果目标函数和约束函数均为线性的，则称该数学规划问题为线性规划，否则称为非线性规划；如果决策变量只能取整数则相应的数学规划问题称为整数规划；如果有多个目标函数，则称该数学规划问题为多目标规划。


# 2、线性规划及凸优化问题

## 2.1 线性规划模型的标准式和矩阵式

### 2.1.1 标准形式

线性规划模型的标准形式可以表示为如下：

$$
\begin{align}
	\max Z = \sum_{j=1}^{n}c_jx_j \\
	s.t.\ & \sum_{j=1}^{n}a_{ij}x_j = b_i \ (i = 1,2,...,m) \\
	& x_j \geq 0\ (j = 1,2,...,n)
\end{align}
$$

其中$x_j$是决策变量，$c_j$表示$$x_j$$的价值系数，$a_{ij}$表示约束方程$i$中$$x_j$$的系数，$b_i$表示约束方程$i$的右端项（right hand side）。

标准形式必须满足以下三个特征：

1）目标函数统一为求极大值（或极小值）；

2）所有约束条件（除变量的非负约束以外）必须都是等式，且右端项（$b_i$）必须全为非负值；

3）所有决策变量的取值必须全为非负值。


### 2.1.2 其他形式转换成标准形式

对于线性规划问题的其他形式均可转换为标准形式，具体转换有四种情况：

1）目标函数的转换

统一求极大值，若是求极小值，则可对目标函数乘以$-1$。

2）变量的转换

如果$$x_j\geq a$$，则定义新变量$$x_j' = x_j - a$$来代替$$x_j$$；
如果$$a \leq x_j \leq b$$，可令$$x_j' = x_j - a$$ 来代替$$x_j$$，而将$$x_j' \leq (b-a)$$当作一个约束条件来处理；
如果$$x_j\leq b$$，则定义新变量$$x_j' = b - x_j$$来代替$$x_j$$；
如果$$x_j$$取值无约束，可令两个新的非负变量$$x_j'$$、$$x_j''$$，然后用$$x_j = x_j'-x_j''$$来替换原问题中的$$x_j$$。

3）右端项的转换

当$$b_i < 0$$时，只需将等式或不等式两边同乘$-1$ 。

4）约束条件的转换

对于$\leq$型约束加入一个变量$$x_s$$，$$x_s \geq 0$$，这时$$x_s$$称为松弛变量；对于$\geq$型约束减去一个变量$$x_S$$，$$x_S \geq 0$$，这时$$x_S$$称为剩余变量。在实际问题中，$$x_s$$表示未被充分利用的资源，而$$x_S$$表示缺少的资源，在引入模型中后它们在目标函数中的系数均为0。


### 2.1.3 矩阵形式

标准形式可以用矩阵形式表示为如下：

$$
\begin{align}
	\max Z = \mathbf{c}\mathbf{x} \\
	s.t.\ & \mathbf{A}\mathbf{x} = \mathbf{b} \\
	& \mathbf{x} \geq \mathbf{0}
\end{align}
$$

其中$$\mathbf{c} = (c_1,c_2,...,c_n)$$为行向量，$$\mathbf{x} = (x_1,x_2,...,x_n)^T, \mathbf{b} = (b_1,b_2,...,b_m)\geq \mathbf{0}$$ 均为列向量，$\mathbf{A}$为一个$m \times n$矩阵，表示为如下：

$$\mathbf{A} = \begin{bmatrix} 	a_{11} & a_{12} & \dots & a_{1n}\\ 
				a_{21} & a_{22} & \dots & a_{2n} \\ 
				\vdots & \vdots & \ddots & \vdots \\ 
				a_{m1} & a_{m2} & \dots & a_{mn} 
		\end{bmatrix}
$$

## 2.2 单纯形法

### 2.2.1 可行解、最优解和基本解

对于线性规划的标准形式，若找到一个$$\mathbf{x} = (x_1,x_2,...,x_n)^T$$，其满足所有约束条件，且每个变量的值非负，则称$$\mathbf{x}$$为该问题的一个可行解。使目标函数值达到最大值（或最小值）的可行解即为该问题的最优解。

如果矩阵形式中$\mathbf{A}$的秩为$m$，即假设不存在冗余的约束条件，且方程数量$m$小于或等于变量数量$n$，则必定有一个或多个解，如果这些解满足非负条件，则是可行解。$m$列不相关的列向量对应的变量称为基变量（basic varibale），其余的$(n-m)$个变量称为非基变量（non-basic variable）。令非基变量为0，得到$m$个基变量的解称为基本解（basic solution），如果基本解某些基变量为零，那么这个基本解称为退化的基本解，如果基本解满足非负条件，则称为基本可行解（basic feasible solution），如果该基本可行解也是最优解，则称为最优基本可行解。

给定线性模型的标准形式，为了构造出初始基变量，约束条件还可能需要加上人工变量。人工变量最终必须等于0才能保持原问题的性质不变，因此需要在目标函数中令其系数为$-M$。$M$为一个无限大的正数，这是一个惩罚项，倘若人工变量不为零，则目标函数就永远达不到最优，所以必须将人工变量逐步从基变量中替换出去变成非基变量。如果最终人工变量没有被替换出去，那么这个问题就没有可行解，当然也没有最优解。

### 2.2.2 求解步骤

1）初始化

确定初始基变量和初始基本可行解，建立初始单纯形表。

2）迭代

a）最优性检验。若在当前表的目标函数对应的行中，所有非基变量的系数非负，则可判断得到最优解，可停止计算；否则转入下一步。

b）确定出基和入基变量。挑选目标函数对应行中系数最小（肯定为负）的非基变量作为进基变量。假设$$x_k$$为进基变量，按$\Theta$规则：

$$\theta = \min\left( \frac{b_i}{a_{ik}} | a_{ik} > 0 \right) = \frac{b_l}{a_{lk}}$$

确定$$x_l$$ 为出基变量，如果所有$$a_{ik}$$都非正，则此问题无界，可停止计算；否则转入下一步。

c）以$$a_{lk}$$为主元素对$$x_k$$所对应的列向量进行变换（采用高斯消元法），实现$$x_k$$入基，$$x_l$$出基。

d）重复以上步骤，直到停止计算。


### 2.2.3 求解过程示例

模型的初始形式：

$$
\begin{align}
	\max Z = 3x_1 + 5x_2 \\
	s.t.\ & x_1 \leq 4 \\
	& 2x_2 \leq 12 \\
	& 3x_1 + 2x_2 \leq 18 \\
	& x_1,x_2\geq 0 \\
\end{align}
$$

可以改写为如下形式：

$$
\begin{align}
	\max Z\\
	s.t.\ & Z - 3x_1 - 5x_2 = 0 \\ 
	& x_1 + x_3 = 4 \\
	& 2x_2 + x_4 = 12 \\
	& 3x_1 + 2x_2 + x_5 = 18 \\
	& x_j\geq 0\ (j=1,2,...,5) \\
\end{align}
$$

首先构造初始单纯形表，初始基为$$(Z,x_3,x_4,x_5)$$，初始基本可行解为$(0,4,12,18)$。

<table>
    <tr>
	<th rowspan="2">迭代序号</th>
	<th rowspan="2">基变量</th>
	<th rowspan="2">方程序号</th>
	<th colspan="6" align="center">系数</th>
         <th rowspan="2">右端项</th>
    </tr>
    <tr>
        <td>$Z$</td>
        <td>$x_1$</td>
        <td>$x_2$</td>
	<td>$x_3$</td>
	<td>$x_4$</td>
	<td>$x_5$</td>
    </tr>
    <tr>
	<td rowspan="4">0</td>
	<td>$Z$</td>
        <td>(0)</td>
	<td>1</td>
	<td>-3</td>
	<td>-5</td>
	<td>0</td>
	<td>0</td>
	<td>0</td>
	<td>0</td>
    </tr>
    <tr>
	<td>$x_3$</td>
        <td>(1)</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>0</td>
	<td>4</td>
    </tr>
    <tr>
	<td>$x_4$</td>
        <td>(2)</td>
	<td>0</td>
	<td>0</td>
	<td>2</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>12</td>
    </tr>
    <tr>
	<td>$x_5$</td>
        <td>(3)</td>
	<td>0</td>
	<td>3</td>
	<td>2</td>
	<td>0</td>
	<td>0</td>
	<td>1</td>
	<td>18</td>
    </tr>
</table>

然后进入迭代。确定$$x_2$$入基，$$x_4$$出基，基于$$x_4$$所在行进行高斯消元，得到下表：

<table>
    <tr>
	<th rowspan="2">迭代序号</th>
	<th rowspan="2">基变量</th>
	<th rowspan="2">方程序号</th>
	<th colspan="6" align="center">系数</th>
         <th rowspan="2">右端项</th>
    </tr>
    <tr>
        <td>$Z$</td>
        <td>$x_1$</td>
        <td>$x_2$</td>
	<td>$x_3$</td>
	<td>$x_4$</td>
	<td>$x_5$</td>
    </tr>
    <tr>
	<td rowspan="4">1</td>
	<td>$Z$</td>
        <td>(0)</td>
	<td>1</td>
	<td>-3</td>
	<td>0</td>
	<td>0</td>
	<td>$\frac{5}{2}$</td>
	<td>0</td>
	<td>30</td>
    </tr>
    <tr>
	<td>$x_3$</td>
        <td>(1)</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>0</td>
	<td>4</td>
    </tr>
    <tr>
	<td>$x_2$</td>
        <td>(2)</td>
	<td>0</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>$\frac{1}{2}$</td>
	<td>0</td>
	<td>6</td>
    </tr>
    <tr>
	<td>$x_5$</td>
        <td>(3)</td>
	<td>0</td>
	<td>3</td>
	<td>0</td>
	<td>0</td>
	<td>-1</td>
	<td>1</td>
	<td>6</td>
    </tr>
</table>

继续迭代。确定$$x_1$$入基，$$x_5$$出基，基于$$x_5$$所在行进行高斯消元，得到下表：

<table>
    <tr>
	<th rowspan="2">迭代序号</th>
	<th rowspan="2">基变量</th>
	<th rowspan="2">方程序号</th>
	<th colspan="6" align="center">系数</th>
         <th rowspan="2">右端项</th>
    </tr>
    <tr>
        <td>$Z$</td>
        <td>$x_1$</td>
        <td>$x_2$</td>
	<td>$x_3$</td>
	<td>$x_4$</td>
	<td>$x_5$</td>
    </tr>
    <tr>
	<td rowspan="4">2</td>
	<td>$Z$</td>
        <td>(0)</td>
	<td>1</td>
	<td>0</td>
	<td>0</td>
	<td>0</td>
	<td>$\frac{3}{2}$</td>
	<td>1</td>
	<td>36</td>
    </tr>
    <tr>
	<td>$x_3$</td>
        <td>(1)</td>
	<td>0</td>
	<td>0</td>
	<td>0</td>
	<td>1</td>
	<td>$\frac{1}{3}$</td>
	<td>$-\frac{1}{3}$</td>
	<td>2</td>
    </tr>
    <tr>
	<td>$x_2$</td>
        <td>(2)</td>
	<td>0</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>$\frac{1}{2}$</td>
	<td>0</td>
	<td>6</td>
    </tr>
    <tr>
	<td>$x_1$</td>
        <td>(3)</td>
	<td>0</td>
	<td>1</td>
	<td>0</td>
	<td>0</td>
	<td>$-\frac{1}{3}$</td>
	<td>$\frac{1}{3}$</td>
	<td>2</td>
    </tr>
</table>

这时候根据最优性检验，目标函数所在行中所有非基变量的系数非负，可判断得到最优解，停止计算。此时最优解为$$x_1 = 2, x_2 = 6$$，最优值为36。

### 2.2.4 单纯形法的突破

1）入基变量相持

在入基时，会挑选目标函数对应行中系数最小（肯定为负）的非基变量作为进基变量，但如果有两个非基变量的系数同时最小且为负，则形成了入基变量相持。此时，只需要任选一个作为入基变量即可。

2）出基变量相持

在出基时，如果多个基变量均满足出基条件，且$\theta$值相等，则形成出基变量相持。如果产生出基和入基的循环，需要通过改变出基变量跳出循环。

3）多个最优解

单纯形法在找到一个最优解时会自动停止，而线性规划有的额时候会有多个最优解，这些最优解可以通过所有的最优基本可行解的凸组合（convex combination）得到，即由它们的任意加权平均得到，权数项为非负且之和为1。一个拥有多个最优基本可行解的线性规划问题，在最终表中目标函数行的非基变量系数至少有一个为0，所以增加任意一个这样的变量都不会改变$Z$的值。因此，可以通过单纯形法进行进一步的迭代来获得其他最优基本可行解，进一步迭代时，每次选择1个系数为0的非基变量入基。

## 2.3 对偶问题

### 2.3.1 单纯形法的矩阵表示

考虑如下线性规划问题：

$$
\begin{align}
	\max Z = \mathbf{c}\mathbf{x} \\
	s.t.\ & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
	& \mathbf{x} \geq \mathbf{0}
\end{align}
$$

将其化为标准形式后为：

$$
\begin{align}
	\max Z = \mathbf{c}\mathbf{x} \\
	s.t.\ & [\mathbf{A}\ \mathbf{I}]\begin{bmatrix}\mathbf{x} \\ \mathbf{x}_s \end{bmatrix} = \mathbf{b} \\
	& \mathbf{x},\mathbf{x}_s \geq \mathbf{0}
\end{align}
$$

其中$$\mathbf{x}_s$$为加入后的松弛变量的列向量。

基于该标准形式进行单纯形法，假设在若干次迭代后，得到的基变量所构成的列向量为$$\mathbf{x}_B$$，称为基变量向量（vector of basic variables），$$[\mathbf{A}\ \mathbf{I}]$$中对应$$\mathbf{x}_B$$的列向量所构成的矩阵为$\mathbf{B}$，称为基矩阵（basic matrix）。令$$\mathbf{c}_B$$表示目标函数中对应$$\mathbf{x}_B$$的系数列向量。

于是单纯形法的矩阵形式可以表示如下表：

<table>
    <tr>
	<th rowspan="2">迭代序号</th>
	<th rowspan="2">基变量</th>
	<th rowspan="2">方程序号</th>
	<th colspan="3" align="center">系数</th>
         <th rowspan="2">右端项</th>
    </tr>
    <tr>
        <td>$Z$</td>
       	<td>原变量</td>
	<td>松弛变量</td>
    </tr>
    <tr>
	<td rowspan="2">0</td>
	<td>$Z$</td>
        <td>(0)</td>
	<td>1</td>
	<td>$-\mathbf{c}$</td>
	<td>$\mathbf{0}$</td>
	<td>0</td>
    </tr>
    <tr>
	<td>$\mathbf{x}_s$</td>
        <td>(1,2,...,m)</td>
	<td>$\mathbf{0}$</td>
	<td>$\mathbf{A}$</td>
	<td>$\mathbf{I}$</td>
	<td>$\mathbf{b}$</td>
    </tr>
    <tr>
	<td rowspan="2">$\vdots$</td>
	<td></td>
        <td></td>
	<td></td>
	<td></td>
	<td></td>
	<td></td>
    </tr>
    <tr>
	<td></td>
        <td></td>
	<td></td>
	<td></td>
	<td></td>
	<td></td>
    </tr>
    <tr>
	<td rowspan="2">若干次</td>
	<td>$Z$</td>
        <td>(0)</td>
	<td>1</td>
	<td>$\mathbf{c}_B\mathbf{B}^{-1}\mathbf{A}-\mathbf{c}$</td>
	<td>$\mathbf{c}_B\mathbf{B}^{-1}$</td>
	<td>$\mathbf{c}_B\mathbf{B}^{-1}\mathbf{b}$</td>
    </tr>
    <tr>
	<td>$\mathbf{x}_B$</td>
        <td>(1,2,...,m)</td>
	<td>$\mathbf{0}$</td>
	<td>$\mathbf{B}^{-1}\mathbf{A}$</td>
	<td>$\mathbf{B}^{-1}$</td>
	<td>$\mathbf{B}^{-1}\mathbf{b}$</td>
    </tr>
</table>

### 2.3.2 对偶问题的实质

单纯形法得到最优解的条件是目标函数所在行所有系数非负，因此在上述的单纯形表中如果$$\mathbf{c}_B\mathbf{B}^{-1}\mathbf{A}-\mathbf{c} \geq \mathbf{0}$$ 和 $$\mathbf{c}_B\mathbf{B}^{-1} \geq \mathbf{0}$$，就能得到的最优值$$\mathbf{c}_B\mathbf{B}^{-1}\mathbf{b}$$。

令$$\mathbf{y} = \mathbf{c}_B\mathbf{B}^{-1}$$，可以得到当满足条件$\mathbf{yA} \geq \mathbf{c}$和$\mathbf{y} \geq \mathbf{0}$时，就能达到最优值$$\mathbf{y}^*\mathbf{b}$$，而当$$\mathbf{yb} < \mathbf{y}^*\mathbf{b}$$时是不能同时满足条件$\mathbf{yA} \geq \mathbf{c}$和$\mathbf{y} \geq \mathbf{0}$的。

因此对于原问题：

$$
\begin{align}
	\max Z = \mathbf{cx} \\
	s.t.\ & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
	& \mathbf{x} \geq \mathbf{0}
\end{align}
$$

可以构造一个对偶问题：

$$
\begin{align}
	\min W = \mathbf{yb} \\
	s.t.\ & \mathbf{yA} \geq \mathbf{c} \\
	& \mathbf{y} \geq \mathbf{0}
\end{align}
$$

其中$\mathbf{y}$表示有$m$个变量构成的行向量。

### 2.3.3 对偶问题的经济解释

对于原问题，$$x_j$$可以理解为生产产品$j$的数量，$$c_j$$表示产品$j$的单位利润，$$b_i$$表示资源$i$的拥有量，$$a_{ij}$$表示资源$i$在生产产品$j$时的单位消耗率。该问题模型可以理解为如何利用现有资源的生产产品并得到最多总利润。

而对偶问题中，$$y_i$$可以理解为资源$i$的单位机会成本，即增加单位资源$i$可能引起其他地方的利润损失。$$y_i \geq 0$$约束了每种资源的单位机会成本必须大于或等于0。$$\sum_{i=1}^{m}a_{ij}y_i \geq c_j$$是$$y_i$$应该满足的约束条件，表示此时再生产任何产品$j$所需资源的机会成本都是大于或等于其带来的利润的，即已经不再适合进行任何生产了。目标函数$$\min W= \sum_{i=1}^{m}b_iy_i$$表示应该最小化所有投入资源的总机会成本，也即使投入这些资源进行生产所引起其他地方的利润损失最小。

在原问题取最优时，资源$i$的影子价格可以等于其单位机会成本（满足对偶问题的所有约束条件）。如果资源$i$未被充分利用（其对应的松弛变量大于0），其影子价格一定会下降为0（即$$y_i=0$$），这是对供求关系法则的遵循；那些被充分利用的资源才有影子价格，对于每种需要生产（$$x_j > 0$$）的产品$j$，其消耗资源的影子价格总和应该等于生产该产品带来的利润，即$$\sum_{i=1}^{m}a_{ij}y_i = c_j$$。因此，原问题取最优时对偶问题也取最优，原问题的最优值等于对偶问题的最优值。


# 3、整数规划

当数学规划模型里的某些决策变量规定只能取整数时，称该问题为整数规划。如果所有决策变量都是整数型，则称该问题为纯整数规划；如果一个问题中既有连续型的变量又有整数型的变量，则称为混合整数规划。显然，整数规划里又分整数线性规划和整数非线性规划。

本章主要考虑整数线性规划问题，即便是整数线性规划也没有一个有效的算法能确保在多项式时间内求解。

## 3.1 幺模矩阵（Unimodular Matrix）

有一类整数线性规划问题可以采用求解标准线性规划问题的方式进行求解。这类问题要用到幺模矩阵的概念。

**定义：** 对于$m\times n$的整数矩阵$\mathbf{A}\ (m\leq n)$，如果其所有$m$阶非零子式都为$\pm 1$，那么$\mathbf{A}$就是幺模矩阵。

**引理：** 对于线性方程组$\mathbf{Ax}=\mathbf{b}$，其中，$\mathbf{A}\in\mathbb{Z}^{m\times n}\ (m\leq n)$是幺模矩阵，$\mathbf{b}\in\mathbb{Z}^m$，它的所有基本解都是整数解。

**推论：** 如果线性规划的约束方程为

$$\mathbf{Ax}=\mathbf{b} \\ \mathbf{x}\geq \mathbf{0}$$

其中，$\mathbf{A}\in\mathbb{Z}^{m\times n}\ (m\leq n)$是幺模矩阵，$\mathbf{b}\in\mathbb{Z}^m$，那么其所有基本可行解都是整数的。

**定义：** 对于$m\times n$的整数矩阵$\mathbf{A}$，如果其所有非零子式都为$\pm 1$，那么$\mathbf{A}$是完全幺模的。

**定理：** 如果$m\times n$的整数矩阵$\mathbf{A}$是完全幺模的，那么矩阵$[\mathbf{A},\mathbf{I}]$是幺模矩阵。

**推论：** 如果线性规划的约束方程为

$$[\mathbf{A},\mathbf{I}]\mathbf{x}=\mathbf{b} \\ \mathbf{x}\geq \mathbf{0}$$

其中，$\mathbf{A}\in\mathbb{Z}^{m\times n}$是完全幺模的，$\mathbf{b}\in\mathbb{Z}^m$，那么其所有基本可行解都是整数的。


## 3.2 割平面法（Cutting Plane）

考虑整数规划问题：

$$
\begin{align}
	\max Z = \mathbf{c}\mathbf{x} \\
	s.t.\ & \mathbf{A}\mathbf{x} = \mathbf{b} \\
	& \mathbf{x} \geq \mathbf{0} \\
	& \mathbf{x} \in \mathbb{Z}^n
\end{align}
$$

采用单纯形法可以求得线性规划问题

$$
\begin{align}
	\max Z = \mathbf{c}\mathbf{x} \\
	s.t.\ & \mathbf{A}\mathbf{x} = \mathbf{b} \\
	& \mathbf{x} \geq \mathbf{0} \\
\end{align}
$$

的一个最优基本可行解。假设前$m$个列向量组成了最优基本可行解的基矩阵，则相应的标准单纯形矩阵为

$$
\begin{array}
&\mathbf{a}_1	&\mathbf{a}_2	&\cdots	&\mathbf{a}_m	&\mathbf{a}_{m+1}	&\cdots	&\mathbf{a}_n	&\mathbf{b}_0	\\
1		&0		&\cdots	&0		&a_{1,m+1}		&\cdots	&a_{1,n}	&b_{1,0}	\\	
0		&1		&\cdots	&0		&a_{2,m+1}		&\cdots	&a_{2,n}	&b_{2,0}	\\
\vdots		&\vdots		&	&\vdots		&\vdots			&	&\vdots		&\vdots		\\
0		&0		&\cdots	&1		&a_{m,m+1}		&\cdots	&a_{m,n}	&b_{m,0}	\\
\end{array}
$$

假设最优基本可行解中的第$i$个元素$b_{i,0}$不是整数。我们可以利用第$i$个等式

$$x_i + \sum_{j=m+1}^na_{i,j}x_j = b_{i,0}$$

构造出新增的约束条件，将当前的非整数最优解从可行集中切除，同时还保留所有整数可行解。

考虑不等式约束

$$x_i + \sum_{j=m+1}^n\lfloor a_{i,j}\rfloor x_j \leq b_{i,0}$$

因为$$\lfloor a_{i,j}\rfloor\leq a_{i,j}$$，所以对于任意可行解$\mathbf{x}\geq\mathbf{0}$，也满足这一不等式约束。此外，对于任意整数可行解$\mathbf{x}\geq\mathbf{0}$，不等式约束的左边都是一个整数，因此有

$$x_i + \sum_{j=m+1}^n\lfloor a_{i,j}\rfloor x_j \leq \lfloor b_{i,0}\rfloor$$

将前面的等式减去该不等式约束，可以得到

$$\sum_{j=m+1}^n(a_{i,j} - \lfloor a_{i,j}\rfloor)x_j \geq b_{i,0} - \lfloor b_{i,0}\rfloor$$

任意整数可行解$\mathbf{x}\geq\mathbf{0}$都满足该约束条件。但将最优基本可行解带入这一不等式后，可发现不等式左侧为0，而右侧是一个正数，因此最优基本可行解不满足该不等式约束。因此可以将该不等式约束加入原问题中，这个新约束称为割平面约束。

引入剩余变量$x_{n+1}$，将新的线性规划问题转化为标准形式，可得到等式约束：

$$\sum_{j=m+1}^n(a_{i,j} - \lfloor a_{i,j}\rfloor)x_j - x_{n+1} = b_{i,0} - \lfloor b_{i,0}\rfloor$$

该约束仍然可以称为割平面约束。将该约束加入矩阵$\mathbf{A}$和向量$\mathbf{b}$中，或直接加入其标准单纯形矩阵中，可利用单纯形法求得一个新的最优基本可行解，验证是否满足整数条件，如不满足继续重复上述过程，直到找到一个全是整数的最优基本可行解（注意，求解过程中引入的剩余变量并不要求必须为整数）。

## 3.3 分支定界法（Branch and Bound）

假设考虑的是最小化问题。分支表示可行空间的分解，切分后的可行空间可分别进行考虑；定界是指对于每个分支解空间下界的确定，如果某个分支的解下界大于另一分支中的整数解，则前一个分支就可以放弃，一般称作剪支。

假设要解决整数规划的线性松弛问题的最优解为$\mathbf{x}^0$，如果恰好都是整数，则对原来的整数规划也是最优解。否则，$\mathbf{x}^0$将被看作原来整数规划问题的下界。

如果$\mathbf{x}^0$中的一个变量不是整数，即$$x_j=r$$，则分支定界过程进行如下：通过增加两个互不相容的约束，将原整数规划问题分割为两个子问题。在第一个子问题（记为问题1）中，通过增加如下约束来修改：

$$x_j \leq \lfloor r\rfloor$$

另一个子问题（记为问题2）中，增加如下约束：

$$x_j \geq \lceil r\rceil$$

显然，原来整数规划的最优解必然存在于两个子问题可行域中的一个。

现在继续考虑子问题的线性松弛问题（比如问题1）并求得最优解。如果最优解为整数，则它同样也是该分支的最优解，所以不用再进行分支；如果最优解不是整数，则继续通过增加两个互不相容的约束，将问题1分支为问题1.1和问题1.2。

通过这种方法可创建出一棵分支树。从每个对应于非整数最优解的节点出发，分支得到两个子节点，当所有节点都有了一个整数最优解或一个大于其他节点整数解的分数解时停止。拥有最好整数解的节点即为原问题的最优解。



# 4、多目标规划 

绝大多数工程实际问题需要同时处理多个目标，而且这些目标之间往往存在冲突，即改进一个目标会导致另一个目标恶化。这种目标函数之间存在冲突的优化问题不存在唯一的最优解。该类问题称为多目标规划问题，可以采用标准形式表示如下：

$$
\begin{align}
	\min \mathbf{f}(\mathbf{x}) = \begin{bmatrix}f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_l(\mathbf{x}) \end{bmatrix} \\
	s.t.\ & \mathbf{x} \in \Omega \\
\end{align}
$$

其中，$$\mathbf{x} = (x_1,x_2,...,x_n)^T$$ 称为决策变量；$\mathbf{f}:\mathbb{R}^n\rightarrow \mathbb{R}^l$ 称为多目标函数；$\mathbf{x} \in \Omega$ 称为约束集。在该模型中没有区分函数约束和集合约束，函数约束可以包含在约束集$\Omega$中。

## 4.1 帕累托（Pareto）解

**定义：** 已知$\mathbf{f}:\mathbb{R}^n\rightarrow \mathbb{R}^l, \mathbf{x} \in \Omega$，考虑优化问题：

$$
\begin{align}
	\min \mathbf{f}(\mathbf{x}) = \begin{bmatrix}f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_l(\mathbf{x}) \end{bmatrix} \\
	s.t.\ & \mathbf{x} \in \Omega \\
\end{align}
$$

对于一个点$\mathbf{x}^* \in \Omega$，如果不存在$\mathbf{x} \in \Omega$，使得对于任意$i=1,2,..,l$，有

$$f_i(\mathbf{x}) \leq f_i(\mathbf{x}^*)$$

且至少对于一个$i$，有

$$f_i(\mathbf{x}) < f_i(\mathbf{x}^*)$$

则称$\mathbf{x}^*$是一个帕累托解。

帕累托解也称为非支配解，它意味着不存在另外一个可行的决策变量$\mathbf{x}$能够使得在某些目标函数减少的同时不会导致至少一个其他目标函数增加。绝大多数多目标优化算法都会用到“支配”这一概念。

帕累托解的集合称为帕累托前沿。

## 4.2 帕累托前沿的求解

首先引入一些符号和记法。记

$$\mathbf{x}^{*r} = [x_1^{*r},x_2^{*r},...,x_n^{*r}]^T$$

为第$r$个候选的帕累托解，$r=1,2,...,R$，$R$是当前候选的帕累托解的数量。

对于任何新生成的候选解$\mathbf{x}^j$，计算目标向量$\mathbf{f}(\mathbf{x}^j)$，然后将其与当前候选的帕累托解进行比较，可能会出现以下情况：

1）$\mathbf{x}^j$至少支配一个当前候选的帕累托解；

2）$\mathbf{x}^j$不支配任何当前候选的帕累托解；

3）$\mathbf{x}^j$受到一个当前候选的帕累托解的支配。

对于第一种情况，如果$\mathbf{x}^j$至少支配一个当前候选的帕累托解，那么就从候选集合删除被支配的解，并将新的解$\mathbf{x}^j$加入到候选集合中；在第二种情况下，如果$\mathbf{x}^j$不支配任何当前候选的帕累托解，将$\mathbf{x}^j$加入到候选集合中；对于第三种情况，$\mathbf{x}^j$受到一个当前候选的帕累托解的支配时，不改变当前候选集合。

基于以上思路，可以给出一个帕累托前沿生成算法：

第一步：生成初始候选解$\mathbf{x}^1$，计算$\mathbf{f}(\mathbf{x}^1)$。把$\mathbf{x}^1$加入到初始的候选集合中。初始化$R$和$j$，令$R=1,j=1$。

第二步：令$j=j+1$，如果$j\leq J$（$J$表示为得到最优解而必须进行分析的候选解数目，是算法的输入），则生成$\mathbf{x}^j$并转到第三步。否则，算法终止，将候选集合输出为帕累托前沿。

第三步：令$r=1,q=0$（$q$表示要从当前候选集合中移除的解的数量）。

第四步：如果$\mathbf{x}^j$支配$$\mathbf{x}^{*r}$$，则令$q=q+1$，并将$$\mathbf{x}^{*r}$$标记为应该被排除的解，转到第六步。

第五步：如果$\mathbf{x}^j$被$\mathbf{x}^{*r}$所支配，转到第二步。

第六步：令$r=r+1$，如果$r\leq R$，转到第四步。

第七步：如果$q\neq 0$，则从候选集合中移除第四步标记过的解，令$R=R-q$。

第八步：将解$\mathbf{x}^j$作为新的候选帕累托解加入到候选集合，令$R=R+1$，转到第二步。

以上帕累托前沿的生成算法比较简单，还可以用其他一些启发式算法，如遗传算法等。

## 4.3 多目标优化到单目标优化的转换

在某些情况下，可以将多目标优化问题转换为单目标优化问题。一般可能用到以下四种方法：

1）加权求和法。将目标函数向量中的各元素进行线性组合（组合系数必须为正数），以此作为单目标优化问题的目标函数。即

$$f(\mathbf{x}) = \mathbf{w}^T\mathbf{f}(\mathbf{x})$$

其中，向量$\mathbf{w}$的元素称为权值，全部为正数，反映的是目标向量中的各元素的相对重要度。不过，这些权值的确定可能非常困难。

2）极小极大法。以目标向量中的最大元素作为单目标函数。即

$$f(\mathbf{x})=\max\{f_1(\mathbf{x}),f_2(\mathbf{x}),...,f_l(\mathbf{x})\}$$


3）$p$范数法。如果目标向量的元素全部非负，那么可以使用目标函数向量的$p$范数作为但目标函数。即

$$f(\mathbf{x}) = \|\mathbf{f}(\mathbf{x})\|_p$$

当$p$为有限值时，为了使目标函数可微，可用$p$范数的$p$次方来代替范数本身。

4）约束法。将目标向量中的一个元素作为单目标函数，其他元素全部作为约束条件。即，可构造但目标优化问题为：

$$
\begin{align}
	\min f_1(\mathbf{x}) \\
	s.t.\ & f_1(\mathbf{x}) \leq b_2 \\
	& \vdots \\
	& f_l(\mathbf{x}) \leq b_l \\
\end{align}
$$

其中，$$b_2,...,b_l$$是常量，分别表示对目标$$f_2,...,f_l$$的期望值。显然，只有确定了这些期望值后才能使用该方法。
