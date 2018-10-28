---
layout: post
title: "状态估计" 
---
# 1、状态估计的概念

根据可获取的观测数据估算动态系统内部状态的方法称为状态估计（state estimation）。对系统的输入输出进行量测而得到的数据只能反映系统的外部特性，而系统的动态规律需要用内部状态变量（通常无法直接量测）来描述。因此状态估计对于了解和控制一个系统具有重要意义。

# 2、状态方程和观测方程

假设系统是离散的，其状态方程可以表示为：

$$\mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1},\mathbf{u}_k) + \mathbf{w}_k$$

其中$\mathbf{w}_k$是过程噪声（process noise），并假定其符合均值为0，协方差矩阵为$\mathbf{Q}_k$的多元状态分布，即

$$\mathbf{w}_k \sim N(0,\mathbf{Q}_k)$$

$$\mathbf{u}_k$$是控制向量，一般情况下我们认为它是关于$$\mathbf{x}_{k-1}$$的函数（基于观测$$\mathbf{z}_{k-1}$$来决定该输入什么样的$$\mathbf{u}_k$$）。

观测方程可以表示为：

$$\mathbf{z_k} = \mathbf{h}(\mathbf{x}_k) + \mathbf{v}_k$$

其中$\mathbf{v}_k$是观测噪声（observation noise），并假定其符合均值为0，协方差矩阵为$\mathbf{R}_k$的多元状态分布，即

$$\mathbf{v}_k \sim N(0,\mathbf{R}_k)$$

上述方程中，有两个基本假设：

1）马尔可夫假设，即当前状态只与上一状态有关，而与上一个状态之前的状态无关。公式表示为：

$$P(\mathbf{x}_{t} \mid \mathbf{x}_{1:t-1}) = P(\mathbf{x}_{t}  \mid \mathbf{x}_{t-1})$$

公式中的$P$称为状态转移概率。

2）观测假设，即当前观测值只依赖于当前状态，与之前时刻的状态无关。公式表示为：

$$P(\mathbf{z}_{t} \mid \mathbf{x}_{1:t}) = P(\mathbf{z}_{t}  \mid \mathbf{x}_t)$$

公式中的$P$称为观测概率。

因此，一个离散系统模型也可以由状态转移概率矩阵和观测概率矩阵唯一确定。

# 3、状态估计的分类

如果是根据现在及现在以前的所有观测数据，估计过去某个时刻的状态，即估算

$$P(\mathbf{x}_k \mid \mathbf{z}_{1:t}) (0 < k < t) $$

则称该过程为平滑（smoothing）。

如果是根据现在及现在以前的所有观测数据，估计当前时刻的状态，即估算

$$P(\mathbf{x}_t \mid \mathbf{z}_{1:t})$$

则称该过程为滤波（filtering）。

如果是根据现在及现在以前的所有观测数据，估计未来某个时刻的状态，即估算

$$P(\mathbf{x}_{t+k} \mid \mathbf{z}_{1:t}) (k > 0) $$

则称该过程为预测（prediction）。

# 4、滤波问题

如前所属，滤波是根据现在及现在以前的所有观测数据，估计当前时刻的状态。估算的输入是过去所有的观测数据，意味着计算每一时刻的状态概率都要回顾整个历史观测数据，那么随着时间的推移，更新代价会越来越大。所以需要找到一种递归的方法，根据时刻$t-1$的滤波结果，和时刻$t$时刻的观测数据，就可以计算出$t$时刻的滤波结果。

## 4.1 卡尔曼滤波

卡尔曼滤波主要应用于线性系统模型，可以将过程方程和观测方程重写为如下：

$$\mathbf{x}_k = \mathbf{F}_k\mathbf{x}_{k-1}+ \mathbf{B}_k\mathbf{u}_k + \mathbf{w}_k$$

$$\mathbf{z_k} = \mathbf{H}_k\mathbf{x}_k + \mathbf{v}_k$$

卡尔曼滤波使用前向递归的方式来对当前状态进行预估。

假设系统的状态方程、观测方程和控制向量都是已知的，对于当前时刻$t$，状态$\mathbf{x}_t$可以分别通过以下两步来进行推断：

1）预测（predict）。利用$t-1$时刻的估计值和过程方程来递推，得到一个初步预测结果$\widehat{\mathbf{x}}_{t\|t-1}$；

2）更新（update）。利用$t$时刻的观测结果$$\mathbf{z}_t$$来对上一步的预测结果进行校正，得到更新结果$\widehat{\mathbf{x}}_{t\|t}$。

需要注意的是，上述的$$\widehat{\mathbf{x}}_{t\mid t-1}$$、$$\widehat{\mathbf{x}}_{t\mid t}$$都是对$$\mathbf{x}_t$$的估计均值，$$\mathbf{z}_t$$是对$$\mathbf{H}_t\mathbf{x}_t$$的估计均值，和它们对应的还有一个协方差矩阵。$$\widehat{\mathbf{x}}_{t\mid t-1}$$、$$\widehat{\mathbf{x}}_{t\mid t}$$对应的协方差矩阵分别为$$\mathbf{P}_{t\mid t-1}$$、$$\mathbf{P}_{t\mid t}$$，$$\mathbf{z}_t$$则对应的协方差为$$\mathbf{R}_t$$，它们对应的分布均为多维正态分布。

预测过程可以表示为以下两个方程：

$$\widehat{\mathbf{x}}_{t\mid t-1} = \mathbf{F}_t\widehat{\mathbf{x}}_{t-1\mid t-1}+ \mathbf{B}_t\mathbf{u}_t$$

$$\mathbf{P}_{t\mid t-1} = \mathbf{F}_t\mathbf{P}_{t-1\mid t-1}\mathbf{F}_t^T + \mathbf{Q}_t$$

更新过程可以表示为如下三个方程：

$$\mathbf{K}_t = \mathbf{P}_{t\mid t-1}\mathbf{H}_t^T(\mathbf{H}_t\mathbf{P}_{t\mid t-1}\mathbf{H}_t^T + \mathbf{R}_t)^{-1}$$

$$\widehat{\mathbf{x}}_{t\mid t} = \widehat{\mathbf{x}}_{t\mid t-1}+ \mathbf{K}_t(\mathbf{z}_t-\mathbf{H}_t\widehat{\mathbf{x}}_{t\mid t-1})$$

$$\mathbf{P}_{t\mid t} = \mathbf{P}_{t\mid t-1} - \mathbf{K}_t\mathbf{H}_t\mathbf{P}_{t\mid t-1}$$

其中$$\mathbf{K}_t$$称为Kalman增益，它就像一个补偿，决定预测值应该变化多少幅度，才能变成更新值。



## 4.2 贝叶斯滤波


假设状态转移概率矩阵和观测概率矩阵是已知的，为了得到递归方法，贝叶斯滤波也可以写成两个步骤，预测（predict）和更新（update）：

1）预测（predict）。利用$t-1$时刻的估计值和状态转移矩阵来递推，得到一个初步预测结果（先验概率）：

$$\begin{align}
P(\mathbf{x}_{t} \mid \mathbf{z}_{1:t-1}) &= \sum_{\mathbf{x}_{t-1}}P(\mathbf{x}_{t}\mid \mathbf{x}_{t-1},\mathbf{z}_{1:t-1})P(\mathbf{x}_{t-1}\mid \mathbf{z}_{1:t-1}) \text{（全概率公式）} \\
					&= \sum_{\mathbf{x}_{t-1}}P(\mathbf{x}_{t}\mid \mathbf{x}_{t-1})P(\mathbf{x}_{t-1}\mid \mathbf{z}_{1:t-1}) \text{（马尔可夫假设）} \\
\end{align}$$

2）更新（update）。利用$t$时刻的观测结果$$\mathbf{z}_t$$和观测概率矩阵对上一步的预测结果（先验概率）进行校正，得到更新结果（后验概率）：

$$\begin{align}
P(\mathbf{x}_{t} \mid \mathbf{z}_{1:t}) & = P(\mathbf{x}_{t} \mid \mathbf{z}_{1:t-1},\mathbf{z}_{t}) \\
						& = \frac{P(\mathbf{z}_{t} \mid \mathbf{x}_{t},\mathbf{z}_{1:t-1})P(\mathbf{x}_{t}\mid \mathbf{z}_{1:t-1})}{P(\mathbf{z}_{t}\mid \mathbf{z}_{1:t-1})} \text{（贝叶斯定理）} \\
						& = \frac{P(\mathbf{z}_{t} \mid \mathbf{x}_{t})P(\mathbf{x}_{t}\mid \mathbf{z}_{1:t-1})}{P(\mathbf{z}_{t}\mid \mathbf{z}_{1:t-1})} \text{（观测假设）} \\
						& = \frac{P(\mathbf{z}_{t} \mid \mathbf{x}_{t})P(\mathbf{x}_{t}\mid \mathbf{z}_{1:t-1})}{\sum_{\mathbf{x}_t}P(\mathbf{z}_{t}\mid \mathbf{x}_t)P(\mathbf{x}_t\mid\mathbf{z}_{1:t-1})} \text{（全概率公式）} 
\end{align}$$


然而，在大多数情况下，模型是未知的，也即状态转移概率矩阵或观测概率矩阵未知，这就需要蒙特卡洛仿真来近似求解了。

## 4.3 粒子滤波

粒子滤波（particle filter）是贝叶斯滤波的一种非参数实现，其思想基于蒙特卡洛仿真，用粒子集来表示概率，可以用在任何形式的状态空间模型上。粒子滤波的前提条件是状态转移函数和观测函数已知，但是过程噪声和观测噪声的分布可以是未知的（均值为0）。

粒子滤波的基本过程包括：

1）初始化阶段

一般采用均匀分布初始化粒子，生成固定数量$M$个，每个粒子代表一个可能的状态。

2）状态转移阶段

根据状态转移函数和最新的控制向量$$\mathbf{u}_t$$，将粒子集$$\chi_{t-1}$$进行状态转移得到一个临时粒子集$$\bar{\chi}_{t}$$，如果知道状态转移概率，还可以根据状态转移概率来进行随机转移。此时临时粒子集$$\bar{\chi}_{t}$$中表示的是先验概率。

3）权重计算阶段

根据观察函数，可以计算临时粒子集$$\bar{\chi}_{t}$$每个粒子的观测值，与新的观测值$\mathbf{z}_t$进行相似度计算，可以为每个粒子计算得到一个权值。如果知道观察概率，则可以将粒子$$\mathbf{x}_t^{[m]} (1\leq m\leq M)$$的权值设置为$$w_t^{[m]} = P(\mathbf{z}_t\mid \mathbf{x}_t^{[m]})$$。在求得所有粒子的权重后，进行归一化处理。

4）重要性采样阶段

在临时粒子集$$\bar{\chi}_{t}$$中抽取替换$M$个粒子，抽取每个粒子的概率由其归一化后的权值给定。重采样将临时粒子集变成同样大小的粒子集$$\chi_t$$，该粒子集表示$t$时刻的后验概率。
