---
layout: post
title: "深度强化学习" 
---

# 1、值函数拟合（value function approximation）

当状态空间或动作空间非常大，甚至是连续的时，原来传统的强化学习方法不再有效。这个时候可以直接针对值函数进行学习。

## 1.1 Q函数拟合

因为无模型情况下，学习$Q$函数是更有效的方式，我们可以采用$Q_{\boldsymbol{w}}(s,a)$（其中$\boldsymbol{w}$为参数向量）来拟合Q函数。

我们希望通过$Q_{\boldsymbol{w}}$来拟合真实的值函数$Q^{\pi}$，若使用最小二乘误差为目标，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t + \frac{1}{2}\alpha\nabla_{\boldsymbol{w}} (Q^{\pi}(s_t,a_t)-Q_{\boldsymbol{w}_t}(s_t,a_t))^2 \\
			&= \boldsymbol{w}_t + \alpha (Q^{\pi}(s_t,a_t)-Q_{\boldsymbol{w}_t}(s_t,a_t))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}_t}(s_t,a_t)
\end{align}
$$

我们并不知道策略的真实值函数$Q_{\pi}(s_t,a_t)$，但可借助当前值函数的估计来代替，如时序差分学习中使用的$$r_t+\gamma Q_{\boldsymbol{w}_t}(s_{t+1},a_{t+1})$$，因此更新规则可以修改为：

$$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \alpha (r_t+\gamma Q_{\boldsymbol{w}_t}(s_{t+1},a_{t+1})-Q_{\boldsymbol{w}_t}(s_t,a_t))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}_t}(s_t,a_t)$$

这种在真标签中也使用了估值的梯度下降法叫“半梯度下降法”（semi-gradient descent）。

使用拟合函数来替代Q-learning算法中的值函数，即可得如下算法：

1）初始化：

$$\boldsymbol{w}=\mathbf{0}, \forall s \in S, \forall a\in A_s, \pi(s,a)=\text{arbitrary}$$；

设定初始状态$s$；

2）迭代：

基于$s$，根据当前策略$\pi$的$\epsilon$-贪心策略$\pi^{\epsilon}$，生成一个动作$a$；

基于$s$和$a$，观测奖励$r$和下一状态$s'$；

根据$\pi$生成下一贪心动作$a'$；

更新$\boldsymbol{w}$：

$$\boldsymbol{w} = \boldsymbol{w} + \alpha (r+\gamma Q_{\boldsymbol{w}}(s',a')-Q_{\boldsymbol{w}}(s,a))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}}(s,a)$$

更新策略$\pi(s) = arg\max_{a'' \in A_s}Q_{\boldsymbol{w}}(s,a'')$；

更新当前状态：$s=s'$。

因为在函数拟合中使用的训练样本并不满足独立同分布特性，因此该算法不一定能保证收敛。

## 1.2 V函数拟合

类似地，我们可以采用$V_{\boldsymbol{w}}(s)$（其中$\boldsymbol{w}$为参数向量）来拟合V函数，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t + \frac{1}{2}\alpha\nabla_{\boldsymbol{w}} (V^{\pi}(s_t)-V_{\boldsymbol{w}_t}(s_t))^2 \\
			&= \boldsymbol{w}_t + \alpha (V^{\pi}(s_t)-V_{\boldsymbol{w}_t}(s_t))\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}_t}(s_t)
\end{align}
$$

## 1.3 Memory Replay

Memory Replay机制是深度强化学习启蒙时期最重要的技巧之一，因为它的引入将深度强化学习推进了一大步。最早提出的深度强化学习是一个深度Q网络结构模型（Deep Q-Network，DQN）。因为训练样本相互关联（即每次得到的Q值估计是相互关联的），训练过程中策略会进行剧烈的振荡，从而使收敛速度十分缓慢。该问题严重影响了深度学习在强化学习中的应用。

Memory Replay的引入主要起如下作用：

1）打破可能陷入局部最优的可能；

2）模拟监督学习；

3）打破数据之间的关联性。

正是Memory Replay具有这样的作用才使得深度学习算法被顺利地应用在了强化学习领域。Memory Replay的具体操作步骤如下：

1）在算法执行前首先开辟一个Memory空间$D$；

2）在每个时间步$t$，将Agent与环境交互得到的转移样本$e_t=(s_t,a_t,r_t,s_{t+1})$存储到Memory空间$$D=\{e_1,e_2,...,e_t\}$$，当达到空间最大值后替换原来的采样样本；

3）从$D$中随机抽取一个批量（batch）的转移样本；

4）使用随机选取的样本，根据贝尔曼方程估算得到这些样本中存在的$(s,a)$对应的Q值；

5）通过该估值和当前网络的估计值的差量来更新网络模型的参数。

这种随机采样的方式，大大降低了样本之间的关联性，使得一个强化学习的问题变成了一个类似于监督学习的问题。

# 2 策略梯度法（policy gradient）

策略梯度法是另一种深度强化学习方法，它将策略看作一个基于策略参数的概率函数，通过不断计算策略期望总奖励关于策略参数的梯度来更新策略参数，最终收敛于最优策略。策略梯度法对于状态空间或者动作空间特别大甚至是连续的情况尤其有效。

策略可以表示为$\pi_{\boldsymbol{\theta}}(s,a)$，即在策略参数$\boldsymbol{\theta}$下，根据当前状态$s$选择动作$a$的概率。

我们用函数$J(\boldsymbol{\theta})$来估计状态值函数$V^{\pi_{\boldsymbol{\theta}}}(s_0)$，即表示始于状态$s_0$的策略$\pi_{\boldsymbol{\theta}}$的整体表现。学习过程可以表示为：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)$$

可见整个更新过程就是一个梯度上升法。

## 2.1 策略梯度定理

设在初始状态$s_0$下，服从策略$$\pi_{\boldsymbol{\theta}}$$的状态分布为$$\mu_{\pi_{\boldsymbol{\theta}}}(s)$$（即初始状态为$s_0$，策略$$\pi_{\boldsymbol{\theta}}$$作用下，状态$s$出现的次数比例期望，$$\sum_s \mu_{\pi_{\boldsymbol{\theta}}}(s) = 1$$）。策略梯度定理可以表示为如下：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \propto \sum_{s}\mu_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

因此，性能梯度可以用策略梯度来表示。

## 2.2 REINFORCE算法

根据策略梯度定理，可以进而推导如下：

$$\begin{align}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t) &\propto \sum_{s}\mu_{\pi_{\boldsymbol{\theta}_t}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}_t}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}_t}(s,a) \\
					&= E_{\pi_{\boldsymbol{\theta}_t}}\left(\sum_{a}\gamma^tQ^{\pi_{\boldsymbol{\theta}_t}}(S_t,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}_t}(S_t,a)\right) \\
					&=E_{\pi_{\boldsymbol{\theta}_t}}\left(\gamma^t\sum_{a}\pi_{\boldsymbol{\theta}_t}(S_t,a)Q^{\pi_{\boldsymbol{\theta}_t}}(S_t,a)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}_t}(S_t,a)}{\pi_{\boldsymbol{\theta}_t}(S_t,a)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}_t}}\left(\gamma^tQ^{\pi_{\boldsymbol{\theta}_t}}(S_t,A_t)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}_t}(S_t,A_t)}{\pi_{\boldsymbol{\theta}_t}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}_t}}\left(\gamma^tG_t\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}_t}(S_t,A_t)}{\pi_{\boldsymbol{\theta}_t}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}_t}}\left(\gamma^tG_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}_t}(S_t,A_t)\right) \\
\end{align}$$


所以策略梯度法的更新公式可以写为：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\gamma^t G_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}_t}(s_t,a_t)$$

利用上述这种更新方法来更新参数的策略梯度法叫做REINFORCE法，也是最基本的一种方法。但这种方法的方差很大，因为其更新的幅度依赖于某episode中$t$时刻到结束时刻的真实样本回报$G_t$。所以更常见的一种做法是引入一个基准（baseline）$b(s)$，于是有：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\gamma^t (G_t-b(S_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}_t}(s_t,a_t)$$

至于$b(s)$是什么，取决于算法的设计，但一般的做法是取$b(s) = V_{\boldsymbol{w}}(s)$。容易得到，这种情况下参数的更新主要取决于在状态$s_t$下执行动作$a_t$所得总奖励相对于状态均值的优势，如果有优势，则更新后的参数会增加执行该动作的概率；如果没有劣势，则更新后的参数会减少执行该动作的概率。

具体的带基准REINFOCE算法流程如下：

1）初始化参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$；

2）迭代：

根据策略$\pi_{\boldsymbol{\theta}}$生成一个采样片段$ < s_0,a_0,r_1,s_1,a_1,r_2,s_2,a_2,r_3,...> $，对于片段当中的每一步$t=0,1,2,...$，执行如下步骤：

$$G_t \leftarrow \text{从第}t\text{步计算的回报值}$$

$$\delta \leftarrow G_t - V_{\boldsymbol{w}}(s_t)$$

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}}(s_t)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}\gamma^t\delta\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$

## 2.3 行动者-评论家（Actor-Critic）算法

REINFORCE算法再更新时，需要每个episode的回报$G_t$，因此它是off-line的。而Actor-Critic算法是on-line的，它采用单步的奖励和下个状态估值的和式$$r_{t+1}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来代替REINFORCE算法中的全部回报$G_t$，并且采用一个学习到的状态值函数作为基准。

参数更新公式为：

$$\begin{align}
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t + \alpha\gamma^t (G^1_t-b(S_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}_t}(s_t,a_t) \\
		&= \boldsymbol{\theta}_t + \alpha\gamma^t (r_{t+1}+\gamma V_{\boldsymbol{w}}(s_{t+1})-V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}_t}(s_t,a_t) \\
\end{align}$$

因此，Actor-Critic算法的具体流程如下：

1）初始化参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$，当前状态$s=s_0$，梯度乘子$I=1$；

2）迭代：

基于当前状态$s$，根据策略$\pi_{\boldsymbol{\theta}}$生成一个动作$a$并得到奖励$r$和下一状态$s'$，并执行如下步骤：

$$\delta \leftarrow r+\gamma V_{\boldsymbol{w}}(s')-V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\delta\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s,a)$$

$$I \leftarrow \gamma I$$



# 3 蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）

蒙特卡洛树搜索是一种在强化学习问题中做出最优决策的方法，主要用于各种组合博弈中的行动策略规划，最终会生成一棵完全展开的博弈树。在博弈树中，每个节点表示的是一个状态$s$，同时还包含两个重要信息：一个是根据模拟结果估计的值$v_s$，另一个是该节点已经被访问的次数$n_s$；每一条边表示的是在某一状态下采取的动作$a$。

蒙特卡洛树搜索的主要概念是搜索，即沿着博弈树向下的一组遍历过程。单次遍历过程如下：

1）选择（Selection）：从博弈树的根结点（当前的博弈状态）按照一定的选择规则延伸到一个没有完全展开（即有未被访问的子节点）的节点L；

2）扩展（Expansion）：如果节点L不是终止节点（即不会导致博弈游戏终止），则选择一个未访问的子节点C作为模拟的起始点；

3）模拟（Simulation）：从节点C开始运行一个模拟，即从起始博弈状态开始，逐步采用一个rollout策略函数进行动作选择，直到博弈游戏终止得到输出结果，当然，也可以直接使用一个评估模型来评估节点C得到输出结果；

4）反向传播（Back Propagation）：将节点C标注为已访问，并将模拟的结果反向传播回当前博弈树的根结点，更新传播过程中每个节点的统计数据（$v_s$和$n_s$）。

在对博弈树向下遍历时，节点选择通过选择最大化某个量来实现。可以使用MCTS的一个特例——上线置信区间方法（Upper Confidence Bounds for Trees，UCT）来说明。

在UCT方法中，使用UCB（Upper Confidence Bound）函数，该函数给出了在当前节点$s$，其子节点$s_i$被选择的可能性：

$$UCB(s_i,s) = \frac{v_{s_i}}{n_{s_i}} + c\sqrt{\frac{\log(n_s)}{n_{s_i}}}$$

其中$c$是可调整参数。UCB函数具有利用（exploitation）和探索（exploration）的思想，其中第一个组件$\frac{v_{s_i}}{n_{s_i}}$是exploitation组件，可以理解为总模拟奖励（simulation reward）除以总访问次数，即节点$s_i$的胜率评估结果。第二个组件$\sqrt{\frac{\log(n_s)}{n_{s_i}}}$是exploration组件，它支持访问未被探索的节点，这些节点相对来说更少被访问（$n_{s_i}$较小）。而参数$c$则用于控制蒙特卡洛树搜索中explitation和exploration组件之间的权衡。

下面给出UCT方法流程：

1）从博弈树的根结点开始根据UCT函数向下搜索；

2）假设遇到节点s，如果节点s存在从未访问过的子节点则执行步骤3，否则执行步骤4；

3）扩展未访问的子节点并进行模拟评估，得到结果后更新该子节点至根节点路径上所有节点的估计值和访问次数，执行步骤1；

4）计算每个子节点的UCB值，将UCB值最高的子节点作为节点s，执行步骤2；

5）算法可随时终止，通常是达到给定时间或访问总次数。


# 4 AlphaGo算法框架

AlphaGo完整的学习系统主要由以下四个部分组成：

1）策略网络（policy network）。又分为监督学习的策略网络和强化学习的策略网络。策略网络的作用是根据当前的棋局来预测和采样下一步走棋。

2）滚轮策略（rollout policy）。也是用于预测和采样下一步走棋，但是预测速度是策略网络的1500倍。

3）估值网络（value network）。用于估计当前棋局的价值。

4）MCTS。将策略网络、滚轮策略和估值网络融合进策略搜索的过程中，形成一个完整的走棋系统。

具体实施分为四个阶段：

在第一阶段，使用策略网络$p_{\boldsymbol{\sigma}}$来直接对来自人类专家下棋的样本数据进行学习。$p_{\boldsymbol{\sigma}}$是一个13层的深度卷积网络，具体的训练方式为梯度下降法：

$$\Delta\boldsymbol{\sigma} \propto \frac{\partial \log p_{\boldsymbol{\sigma}}(a\mid s)}{\partial\boldsymbol{\sigma}}$$

在测试集上使用所有输入特征进行训练，预测人类专家走子动作的准确率为57.0%。同时，还使用局部特征训练了一个可以迅速走子采样的滚轮策略$p_{\boldsymbol{\pi}}$，预测人类专家走子动作的准确率为24.2%。$p_{\boldsymbol{\sigma}}$每下一步棋是3毫秒，而$p_{\boldsymbol{\pi}}$是2微秒。

在第二阶段，使用$p_{\boldsymbol{\sigma}}$作为输入，通过强化学习的策略梯度方法来训练策略网络$p_{\boldsymbol{\rho}}$：

$$\Delta\boldsymbol{\rho} \propto \frac{\partial \log p_{\boldsymbol{\rho}}(a\mid s)}{\partial\boldsymbol{\rho}}z$$

其中$z$表示一局棋最终所获的收益，胜为+1，负为-1，平0。

具体训练方式是：随机选择之前迭代轮的策略网络和当前的策略网络进行对弈，并利用策略梯度法（该方法属于REINFORCE基本算法）来更新参数，最终得到增强的策略网络$p_{\boldsymbol{\rho}}$。$p_{\boldsymbol{\rho}}$与$p_{\boldsymbol{\sigma}}$在结构上是完全相同的。增强后的$p_{\boldsymbol{\rho}}$与$p_{\boldsymbol{\sigma}}$对抗时胜率超过了80%。

在第三阶段，主要关注的是对当前棋局的价值评估。具体通过最小化估值网络输出$v_{\boldsymbol{\theta}}(s)$和收益$z$（通过增强后的$p_{\boldsymbol{\rho}}$自我对弈得到）之间的均方误差来训练估值网络（该方法属于蒙特卡洛方法）：

$$\Delta\boldsymbol{\theta} \propto \frac{\partial v_{\boldsymbol{\theta}}(s)}{\partial\boldsymbol{\theta}}(z-v_{\boldsymbol{\theta}}(s))$$

估值网络采用的结构与策略网络类似。

在第四阶段，主要基于策略网络、滚轮策略和估值网络进行蒙特卡洛树搜索。

每个博弈树的边$(s,a)$存储着三个重要量：状态-动作值$Q(s,a)$、访问计数$N(s,a)$和先验概率$P(s,a)$。

在每个时间步$t$，从状态$s_t$中选择一个走子动作$a_t$：

$$a_t = arg\max_a(Q(s_t,a)+u(s_t,a))$$

其中，$u(s_t,a)$表示额外的奖励，目的是鼓励探索前提下最大化走子动作的值：

$$u(s,a) \propto \frac{P(s,a)}{1+N(s,a)}$$

其中，$P(s,a) = p_{\boldsymbol{\rho}}(a\mid s)$，即用策略网络的输出作为先验概率。$u(s,a)$与先验概率成正比，与访问计数成反比。

当遍历$L$步到达一个叶节点$s_L$时，综合估值网络输出$v_{\boldsymbol{\theta}}(s_L)$和使用滚轮策略$p_{\boldsymbol{\pi}}$的模拟结果$z_L$来获得叶子节点的值：

$$V(s_l) = (1-\lambda)v_{\boldsymbol{\theta}}(s_L) + \lambda z_L$$

边$(s,a)$的访问计数和对应的值计算方式如下：

$$N(s,a) = \sum_{i=1}^N 1(s,a,i)$$

$$Q(s,a) = \frac{\sum_{i=1}^N 1(s,a,i)V(s_L^i)}{N(s,a)}$$

其中，$1(s,a,i)$与状态动作对$(s,a)$是否在第$i$次模拟中被访问有关，是为1，否为0；$s_L^i$表示第$i$次模拟时的叶子节点。

一旦完成了搜索过程，则Agent走子时从训练好的博弈树的根结点位置选择访问计数最多的走子动作。之前搜索过的数据可能仍然在当前考虑的博弈树范围内，这就可以重复使用数据而不是从头建新的博弈树，这可以减少MCTS的时间。
