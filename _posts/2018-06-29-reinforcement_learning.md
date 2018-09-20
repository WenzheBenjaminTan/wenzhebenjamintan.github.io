---
layout: post
title: "强化学习" 
---

# 1、有模型学习

## 1.1 马尔可夫决策过程

首先还是回到动态规划里面提到的马尔可夫决策过程：

$$(S,A,P.(.,.),R.(.,.),\gamma)$$

其中，

$S$是一组有限状态集；

$A$是一组有限的动作集（或者用$A_s$在给定状态$s$下可用的动作集）；

$P_a(s,s') = P(s_{t+1}=s' \mid s_t=s,a_t=a)$表示状态$s$下采用动作$a$后，得到下一状态为$s'$的概率，该函数是与时间$t$（时齐的情况）以及时间$t$之前的历史状态无关的。

$R_a(s,s')$是实现前述状态转移后得到的奖励，在有的的模型中，奖励函数只与状态转移有关，而与动作无关，即可以表示为$R(s,s')$；

$\gamma \in (0,1]$是奖励折扣系数，代表未来奖励的重要程度。


考虑$n$为$+\infty$的情况，这时候没法用传统动态规划递归求解了。一般来说，强化学习只关心**平稳确定型策略**$\pi$，即这个策略与当前所在时刻$t$无关，而只与当前的状态$s_t$有关，且通过当前状态能确定性地得到动作$a_t=\pi(s_t)$。对于有限的状态集$S$和行为集$A$，共可能有$$\mid A \mid^{\mid S \mid}$$种平稳确定型策略。

我们定义在阶段$t$，某一状态的回报（return）为：

$$\begin{align}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + \cdots \\
&= R_{t+1} + \gamma G_{t+1}
\end{align}
$$

定义在平稳策略$\pi$下，状态$s$的状态值函数为：

$$V^{\pi}(s) = E_{\pi}(G_t\mid s_t=s) = E_{\pi}(\sum_{k=0}^{+\infty}\gamma^kR_{t+k+1}\mid s_t=s)$$

于是对于平稳策略$\pi$，其总的回报期望为$\sum_{s\in S}V^{\pi}(s)$，因此最优平稳策略（可以证明，当$n$为$+\infty$时，最优平稳策略也是最优策略，而当$n$为有限值时，最优平稳策略不一定是最优策略）定义为：

$$\pi^* = arg\max_{\pi} \sum_{s\in S}V^{\pi}(s)$$

最优平稳策略所对应的状态值函数，称为最优状态值函数：

$$V^*(s) = V^{\pi^*}(s) (\forall s\in S)$$

可以证明，最优状态值函数满足：

$$V^*(s) = V^{\pi^*}(s) = \max_{\pi}V^{\pi}(s) (\forall s\in S)$$


为了进一步考虑确定动作为$a$后的回报期望，可以再定义一个状态-动作值函数：

$$Q^{\pi}(s,a) = E_{\pi}(G_t\mid s_t=s,a_t=a) = E_{\pi}(\sum_{k=0}^{+\infty}\gamma^kR_{t+k+1}\mid s_t=s, a_t =a )$$

$Q^{\pi}(s,a)$表示当处于状态$s$时执行动作$a$后遵循平稳策略$\pi$的价值。状态-动作值函数与状态值函数的关系可以表示为：

$$V^{\pi}(s) = Q^{\pi}(s,\pi(s))$$

$$Q^{\pi}(s,a) = \sum_{s' \in S}P_{a}(s,s')(R_{a}(s,s') + \gamma V^{\pi}(s'))$$


类似地，定义$$Q^*(s,a)$$为处于状态$s$时执行动作$a$后遵循最优平稳策略$$\pi^*$$的期望累计奖励，即最优状态-动作值函数：

$$Q^*(s,a) = Q^{\pi^*}(s,a) (\forall s\in S,a\in A)$$

同样可以证明，最优状态-动作值函数满足：

$$Q^*(s,a) = Q^{\pi^*}(s,a) = \max_{\pi}Q^{\pi}(s,a) (\forall s\in S,a\in A)$$

## 1.2 贝尔曼方程（Bellman's equation）

将状态值函数和状态-动作值函数分别写成递归的形式，就得到了贝尔曼方程：

$$V^{\pi}(s) = \sum_{s' \in S}P_{\pi(s)}(s,s')(R_{\pi(s)}(s,s') + \gamma V^{\pi}(s'))$$

$$Q^{\pi}(s,a) = \sum_{s' \in S}P_{a}(s,s')(R_{a}(s,s') + \gamma Q^{\pi}(s',\pi(s')))$$


根据动态规划的递归方程，可以得到：

$$\begin{align}
V^*(s) &= \max_{a \in A_s} \left\{ \sum_{s' \in S}P_a(s,s')(R_a(s,s')+\gamma V^*(s')) \right\} \\
\end{align}
$$

从上式可以得到$$V^*(s) = \max_{a \in A_s}Q^*(s,a)$$，进而可将上式转换成$Q^*$的迭代：

$$\begin{align}
Q^*(s,a) =  \sum_{s' \in S}P_a(s,s')(R_a(s,s')+\gamma\max_{a' \in A_{s'}} Q^*(s',a'))
\end{align}
$$

以上两个方程称为最优贝尔曼方程。

最优贝尔曼方程揭示了非最优策略的改进方式：将策略选择的动作改变为当前最优的动作。设当前策略为$\pi$，我们选择下一策略为$\pi'(s) = arg \max_{a\in A_s}Q^{\pi}(s,a) (\forall s\in S)$。容易得出：

$$\begin{align}
V^{\pi}(s) &\leq Q^{\pi}(s,\pi'(s)) \\
	&= \sum_{s' \in S}P_{\pi'(s)}(s,s')(R_{\pi'(s)}(s,s') + \gamma V^{\pi}(s')) \\
	&\leq \sum_{s' \in S}P_{\pi'(s)}(s,s')(R_{\pi'(s)}(s,s') + \gamma Q^{\pi}(s',\pi'(s'))) \\
\label{1}	&\leq V^{\pi'}(s)
\end{align}
$$

不断重复上述迭代，直到$\pi'$与$\pi$一致，不再发生变化，此时就满足最优贝尔曼方程，即找到了最优策略。

另一方面，如果我们获得了$Q^*(s,a) (\forall s\in S,a\in A)$的值，那么也可以得到最优策略：

$$\pi^*(s) = arg \max_{a\in A_s}Q^*(s,a) (\forall s\in S)$$

## 1.3 策略迭代

由贝尔曼方程，在策略$\pi$作用下，状态值函数可以递归地表示为：

$$V^{\pi}(s) = \sum_{s' \in S}P_{\pi(s)}(s,s')(R_{\pi(s)}(s,s') + \gamma V^{\pi}(s'))$$

给定MDP五元组模型$$(S,A,P.(.,.),R.(.,.),\gamma)$$，策略迭代流程如下：

1）初始化：$$\forall s \in S, V(s)=0, \pi(s)=\text{arbitrary}$$；

2）迭代： 

策略评估：$$\forall s \in S, V(s) = \sum_{s' \in S}P_{\pi(s)}(s,s')(R_{\pi(s)}(s,s') + \gamma V(s'))$$（也可以先单独对该步进行迭代，直到得到满意的对$\pi$的评估）；

策略改进：$$\forall s \in S, \pi'(s) = arg \max_{a\in A_s}\sum_{s' \in S}P_a(s,s')(R_a(s,s') + \gamma V(s'))$$；

3）终止条件判断：如果$$\forall s \in S, \pi'(s) = \pi(s)$$，则终止迭代并输出$\pi$作为最优策略；否则令$\pi = \pi'$，继续迭代。

策略迭代不断收敛，一直到不可能继续改进策略为止，因此可以确保所得策略是最优的。


## 1.4 状态值迭代

策略迭代在每次迭代改进策略后都需要重新进行策略评估，这通常比较耗时。由\eqref{1}可知，策略改进与状态值函数的改进是一致的，因此，可将策略改进视为状态值函数的改善。

由最优贝尔曼方程，可知

$$ V^*(s) = \max_{a \in A_s} \left\{ \sum_{s' \in S}P_a(s,s')(R_a(s,s') + \gamma V^*(s')) \right\} $$

因此，给定MDP五元组模型$$(S,A,P.(.,.),R.(.,.),\gamma)$$和收敛阈值$\theta$，可以设计如下状态值迭代流程：

1）初始化：$$\forall s \in S, V(s)=0$$；

2）迭代：

值迭代：$$\forall s \in S, V'(s) = \max_{a \in A_s} \left\{ \sum_{s' \in S}P_a(s,s')(R_a(s,s') + \gamma V(s')) \right\} $$；

3）终止条件判断：如果$$\max_{s\in S}\mid V'(s)-V(s)\mid \geq \theta$$，则令$V=V'$，继续迭代；否则，根据$V'$输出最优策略$\pi'$如下：

$$\forall s \in S, \pi'(s) = arg \max_{a\in A_s}\sum_{s' \in S}P_a(s,s')(R_a(s,s') + \gamma V'(s'))$$

已经证明该迭代方法收敛于正确的$V^*$值，从而输出的策略也收敛于最优策略。

## 1.5 策略迭代与状态值迭代的比较

对于每一次迭代的复杂度来说，策略迭代比状态值迭代要高，但是策略迭代的收敛速度更快。因此当模型的状态空间较小时，最好选用策略迭代方法；而当状态空间较大时，状态值迭代的总计算量要更小一些。



# 3、无模型学习

在现实的强化学习任务中，状态转移概率$P$和奖励函数$R$往往都是很难得知的，甚至一共有多少状态都不一定知道。如果学习算法不依赖于环境模型，依靠经验（这里的经验是指state、action、reward的采样序列）就可以求解最优策略，我们称这种强化学习算法为无模型学习算法。

在无模型的情况下，策略迭代算法首先遇到的问题就是策略无法评估，因为状态转移概率未知。另一方面，策略迭代算法估计的是状态值函数$V$，而最终策略是通过状态-动作值函数$Q$来获得的。当模型已知时，从$V$到$Q$转换比较简单，但是当模型未知时，这也会变得非常困难。于是，我们将估计对象从$V$转变为$Q$，即估计每一对“状态-动作”的值。

## 3.1 蒙特卡洛强化学习

为了评估状态-动作值函数，我们通过在给定状态下执行选择的动作，然后根据某种策略进行采样，观察状态的转移和得到的奖赏。通过多次“采样”，然后求取平均累积奖赏作为期望累积奖赏的近似，这称为蒙特卡洛强化学习。

具体来说，我们在模型未知的情况下，从起始状态出发，使用某种策略进行采样，获得一个episode的轨迹：

$$ < s_0,a_0,r_1,s_1,a_1,r_2,s_2,a_2,r_3,...> $$

然后，对轨迹中出现的每一对“状态-动作”，记录其后获得的奖励之和（分为first-visit和every-visit两种记录方法），作为该“状态-动作”对的一次累积奖励采样值（记为一次return）。多次采样得到多条轨迹后，将每个“状态-动作”对的returns进行平均，就得到该策略下状态-动作值函数的估计。

所有的无模型学习方法都面临一个困境：它们的目标是学习一系列优化动作的奖励值，然而为了寻找优化的动作，它们不能总选择最优化的动作，而需要探索新的动作，因此，这些学习过程都会有一个利用（exploitation）与探索（exploration）之间的权衡。具体到蒙特卡洛方法来说，有可能在采样过程中很多“状态-动作”对从未出现过，这样就没有returns来进行平均，因此也无法对这些“状态-动作”对进行评估。

一种解决方法是exploring starts，顾名思义，该方法通过episode的起点来做exploration，也就是每一对state-action都会以非零的概率被选中作为episode的起点。exploring starts方法很多时候可能会受问题或环境的约束无法实现。另外一种方法是将确定型的策略改成非确定型策略，使得该策略对每个状态来说，所有动作被选中的概率都是非零的。

### 3.1.1 起点探索（exploring starts）方法

按照策略迭代方法，我们先对当前策略进行评估，完成评估后再对当前策略进行改进。此时对所有$s\in S$，有：

$$
\begin{align}
Q^{\pi_k}(s,\pi_{k+1}(s)) &= Q^{\pi_k}(s,arg\max_{a\in A_s}Q^{\pi_k}(s,a)) \\
			&= \max_{a\in A_s}Q^{\pi_k}(s,a) \\
			&\geq Q^{\pi_k}(s,\pi_{k}(s)) \\
			&\geq V^{\pi_k}(s)
\end{align}
$$

以上迭代过程可以保证$\pi_k$和$Q^{\pi_k}$分别收敛于最优策略和最优状态-动作值函数。

如果按照这种流程，为了完成对当前策略的评估，即得到接近真实的$Q^{\pi}$函数，理论上我们需要观测无穷多个episodes。

为了避免无限次的episodes，可以放弃在返回到policy improvement之前才完成policy evaluation过程，而是在每个policy evaluation步骤中，都进行policy improvement让$Q$向$Q^{\pi_k}$ 逼近。

以下是exploring starts方法的具体执行流程：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi(s)=arg\max_{a\in A_s}Q(s,a), Returns(s,a)=\text{empty list}$$；

2）迭代：

随机得到$s_0$和$a_0$；

基于$s_0$和$a_0$，根据当前策略$\pi$，生成一个episode；

对于每个出现在该episode中的$(s,a)$，计算其累积奖励并添加到$Returns(s,a)$中，并更新状态-动作值$Q(s,a)=average(Returns(s,a))$；

对于每个出现在episode中的状态$s$，更新策略$\pi(s) = arg\max_{a\in A_s}Q(s,a)$。


### 3.1.2 同策略（on-policy）方法

为了避免exploring starts的假设，我们考虑在初始状态不变的同时，使得所有state-action对都可能被访问到，即将确定型的策略改成非确定型策略，使得该策略对每个状态来说，所有动作被选中的概率都是非零的（我们称这种策略是soft的）。使用非确定型策略又分为两种方法：on-policy方法和off-policy方法。on-policy方法评估和改进的是同一个策略，而off-policy方法评估和改进的是不同的策略。本节介绍on-policy方法，下节介绍off-policy方法。

在on-policy方法中，常常采用$\epsilon$-greedy policy。原始策略为$\pi(s) = arg\max_{a\in A_s}Q(s,a)$，而$\epsilon$-greedy policy $\pi^{\epsilon}$则表示在$1-\epsilon$的概率下会执行原始策略$\pi$，剩下$\epsilon$概率下会以均匀概率选择$A_s$中的一个动作。因此，总的算下来，当前最优动作被选中的概率是$1-\epsilon + \frac{\epsilon}{\mid A_s\mid}$，每个非最优动作被选中的概率是$\frac{\epsilon}{\mid A_s\mid}$。

同策略方法的具体流程如下：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi^{\epsilon}(s,a)=\epsilon\text{-greedy policy}, Returns(s,a)=\text{empty list}$$；

设定初始状态$s_0$；

2）迭代：

基于$s_0$，根据当前策略$\pi^{\epsilon}$，生成一个episode；

对于每个出现在该episode中的$(s,a)$，计算其累积奖励并添加到$Returns(s,a)$中，并更新状态-动作值$Q(s,a)=average(Returns(s,a))$；

对于每个出现在episode中的状态$s$，求得最优动作为$a^* = arg\max_{a\in A_s}Q(s,a)$，且对所有$a\in A_s$，更新策略：

$$\pi^{\epsilon}(s,a) = \begin{cases}1-\epsilon + \frac{\epsilon}{\mid A_s\mid} & a = a^* \\
\frac{\epsilon}{\mid A_s\mid} & a \neq a^* \\
\end{cases}$$

可以证明若采用$\epsilon$-soft 策略（即保证每个动作都可能被选择），经过足够多的迭代，其可收敛到$\epsilon$-soft空间下的最优策略（次优）。

### 3.1.3 异策略（off-policy）方法

同策略方法在迭代过程中优化的是策略$\pi^{\epsilon}$，而引入$\pi^{\epsilon}$是为了探索，不是为了最终使用；实际上我们希望改进的是原始策略。

在异策略方法中，会使用两个不同的策略$\pi$和$\pi'$，策略$\pi$用于学习成为最优策略，称为target policy，而策略$\pi'$则用于探索，称为behavior policy。我们希望评估的是$Q^{\pi}$，但所有episodes都是服从策略$\pi'$生成的。为了利用从策略$\pi'$产生的episodes来评估策略$\pi$的value，我们需要策略$\pi$下的所有state-action对都可能被策略$\pi'$生成，也即要求对所有满足$\pi(s,a) > 0$的$(s,a)$均有$\pi'(s,a) > 0$，这个条件可以称为是覆盖（coverage）。

异策略方法要用到重要性采样（importance sampling），这是给定服从一种分布的样本情况下估计另一种分布下期望值的一般方法。一般地，函数$f$在概率分布$p$下的期望可表达为$E(f) = \int_xp(x)f(x)dx$。可通过从概率分布$p$上的采样$\\{x_1,x_2,...,x_m\\}$来估计$f$的期望，即$\hat{E}(f) = \frac{1}{m}\sum_{i=1}^{m}f(x_i)$。如果有另一个分布$q$，函数$f$在概率分布$p$下的期望可等价表达为$E(f) = \int_xq(x)\frac{p(x)}{q(x)}f(x)dx$，即可看作函数$\frac{p(x)}{q(x)}f(x)$在分布$q$下的期望。因此，也可以通过从概率分布$q$上的采样$$\{x'_1,x'_2,...,x'_m\}$$来估计$f$的期望，即$\hat{E}(f) = \frac{1}{m}\sum_{i=1}^{m}\frac{p(x'_i)}{q(x'_i)}f(x'_i)$。

回到我们的问题上来，如果使用策略$\pi$上的采样轨迹来评估策略$\pi$，实际上就是对累积奖励求期望：$Q(s,a) = \frac{1}{m}\sum_{i=1}^mG_i$，其中$G_i$表示第$i$条轨迹上从状态动作对$(s,a)$开始到结束的累积奖励。如果改用策略$\pi'$的采样轨迹来评估策略$\pi$，则仅需对累积奖励进行加权，即

$$Q(s,a) = \frac{1}{m}\sum_{i=1}^m\frac{P_i^{\pi}}{P_i^{\pi'}}G_i$$

其中$P_i^{\pi}$和$P_i^{\pi'}$分别表示策略$\pi$和$\pi'$产生第$i$条轨迹上从状态动作对$(s,a)$开始到结束部分的概率。

对于一条给定episode轨迹：

$$ < s_t,a_t,r_{t+1},s_{t+1},a_{t+1},r_{t+2},...,s_T> $$

其在策略$\pi$下发生的概率为：

$$P^{\pi} = \prod_{i=t}^{T-1}\pi(s_i,a_i)P_{a_i}(s_i,s_{i+1})$$

因此，该轨迹在target policy和behavior policy下发生的概率比值为（称为重要性采样比值）：

$$\frac{P^{\pi}}{P^{\pi'}} = \frac{\prod_{i=t}^{T-1}\pi(s_i,a_i)P_{a_i}(s_i,s_{i+1})}{\prod_{i=t}^{T-1}\pi'(s_i,a_i)P_{a_i}(s_i,s_{i+1})} = \prod_{i=t}^{T-1}\frac{\pi(s_i,a_i)}{\pi'(s_i,a_i)}$$

可以看出，该重要性采样比值仅依赖于两个策略，而与模型本身无关。

还是按照on-policy方法的例子，假设$\pi$是原始策略，$\pi'$是$\pi$的$\epsilon$-贪心策略$\pi^{\epsilon}$ ，则$\pi(s_i,a_i)$对于$a_i=\pi(s_i)$为1，其余为0，而$\pi^{\epsilon}(s_i,a_i)$对于$a_i=\pi(s_i)$为$1-\epsilon+\frac{\epsilon}{\mid A_{s_i}\mid}$，其余$a_i\in A_{s_i}$为$\frac{\epsilon}{\mid A_{s_i}\mid}$，于是就能利用$\pi^{\epsilon}$产生的轨迹来评估和优化$\pi$了。


异策略方法的具体流程如下：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi(s)=arg\max_{a\in A_s}Q(s,a), Returns(s,a)=\text{empty list}$$；

设定初始状态$s_0$；

2）迭代：

基于$s_0$，根据当前策略$\pi$的$\epsilon$-贪心策略$\pi^{\epsilon}$，生成一个episode；

对于每个出现在该episode中的$(s,a)$，计算其累积奖励，并根据$(s,a)$在轨迹中出现的时间$t$计算权重$\prod_{i=t}^{T-1}\frac{\pi(s_i,a_i)}{\pi^{\epsilon}(s_i,a_i)}$，将权重以及加权后的奖励添加到$Returns(s,a)$中，更新状态-动作值$Q(s,a)=weightedAverage(Returns(s,a))$；

对于每个出现在episode中的状态$s$，更新策略$\pi(s) = arg\max_{a\in A_s}Q(s,a)$。

当每对state-action都能获得无数次的returns，理论上来说$\pi$可以收敛到最优策略。


## 3.2 时序差分（temporal difference）学习

蒙特卡洛强化学习算法基于采样轨迹进行策略优化，克服了模型未知给策略估计造成的困难。但此类算法都是在完成一个episode的采样轨迹后再更新策略的值估计（称为off-line算法），而前面介绍的基于动态规划的策略迭代和值迭代可以在每执行一步策略后就进行值函数更新（称为on-line算法）。两者相比，蒙特卡洛方法的效率要低得多（因为求一个episode的采样轨迹时间本身较长），这里的主要问题是蒙特卡洛方法没有充分利用强化学习任务的MDP结构。而时序差分学习则结合了动态规划与蒙特卡洛方法的思想，能够做到更高效的无模型学习。

根据贝尔曼方程，我们首先将策略$\pi$作用下的$Q^{\pi}$函数写成如下递归形式（贝尔曼方程）：

$$Q^{\pi}(s,a) = \sum_{s' \in S}P_{a}(s,s')(R_{a}(s,s') + \gamma Q^{\pi}(s',\pi(s')))$$

根据最优贝尔曼方程，当取得最优策略时，$Q^*$函数具有如下形式：

$$\begin{align}
Q^*(s,a) =  \sum_{s' \in S}P_a(s,s')(R_a(s,s')+\gamma\max_{a' \in A_{s'}} Q^*(s',a'))
\end{align}
$$


假设在时间$t$时，状态为$s$，执行的动作是$a$，而在$t+1$时刻得到了状态$s'$，我们可以用指数平滑的方式来更新值函数估计$Q^{\pi}(s,a)$：

$$\begin{equation}\label{td}Q^{\pi}_{t+1}(s,a) = Q^{\pi}_{t}(s,a) + \alpha(R_{a}(s,s') + \gamma Q^{\pi}_{t}(s',\pi(s')) - Q^{\pi}_{t}(s,a))
\end{equation}$$

其中指数平滑系数$\alpha$称为更新步长。

### 3.2.1 Sarsa算法

使用\eqref{td}，采用on-policy方法，对$\epsilon$-贪心策略进行执行和优化，就得到了Sarsa算法。具体流程如下：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi^{\epsilon}(s,a)=\epsilon\text{-greedy policy}$$；

设定初始状态$s$，并根据$\pi^{\epsilon}$生成初始动作$a$；

2）迭代：

基于$s$和$a$，观测奖励$r$和下一状态$s'$；

根据$\pi^{\epsilon}$生成下一动作$a'$；

更新$Q(s,a)$：

$$Q(s,a) = Q(s,a) + \alpha(r + \gamma Q(s',a') - Q(s,a))$$

求得状态$s$下的最优动作为$a^* = arg\max_{a'' \in A_s}Q(s,a'')$，且对所有$a'' \in A_s$，更新策略：

$$\pi^{\epsilon}(s,a'') = \begin{cases}1-\epsilon + \frac{\epsilon}{\mid A_s\mid} & a'' = a^* \\
\frac{\epsilon}{\mid A_s\mid} & a'' \neq a^* \\
\end{cases}$$

更新当前状态和当前动作：$s=s',a=a'$。

若采用$\epsilon$-soft 策略，经过足够多的迭代，其可收敛到$\epsilon$-soft空间下的最优策略（次优）。

### 3.2.2 Q-learning 算法

使用\eqref{td}，采用off-policy方法，执行$\epsilon$-贪心策略，但优化原始策略，就得到了Q-learning算法。具体流程如下：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi(s)=arg\max_{a\in A_s}Q(s,a)$$；

设定初始状态$s$；

2）迭代：

基于$s$，根据当前策略$\pi$的$\epsilon$-贪心策略$\pi^{\epsilon}$，生成一个动作$a$；

基于$s$和$a$，观测奖励$r$和下一状态$s'$；

根据$\pi$生成下一贪心动作$a'$；

更新$Q(s,a)$：

$$Q(s,a) = Q(s,a) + \alpha(r + \gamma Q(s',a') - Q(s,a))$$

更新策略$\pi(s) = arg\max_{a'' \in A_s}Q(s,a'')$；

更新当前状态：$s=s'$。

已经从理论上证明，当更新步长$\alpha > 0$随时间递减，Q-learning算法得出的$\pi$可以收敛到最优策略，如果是固定步长，则不能保证收敛，有可能会发散。

### 3.2.3 资格迹（eligibility trace）方法

在式\eqref{td}中，只使用当前步的预测误差$$\delta_t = R_{a}(s,s') + \gamma Q^{\pi}_{t}(s',\pi(s')) - Q^{\pi}_{t}(s,a)$$来更新上一步的$Q$值。

在资格迹的情况下，将及时进行回顾总结。例如，如果当前回报较好，则不仅更新上一状态-动作值，并为导致到达当前状态的某些先前状态-动作值分配一些回报。这将会进一步提高算法的收敛性。

根据折扣未来回报来估计的回报如下：

$$\begin{align}
G_t &= r_{t+1} + \gamma r_{t+2} + \gamma^2r_{t+3} + \cdots \\
&= r_{t+1} + \gamma Q_{t}(s_{t+1},a_{t+1}) \\
&= r_{t+1} + \gamma r_{t+2} + \gamma^2Q_{t}(s_{t+2},a_{t+2})
\end{align}
$$

资格迹法会跟踪访问到当前状态的轨迹，然后将回报分配给最近访问的状态-动作值。而长时间没有被访问的状态-动作对则没有资格获得当前回报。

定义时刻$t$每个状态-动作对的资格迹值为$e_t(s,a)$，并随着时间以$\gamma\lambda$的速率衰减。但对于刚访问过的状态-动作对，其资格迹值为增加1。因此，资格迹值更新规则如下：

$$
e_t(s,a) = \begin{cases} \gamma\lambda e_{t-1}(s,a) + 1 & s=s_t, a=a_t \\
\gamma\lambda e_{t-1}(s,a) & \text{others}
\end{cases}
$$

因此，可以使用资格迹来改进Q学习（称为Q($\lambda$)学习）算法如下：

1）初始化：

$$\forall s \in S, \forall a\in A_s, Q(s,a)=\text{arbitrary}, \pi(s)=arg\max_{a\in A_s}Q(s,a), e(s,a) = 0$$；

设定初始状态$s$；

2）迭代：

基于$s$，根据当前策略$\pi$的$\epsilon$-贪心策略$\pi^{\epsilon}$，生成一个动作$a$；

基于$s$和$a$，观测奖励$r$和下一状态$s'$；

根据$\pi$生成下一贪心动作$a'$；

计算TD误差：$$\delta = r + \gamma Q(s',a') - Q(s,a)$$；

计算资格迹值：$e(s,a) = e(s,a) + 1$；

$$\forall s \in S, \forall a\in A_s$$，更新$Q(s,a)$和$e(s,a)$：

$$Q(s,a) = Q(s,a) + \alpha\delta e(s,a)$$

$$e(s,a) = \gamma\lambda e(s,a)$$

$$\forall s \in S$$，更新策略$\pi(s) = arg\max_{a'' \in A_s}Q(s,a'')$；

更新当前状态：$s=s'$。

Q($\lambda$)学习一般来说比Q学习收敛速率高一些，但复杂度也要高，具体选择哪个应该根据实际情况进行考虑。


# 4、模仿学习

在强化学习的经典任务设置中，机器所能获得的反馈信息仅有每步决策后的奖励，但是在现实任务中，往往能够得到人类专家的决策过程范例。从这样的范例中学习，称为“模仿学习（imitation learning）”。

## 4.1 直接模仿学习

直接模仿学习是指直接对人类专家的“状态-动作”对进行学习，即把状态作为特征，动作作为标记，然后对数据集使用分类（对于离散动作）或回归（对于连续动作）算法即可学得策略模型。学得的这个策略模型可作为机器进行强化学习的初始策略，再通过强化学习方法基于环境反馈进行改进，从而获得更好的策略。

## 4.2 逆强化学习

在很多实际任务中，设计奖励函数往往比较困难，从人类专家提供的范例数据中反推出奖励函数可以解决该问题，这就是逆强化学习（inverse reinforcement learning）。在逆强化学习中，我们知道状态空间、动态空间，并且与直接模仿学习类似，有一个人类专家的范例决策轨迹数据集。逆强化学习的基本思想是：欲使机器做出与范例一致的行为，等价于在某个奖励函数的环境中求解最优策略，该最优策略所产生的轨迹与范例数据一致。换言之，我们要寻找某种奖励函数使得范例数据是最优的，然后即可使用这个奖励函数来训练强化学习策略。



