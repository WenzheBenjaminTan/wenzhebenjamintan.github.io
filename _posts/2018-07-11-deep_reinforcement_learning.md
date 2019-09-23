---
layout: post
title: "深度强化学习" 
---

# 1、值函数拟合（value function approximation）

当状态空间或动作空间无穷大，甚至是连续的时，原来传统的强化学习方法不再有效。这个时候可以直接针对值函数进行学习。

## 1.1 Q函数拟合

### 1.1.1 基本方法

因为无模型情况下，学习$Q$函数是更有效的方式，我们可以采用$Q_{\boldsymbol{w}}(s,a)$（其中$\boldsymbol{w}$为参数向量）来拟合Q函数。

我们希望通过$Q_{\boldsymbol{w}}$来拟合真实的值函数$Q^{\pi}$，若使用最小二乘误差为目标，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \frac{1}{2}\alpha\nabla_{\boldsymbol{w}} (Q^{\pi}(s_t,a_t)-Q_{\boldsymbol{w}_t}(s_t,a_t))^2 \\
			&= \boldsymbol{w}_t + \alpha (Q^{\pi}(s_t,a_t)-Q_{\boldsymbol{w}_t}(s_t,a_t))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}_t}(s_t,a_t)
\end{align}
$$

我们并不知道策略的真实值函数$Q^{\pi}(s_t,a_t)$，但可借助当前值函数的估计来代替，如时序差分学习中使用的$$r_t+\gamma Q_{\boldsymbol{w}_t}(s_{t+1},a_{t+1})$$，因此更新规则可以修改为：

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

更新策略$\pi(s) = arg\max_{\widetilde{a} \in A_s}Q_{\boldsymbol{w}}(s,\widetilde{a})$；

更新当前状态：$s=s'$。

该算法是on-line和off-policy的，但因为在函数拟合中使用的训练样本并不满足独立同分布特性，因此不一定能保证收敛。

### 1.1.2 DQN

深度Q网络结构模型（Deep Q-Network，DQN）可谓是深度强化学习的开山之作，由DeepMind在NIPS 2013上发表，后又在Nature 2015上提出改进版本。

用人工神经网络来拟合Q函数有两种形式：一种形式是将状态和动作当作输入，然后通过人工神经网络分析后得到动作的Q值，这对于动作空间是连续的时尤其有效；而当动作空间是离散的时，可以采用另一种更直观的形式，即只输入状态，然后输出所有的动作Q值。这相当于先接收外部的信息，然后通过大脑加工输出每种动作的值，最后选择拥有最大值的动作当作下一步要做的动作。由于当动作空间是连续的时候，采用第一种形式去求$$arg\max_{\widetilde{a} \in A_s}Q_{\boldsymbol{w}}(s,\widetilde{a})$$是非常困难的，所以DQN一般只采用第二种形式来拟合Q函数，也就是说DQN对于处理连续动作空间的学习问题是有局限的。

采用“半梯度下降法”来训练人工神经网络，因为训练样本相互关联（当利用每条episode顺序训练时），训练过程中策略会进行剧烈的振荡，从而使收敛速度十分缓慢。该问题严重影响了深度学习在强化学习中的应用。为了降低样本关联性，DQN采用了两大利器：Experience Replay 和 Fixed Q-targets。


**1、Experience Replay**

DQN有一个记忆池（replay memory）用于记录之前的经历。前面提到过，Q-learning是一种off-policy的学习方法，它能学习当前经历着的，也能学习过去经历过的，甚至是学习别人的经历。所以每次DQN更新的时候，我们都可以从记忆池中随机抽取一些之前的经历进行学习。

Experience Replay的具体操作步骤如下：

1）在算法执行前首先开辟一个记忆池空间$D$；

2）利用记忆池$D$来采样样本，在每个时间步$t$，将Agent与环境交互得到的转移样本$e^{(t)}=(s_t,a_t,r_t,s_{t+1})$存储到记忆池$$D=\{e^{(1)},e^{(2)},...,e^{(t)},...\}$$，当达到空间最大值后替换原来的样本；

3）从$D$中随机抽取一个小批量（minibatch）的转移样本；

4）使用随机选取的样本，根据贝尔曼方程估算得到这些样本中$(s,a)$对应的目标Q值；

5）通过该目标值和当前网络的估计值的差量来更新网络模型的参数。

记忆池的引入主要起到如下作用：

1）打破可能陷入局部最优的可能；

2）模拟监督学习；

3）打破数据之间的关联性。

**2、Fixed Q-targets**

在Nature 2015版本的DQN中提出了Fixed Q-targets的改进。Fixed Q-targets 也是一种打乱相关性的机理。如果使用Fixed Q-targets，我们就会在DQN中使用到两个结构相同但参数不同的神经网络：一个是MainNet，另一个是TargetNet。MainNet用于评估当前状态-动作对的值函数（称为Q-eval值），TargetNet用于根据贝尔曼方程估算目标Q值（称为Q-target值），然后根据目标值和估计值的差量来更新MainNet的参数，且每经过$C$轮迭代，将MainNet的参数复制给TargetNet。

引入Fixed Q-targets后，一定程度降低了当前Q值和目标Q值的相关性，提高了算法稳定性。

### 1.2.3 Double DQN

我们进一步展开DQN中的Q-target计算公式可得：

$$y^{DQN} = r + \gamma Q(s',arg\max_{a'} Q(s',a';\boldsymbol{\theta}^-); \boldsymbol{\theta}^-)$$

也就是说根据状态$s'$选择动作$a'$的过程，以及估计$Q(s',a')$使用的是同一张$Q$值表，或者说是使用的同一个网络参数，这可能会导致过高的估计值（overestimate）。而Double DQN就是用来解决这种过估计的，它的想法是引入另一个神经网络来减小这种误差。而DQN中本来就有两个神经网络，我们刚好可以利用这一点，改用MainNet来选择动作$a'$，而继续用TargetNet来估计$Q(s',a')$。于是采用Double DQN的Q-target计算公式变为：

$$y^{DoubleDQN} = r + \gamma Q(s',arg\max_{a'} Q(s',a';\boldsymbol{\theta}); \boldsymbol{\theta}^-)$$

其中$$\boldsymbol{\theta}^-$$是TargetNet的参数，$$\boldsymbol{\theta}$$是MainNet的参数。


## 1.2 V函数拟合

类似地，我们可以采用$V_{\boldsymbol{w}}(s)$（其中$\boldsymbol{w}$为参数向量）来拟合$$V^{\pi}$$函数，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t + \alpha (r_t + \gamma V_{\boldsymbol{w}_t}(s_{t+1}) -V_{\boldsymbol{w}_t}(s_t))\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}_t}(s_t)
\end{align}
$$

# 2、策略梯度法（policy gradient）

策略梯度法是另一种深度强化学习方法，它将策略看作一个基于策略参数的函数，通过不断计算策略期望总奖励关于策略参数的梯度来更新策略参数，最终收敛于最优策略。策略梯度法对于状态空间特别大甚至是连续的情况尤其有效。如果动作空间是离散的，常用随机型策略网络来表示，网络的输入是状态$s$，输出是每种动作$a$的概率；如果动作空间是连续的，一般用确定型策略网络表示，网络的输入是状态$s$，输出直接是动作$a$。一般我们说策略梯度法都是针对随机型策略，确定型策略将在DPG算法时具体介绍。

随机型策略可以表示为$\pi_{\boldsymbol{\theta}}(s,a)$，即在策略参数$\boldsymbol{\theta}$下，根据当前状态$s$选择动作$a$的概率。

我们定义强化学习的目标函数$J(\boldsymbol{\theta})$为在策略$\pi_{\boldsymbol{\theta}}$下的回报期望。学习过程可以表示为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

可见整个更新过程就是一个梯度上升法。

## 2.1 从似然概率角度推导策略梯度

用$\tau$表示一组随机状态-动作轨迹$$ < S_0,A_0,S_1,A_1,...,S_{T-1},A_{T-1},S_T  > $$。并令$$R(\tau) = \sum_{t=0}^{T-1}\gamma^tr_{S_t,A_t}(S_{t+1})$$表示轨迹$\tau$的回报，$$P(\tau\mid\boldsymbol{\theta})$$表示在$$\pi_{\boldsymbol{\theta}}$$作用下轨迹$\tau$出现的似然概率。则强化学习的目标函数可以表示为：

$$J(\boldsymbol{\theta}) = E_{\tau\sim P(\tau\mid\boldsymbol{\theta})}\left[ R(\tau) \right] = \sum_{\tau} R(\tau)P(\tau\mid\boldsymbol{\theta})$$

因此，对目标函数求导可得：

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \sum_{\tau} R(\tau)P(\tau\mid\boldsymbol{\theta})\\
								&= \sum_{\tau} R(\tau) \nabla_{\boldsymbol{\theta}}P(\tau\mid\boldsymbol{\theta}) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \frac{\nabla_{\boldsymbol{\theta}}P(\tau\mid\boldsymbol{\theta})}{P(\tau\mid\boldsymbol{\theta})} \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \nabla_{\boldsymbol{\theta}}\log p(\tau\mid\boldsymbol{\theta}) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \nabla_{\boldsymbol{\theta}}\log \left(p(S_0)\prod_{t=0}^{T-1}\pi_{\boldsymbol{\theta}}(S_t,A_t)p_{S_t,A_t}(S_{t+1})\right) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \nabla_{\boldsymbol{\theta}} \left(\log p(S_0)+ \sum_{t=0}^{T-1}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)+ \sum_{t=0}^{T-1}\log p_{S_t,A_t}(S_{t+1})\right) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})}\left[R(\tau) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\

								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^t r_{S_t,A_t}(S_{t+1}) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\


\end{align}$$

因此，性能梯度可以用策略梯度来表示，但该估计的方差非常大。

注意到对$$t' < t$$，有

$$E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\gamma^{t'} r_{S_{t'},A_{t'}}(S_{t'+1}) \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\gamma^{t'} r_{S_{t'},A_{t'}}(S_{t'+1}) \right]E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = 0 $$

因此

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^t r_{S_t,A_t}(S_{t+1}) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\sum_{t'=t}^{T-1}\gamma^{t'} r_{S_{t'},A_{t'}}(S_{t'+1}) \right] \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^tR_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \right] \label{eq15} \\

\end{align}$$


## 2.2 REINFORCE算法


当利用当前策略$$\pi_{\boldsymbol{\theta}}$$采样$N$条轨迹后，可以利用$N$条轨迹的经验平均逼近目标函数的导数：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1} \gamma^t R_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

上式估计的方差受$$R_t^{(i)}$$的影响仍然会很大。我们可以在回报中引入常数基线$b$以减小方差且保持期望不变。

因为有

$$E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[b \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = 0 $$

所以容易得到修改后的估计：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1} (\gamma^t R_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

我们进而可求使得估计方差最小时的基线$b$。

令$$X = \sum_{t=0}^{T-1} (\gamma^t R_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t^{(i)},A_t^{(i)})$$，则方差为：

$$Var(X) = E(X-\overline{X})^2 = EX^2 - \overline{X}^2$$

其中$$\overline{X} = EX$$与$b$无关，令$$Var(X)$$对$b$的导数为0，即

$$\frac{\partial Var(X)}{\partial b} = E\left[X\frac{\partial X}{\partial b}\right] = 0$$

求解可得：

$$b= \frac{\sum_{i=1}^N \sum_{t=0}^{T-1}\gamma^t R_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})  
}{\sum_{i=1}^N \left(\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\right)^2}$$

于是可以得到基本的REINFORCE算法（policy gradient based reinforcement learning）如下：

1）初始化参数$\boldsymbol{\theta}$和步长$\alpha > 0$；

2）迭代：

根据策略$\pi_{\boldsymbol{\theta}}$生成$N$个采样片段$$ < s_0^{(i)},a_0^{(i)},r_0^{(i)},s_1^{(i)},a_1^{(i)},r_1^{(i)},...,s_{T-1}^{(i)},a_{T-1}^{(i)},r_{T-1}^{(i)},s_T^{(i)} > $$（$$i=1,2,...,N$$），然后计算常数$b$:

$$b= \frac{\sum_{i=1}^N \sum_{t=0}^{T-1}\gamma^t R_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})}{\sum_{i=1}^m \left(\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\right)^2}$$

并更新$\theta$:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \frac{\alpha}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} (\gamma^t R_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

显然，基本的REINFORCE算法需要得到每个完整的episode再更新$\boldsymbol{\theta}$，因此它是off-line和on-policy的。



## 2.3 策略梯度定理

实际上，我们还可以利用贝尔曼方程来对性能梯度和策略梯度之间的关系进行推导：

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} V^{\pi_{\boldsymbol{\theta}}}(s), \forall s \in S \\
								&= \nabla_{\boldsymbol{\theta}} \left[\sum_a \pi_{\boldsymbol{\theta}}(s,a) Q^{\pi_{\boldsymbol{\theta}}}(s,a) \right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a)\nabla_{\boldsymbol{\theta}}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a)\nabla_{\boldsymbol{\theta}} \sum_{s'}p_{s,a}(s')(r_{s,a}(s') +\gamma V^{\pi_{\boldsymbol{\theta}}}(s'))\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a) \sum_{s'}\gamma p_{s,a}(s')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s')\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a) \sum_{s'}\gamma p_{s,a}(s') \sum_{a'} \left[ Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \pi_{\boldsymbol{\theta}}(s',a') \sum_{s''}\gamma p_{s',a'}(s'')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s'')\right] \right] \\
								&= \sum_a  Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \gamma\sum_a\pi_{\boldsymbol{\theta}}(s,a) \sum_{s'} p_{s,a}(s') \sum_{a'} Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \gamma^2\sum_a\pi_{\boldsymbol{\theta}}(s,a) \sum_{s'} p_{s,a}(s') \sum_{a'} \pi_{\boldsymbol{\theta}}(s',a') \sum_{s''} p_{s',a'}(s'')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s'') \\

\end{align}
$$

为了化简公式，我们定义$$P(s\rightarrow x,k,\pi_{\boldsymbol{\theta}})$$为在策略$$\pi_{\boldsymbol{\theta}}$$作用下从状态$s$开始经过$k$时刻到达状态$x$的概率。

那么公式就变为：

$$\begin{align} &= P(s\rightarrow s,0,\pi_{\boldsymbol{\theta}})\sum_a  Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \gamma \sum_{s'} P(s\rightarrow s',1,\pi_{\boldsymbol{\theta}}) \sum_{a'} Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \gamma^2 \sum_{s''} P(s\rightarrow s'',2,\pi_{\boldsymbol{\theta}})\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s'') 
\end{align}$$

将其无限展开可得：

$$\begin{align} &= P(s\rightarrow s,0,\pi_{\boldsymbol{\theta}})\sum_a  Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \gamma \sum_{s'} P(s\rightarrow s',1,\pi_{\boldsymbol{\theta}}) \sum_{a'} Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \cdots + \gamma^k \sum_{s^k} P(s\rightarrow s^k,k,\pi_{\boldsymbol{\theta}})\sum_{a^k} Q^{\pi_{\boldsymbol{\theta}}}(s^k,a^k)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s^k,a^k) + \cdots \\
	&= \sum_{x\in S}\sum_{k=0}^{\infty}\gamma^k P(s\rightarrow x,k,\pi_{\boldsymbol{\theta}})\sum_a  Q^{\pi_{\boldsymbol{\theta}}}(x,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(x,a)
\end{align}$$

我们定义$$\rho_{\pi_{\boldsymbol{\theta}}}(x) = \sum_{k=0}^{\infty}\gamma^k P(s\rightarrow x,k,\pi_{\boldsymbol{\theta}})$$，其表示按照策略$$\pi_{\boldsymbol{\theta}}$$从起始状态$s$到状态$x$的总可能性，并且根据步数进行了折扣加权，称作状态$x$的加权访问概率。

策略梯度定理可以表示为如下：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

它直观地给出了性能梯度与策略梯度之间的关系。

## 2.4 行动者-评论家（Actor-Critic）算法

### 2.4.1 基本的Actor-Critic算法

根据策略梯度定理，可以进而推导如下：

$$\begin{align}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= \sum_s\sum_{k=0}^{\infty} P(s_0\rightarrow s,k,\pi_{\boldsymbol{\theta}})\sum_a \gamma^k Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= E_{\pi_{\boldsymbol{\theta}}}\left(\sum_{a}\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^t\sum_{a}\pi_{\boldsymbol{\theta}}(S_t,a)Q^{\pi_{\boldsymbol{\theta}}}(S_t,a)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)}{\pi_{\boldsymbol{\theta}}(S_t,a)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,A_t)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,A_t)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right) \\
\end{align}$$

其中，对于$$Q^{\pi_{\boldsymbol{\theta}}}(s,a)$$我们可以像值函数拟合方法一样采用$$Q_{\boldsymbol{w}}(s,a)$$来拟合它（$\boldsymbol{w}$为参数向量），于是得到策略梯度法的更新公式：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t Q_{\boldsymbol{w}}(s_t,a_t)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$

这就是基本的Actor-Critic算法，它是on-line和on-policy的。Actor-Critic算法同时对策略和值函数进行建模，通过值函数的估计来辅助策略函数的更新。在这个过程中策略模型被称为（Actor），价值模型被称为评论家（Critic）。


基本的Actor-Critic算法的具体流程如下：

1）初始化：
设定参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$，梯度乘子$I=1$；

设定初始状态$s$，并根据策略$\pi_{\boldsymbol{\theta}}$生成初始动作$a$；

2）迭代：

基于当前状态$s$和动作$a$，观测奖励$r$和下一状态$s'$；

根据策略$\pi_{\boldsymbol{\theta}}$生成下一动作$a'$；

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}(r+\gamma Q_{\boldsymbol{w}}(s',a')-Q_{\boldsymbol{w}}(s,a))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}}(s,a)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}IQ_{\boldsymbol{w}}(s,a)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s,a)$$

$$I \leftarrow \gamma I$$

更新当前状态和动作：$$s \leftarrow s'$$，$$a \leftarrow a'$$。

由于是采用“半梯度下降法”让$$Q_{\boldsymbol{w}}(s,a)$$逐渐去拟合$$Q^{\pi_{\boldsymbol{\theta}}}(s,a)$$，这种方法的收敛性并不好。

### 2.4.2 Advantage Actor-Critic算法


实际上，$$R_t$$可以视为$$Q^{\pi_{\boldsymbol{\theta}}}(s_t,a_t)$$的估计，所以策略梯度法的更新公式可以写为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t R_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$


因为其更新的幅度依赖于某episode中$t$时刻到结束时刻的真实样本回报$R_t$，所以估计方差仍然是很大的。更常见的一种做法也是引入一个基准（baseline）$b(s)$，且可以满足：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) =\sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) =\sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}\left(Q^{\pi_{\boldsymbol{\theta}}}(s,a)-b(s)\right)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

$b(s)$可以取任何常数或函数，只要不和$a$相关就不影响上式的结果。因为：

$$\sum_a b(s) \nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) = b(s) \nabla_{\boldsymbol{\theta}}\sum_a\pi_{\boldsymbol{\theta}}(s,a) = b(s)\nabla_{\boldsymbol{\theta}}1 = 0$$

于是更新公式修改为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t (R_t-b(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$

至于$b(s)$怎么设计，取决于算法，但一般的做法是取$b(s) = V_{\boldsymbol{w}}(s)$，也就是说用另一个函数来估计状态均值。容易得到，这种情况下参数的更新主要取决于在状态$s_t$下执行动作$a_t$所得总奖励相对于状态均值的优势，如果有优势，则更新后的参数会增加执行该动作的概率；如果没有优势，则更新后的参数会减少执行该动作的概率。而$$R_t - b(s_t)$$正是对动作优势函数$$A^{\pi_{\boldsymbol{\theta}}}(s_t,a_t) = Q^{\pi_{\boldsymbol{\theta}}}(s_t,a_t) - V^{\pi_{\boldsymbol{\theta}}}(s_t)$$的估计，因为$$R_t$$为$$Q^{\pi_{\boldsymbol{\theta}}}(s_t,a_t)$$的估计，$$b(s_t) = V_{\boldsymbol{w}}(s_t)$$可以视为$$V^{\pi_{\boldsymbol{\theta}}}(s_t)$$的估计。

此外，为了避免off-line地求得全部回报$$R_t$$，我们采用单步的奖励和下个状态估值的和式$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来估计$R_t$（注意，这会使估计的偏差增大），于是参数更新公式变为：

$$\begin{align}
\boldsymbol{\theta}' &= \boldsymbol{\theta} + \alpha\gamma^t (r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})-V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t) \\
\end{align}$$

这就得到了Advantage Actor-Critic算法，此时评论家（Critic）变成了状态值函数。

根据贝尔曼方程可得到：

$$V^{\pi}(s) = E\left[r_{s,\pi(s)}(s') + \gamma V^{\pi}(s')\right]$$

所以状态值函数模型可以使用$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$与$$V_{\boldsymbol{w}}(s_t)$$之间的均方误差（MSE）作为损失函数。

Advantage Actor-Critic算法的具体流程如下：

1）初始化参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$，当前状态$s=s_0$，梯度乘子$I=1$；

2）迭代：

基于当前状态$s$，根据策略$\pi_{\boldsymbol{\theta}}$生成一个动作$a$并得到奖励$r$和下一状态$s'$，并执行如下步骤：

$$\delta \leftarrow r+\gamma V_{\boldsymbol{w}}(s')-V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\delta\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s,a)$$

$$I \leftarrow \gamma I$$

$$s \leftarrow s'$$

其中$\delta$是优势函数的估计。


### 2.4.3 异步并行版本的Actor-Critic算法（A3C）

真正将Actor-Critic应用到实际中并得到优异效果的是A3C（Asynchronous Advantage Actor-Critic）算法，从算法的名字可以看出，算法突出了异步并行的概念。

由于Actor-Critic算法是on-policy的（每一次模型更新都需要“新样本”），为了更快地收集样本，我们需要用并行的方法来收集。在A3C方法中，我们要同时启动$N$个线程，每个线程中有一个Agent与环境进行交互。收集完样本后，每一个线程将独立完成训练并得到参数更新量，并异步地更新到全局的模型参数中。下一次训练时，线程的模型参数和全局参数完成同步，再使用新的参数进行新的一轮训练。

前面提到Advantage Actor-Critic算法中使用$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来有偏估计$R_t$，这个方法虽然增加了学习的稳定性（即减小了方差），但是学习的偏差也相应变大，为了更好地平衡偏差和方差，A3C方法使用$n$步回报估计法，这个方法可以在训练早期更快地提升价值模型。对应的优势函数估计公式变为：

$$\sum_{i=0}^{n-1}\gamma^ir_{t+i} + \gamma^{n}V_{\boldsymbol{w}}(s_{t+n}) - V_{\boldsymbol{w}}(s_t)$$

最后，为了增加模型的探索性，模型的目标函数中加入了策略的熵。由于熵可以衡量概率分布的不确定性，所以我们希望模型的熵尽可能大一些，这样模型就可以拥有更好的多样性。这样，完整的策略梯度更新公式就变为：

$$\begin{align}
\boldsymbol{\theta}' &= \boldsymbol{\theta} + \alpha\gamma^t (\sum_{i=0}^{n-1}\gamma^ir_{t+i} + \gamma^{n}V_{\boldsymbol{w}}(s_{t+n}) - V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t) + \beta \nabla_{\boldsymbol{\theta}} Ent(\pi_{\boldsymbol{\theta}}(s_t)) \\
\end{align}$$

其中$\beta$为策略的熵在目标中的权重。

价值模型的目标函数与DQN类似，不再赘述。于是，A3C的完整算法如下：

定义：全局时间钟为$T$；全局的价值和策略参数分别为$\boldsymbol{w}$和$\boldsymbol{\theta}$，线程内部的价值和策略参数分别为$\boldsymbol{w'}$和$\boldsymbol{\theta'}$。

01：初始化全局时间钟$T\leftarrow 0$，线程时间钟 $t\leftarrow 0$；

02：repeat（对每个线程）

03：&emsp; 将梯度清零：$d\boldsymbol{w} \leftarrow 0$, $d\boldsymbol{\theta} \leftarrow 0$；

04：&emsp; 同步模型参数：$\boldsymbol{w'} \leftarrow \boldsymbol{w}$, $\boldsymbol{\theta'} \leftarrow \boldsymbol{\theta}$；

05：&emsp; $$t_{start} = t$$；

06：&emsp; 获取该线程内当前状态$$s_t$$；

07：&emsp; repeat

08：&emsp; &emsp; 根据当前策略$$\pi_{\boldsymbol{\theta'}}(s_t,a_t)$$执行$$a_t$$；

09：&emsp; &emsp; $t\leftarrow t+1$；

10：&emsp; &emsp; $T\leftarrow T+1$；

11：&emsp; until $$s_t$$是终止状态或者$$t-t_{start} == n$$

12：&emsp; 如果$$s_t$$是终止状态，令$R=0$，否则令$$R = V_{\boldsymbol{w'}}(s_t)$$；

13：&emsp; for $i = t-1,...,t_{start}$ do

14：&emsp; &emsp; $$R \leftarrow r_i + \gamma R$$；

15：&emsp; &emsp; 积累价值模型梯度：$$d\boldsymbol{w}\leftarrow d\boldsymbol{w} + \partial(R-V_{\boldsymbol{w'}}(s_i))^2/\partial \boldsymbol{w'}$$；

16：&emsp; &emsp; 积累策略模型梯度：$$d\boldsymbol{\theta}\leftarrow d\boldsymbol{\theta} +\gamma^i(R-V_{\boldsymbol{w'}}(s_i))\nabla_{\boldsymbol{\theta'}}\log\pi_{\boldsymbol{\theta'}}(s_i,a_i) + \beta \nabla_{\boldsymbol{\theta'}} Ent(\pi_{\boldsymbol{\theta'}}(s_i))$$；

17：&emsp; end for

18：&emsp; 使用$d\boldsymbol{w}$ 和 $d\boldsymbol{\theta}$分别异步更新$\boldsymbol{w}$和$\boldsymbol{\theta}$；

19：until $T > T_{max}$

### 2.4.4 同步并行版本的Actor-Critic算法

由于十分优异的效果，A3C算法的影响力极大。但大家一直存在一个疑问：算法中的异步更新是否是必要的？凭直觉，异步或者同步更新并不是决定算法优劣的主要因素，那么为什么不尝试使用同步更新的方法呢？

在OpenAI的Baseline项目中，实现了同步并行版本的A2C算法，而且其在并发行和系统简洁性上都优于A3C。下面我们具体介绍一下其实现的细节。

该算法在与环境交互时采用了Master-Slave结构，其中的Master用于执行Agent模型，在Env给出状态观测值后判断应当执行的Action；而Slave用于模拟Env，在收到Agent的Action后，将后续的状态观测值、回报、是否结束等信息返回给Master，Master将Slave返回过来的数据收集起来并进行训练。

整体算法流程分为两个部分：环境交互和样本训练。在与环境交互时，使用$$step\_model$$分别从每个线程的环境中执行$steps$步，并输出一批经验样本。在交互完成后使用$$train\_model$$和这一批经验样本来计算每个线程中的梯度，并取平均值，然后更新$$step\_model$$中的参数。



## 2.5 其他策略梯度法

策略梯度法有两个软肋：

1）波动性：REINFORCE算法通过采样一条或几条轨迹后来估计回报，估计波动就很大，而基本的Actor-Critic算法采用状态-动作值函数拟合后估计的波动更大；Advantage Actor-Critic算法引入状态值时间差分的方式来估计优势函数，通过增加偏差的方式来换取方差的降低；并行版本的Actor-Critic算法只是通过多步回报估计的方式平衡了方差和偏差。总的来说，波动性的问题依然存在。

2）样本利用率：这个问题是所有on-policy算法都要解决的。因为on-policy算法在每一次策略发生改变时，都要丢弃前面产生的样本，这将带来很大的样本浪费，我们需要考虑用off-policy的算法来进行学习。

解决第一个问题的代表算法是TRPO和PPO，而解决第二个问题的代表算法是ACER和DPG。


### 2.5.1 TRPO

TRPO是置信区域策略优化（Trust Region Policy Optimization）算法的简称，它可以确保策略模型在优化时单调提升。其主要思路是找到一种衡量策略之间优劣的计算方法，并以此为目标最大化新策略与旧策略相比的优势。

1) 策略差距的推导

我们先定义基于某个策略的期望价值：

$$\eta(\pi) = E_{s_0,a_0,...\sim \pi}\left[\sum_{t=0}^{\infty}\gamma^tr_{s_t,a_t}(s_{t+1})\right]$$

其中$$s_0\sim P(s_0\rightarrow s_0,0,\pi)$$，$$a_t\sim\pi(s_t)$$，$$s_{t+1}\sim p_{s_t,a_t}(s_{t+1})$$。

接下来给出状态-动作值函数、状态值函数和优势函数的定义：

$$Q^{\pi}(s_t,a_t) = E_{s_{t+1},a_{t+1},...\sim \pi}\left[\sum_{l=0}^{\infty}\gamma^lr_{s_{t+l},a_{t+l}}(s_{t+l+1})\right]$$

$$V^{\pi}(s_t) = E_{a_t,s_{t+1},...\sim \pi}\left[\sum_{l=0}^{\infty}\gamma^lr_{s_{t+l},a_{t+l}}(s_{t+l+1})\right]$$

$$A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)$$

于是我们可以得到：

$$\begin{align}&E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^tA^{\pi}(s_t,a_t)\right] \\
&= E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^tE_{s'\sim p_{s_t,a_t}(s')}\left[r_{s_t,a_t}(s') + \gamma V^{\pi}(s') - V^{\pi}(s_t)\right]\right] \\
&= E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t(r_{s_t,a_t}(s_{t+1}) + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t))\right] \\
&= E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^tr_{s_t,a_t}(s_{t+1})\right] + E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t(\gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t))\right] \\
&= \eta(\widetilde{\pi}) + E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t(\gamma E_{a_{t+1},s_{t+2},...\sim \pi}\left[\sum_{l=0}^{\infty}\gamma^{l}r_{s_{t+l+1},a_{t+l+1}}(s_{t+l+2})\right] - E_{a_{t},s_{t+1},...\sim \pi}\left[\sum_{l=0}^{\infty}\gamma^lr_{s_{t+l},a_{t+l}}(s_{t+l+1})\right])\right] \\
&= \eta(\widetilde{\pi}) + E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t(E_{a_t,s_{t+1},...\sim \pi}\left[\sum_{l=0}^{\infty}\gamma^{l+1}r_{s_{t+l+1},a_{t+l+1}}(s_{t+l+2}) - \sum_{l=0}^{\infty}\gamma^lr_{s_{t+l},a_{t+l}}(s_{t+l+1})\right]) \right] \\
&= \eta(\widetilde{\pi}) + E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t(E_{a_t,s_{t+1},...\sim \pi}\left[-r_{s_{t},a_{t}}(s_{t+1})\right]) \right] \\
&= \eta(\widetilde{\pi}) + E_{s_0,a_0,...\sim \widetilde{\pi}}\left[-\eta(\pi)\right] \\
&= \eta(\widetilde{\pi}) - \eta(\pi)

\end{align}$$

经过前面的推导，我们找到了两个策略差距的基本形式。上面的公式并不能直接计算，我们还要对公式做进一步的变换：

$$\begin{align}
\eta(\widetilde{\pi}) - \eta(\pi) &= E_{s_0,a_0,...\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^tA^{\pi}(s_t,a_t)\right] \\
				&= \sum_{t=0}^{\infty}\sum_sP(s_0\rightarrow s,t,\widetilde{\pi})\sum_a\widetilde{\pi}(s,a)\gamma^tA^{\pi}(s,a) \\
				&= \sum_s\sum_{t=0}^{\infty}\gamma^tP(s_0\rightarrow s,t,\widetilde{\pi})\sum_a\widetilde{\pi}(s,a)A^{\pi}(s,a) \\
				&= \sum_s\rho_{\widetilde{\pi}}(s)\sum_a\widetilde{\pi}(s,a)A^{\pi}(s,a) 
\end{align}
$$

也就是说，我们从某个策略$$\pi_0$$出发，通过计算找到一个策略$$\pi_1$$，使得：

$$\sum_s\rho_{\pi_1}(s)\sum_a\pi_1(s,a)A^{\pi_0}(s,a)\geq 0 $$

那么我们就可以确定策略$$\pi_1$$在总体上优于$$\pi_0$$。依次类推，我们可以不断地找到效果更好的策略，直至达到目标，这就是算法单调提升的基本原理。

2）策略提升的可行公式

上一节得到了策略提升的目标，但是受目标公式所限，这样寻找策略的方法在实际中几乎是不可行的。因为公式中包含$$\rho_{\pi_1}(s)$$，也就是说对于每一个可能的新策略，我们都需要根据该新策略与环境交互得到所有状态的加权访问概率，这样的更新过程会非常慢。

因此，为了让计算变得可行，我们需要找到与上面公式近似且可解的形式，可对原公式近似如下：

$$L_{\pi}(\widetilde{\pi}) = \eta(\pi) + \sum_s\rho_{\pi}(s)\sum_a\widetilde{\pi}(s,a)A^{\pi}(s,a)$$

可以看出，唯一的变动在于状态加权访问概率上，用$$\rho_{\pi}(s)$$取代了$$\rho_{\widetilde{\pi}}(s)$$。实际上可以证明，对当前策略$\pi$来说，$$L_{\pi}$$和$\eta$在数值和一阶导数上都是相等的：

$$L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})$$

$$\nabla_{\theta}L_{\pi_{\theta_0}}(\pi_{\theta})\mid_{\theta=\theta_0} = \nabla_{\theta}\eta(\pi_{\theta})\mid_{\theta=\theta_0}$$

既然数值相同，导数方向相同，那么我们沿着近似目标函数的导数方向做有限步长的变化，也同样可以提升策略。

我们重写一遍新策略的回报函数和替代回报函数如下：

$$\eta(\widetilde{\pi}) = \eta(\pi) + E_{s_t\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t\overline{A}^{\pi,\widetilde{\pi}}(s_t)\right]$$

$$L_{\pi}(\widetilde{\pi}) = \eta(\pi) + E_{s_t\sim \pi}\left[\sum_{t=0}^{\infty}\gamma^t\overline{A}^{\pi,\widetilde{\pi}}(s_t)\right]$$

我们定义$$n_t$$表示当$$i < t$$时，$$a_i\neq \widetilde{a}_i$$的次数，则可得：

$$E_{s_t\sim \widetilde{\pi}}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] = P(n_t=0)E_{s_t\sim \widetilde{\pi}\mid n_t=0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] + P(n_t>0)E_{s_t\sim \widetilde{\pi}\mid n_t>0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]$$

设$$a_i\neq \widetilde{a}_i$$发生的概率为常数$$\alpha$$，则对于前$t$个状态，动作完全相同的概率$$P(n_t=0) = (1-\alpha)^t$$，且此时

$$E_{s_t\sim \widetilde{\pi}\mid n_t=0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] = E_{s_t\sim \pi\mid n_t=0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]$$

因此等式变为：

$$E_{s_t\sim \widetilde{\pi}}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] = (1-\alpha)^tE_{s_t\sim \pi\mid n_t=0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] + (1-(1-\alpha)^t)E_{s_t\sim \widetilde{\pi}\mid n_t>0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]$$

两边分别减去

$$E_{s_t\sim \pi}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] = (1-\alpha)^tE_{s_t\sim \pi\mid n_t=0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] + (1-(1-\alpha)^t)E_{s_t\sim \pi\mid n_t>0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]$$

可得：

$$E_{s_t\sim \widetilde{\pi}}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] - E_{s_t\sim \pi}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] = (1-(1-\alpha)^t)(E_{s_t\sim \widetilde{\pi}\mid n_t>0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] - E_{s_t\sim \pi\mid n_t>0}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]) $$

因此

$$\mid E_{s_t\sim \widetilde{\pi}}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] - E_{s_t\sim \pi}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]\mid \leq 2(1-(1-\alpha)^t)\epsilon$$

其中， $$\epsilon = \max_s\mid \overline{A}^{\pi,\widetilde{\pi}}(s)\mid$$。

进而可得：

$$\begin{align}\mid\eta(\widetilde{\pi}) - L_{\pi}(\widetilde{\pi})\mid &= \mid E_{s_t\sim \widetilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^t\overline{A}^{\pi,\widetilde{\pi}}(s_t)\right] - E_{s_t\sim \pi}\left[\sum_{t=0}^{\infty}\gamma^t\overline{A}^{\pi,\widetilde{\pi}}(s_t)\right]\mid \\
							&= \sum_{t=0}^{\infty}\gamma^t\mid E_{s_t\sim \widetilde{\pi}}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right] - E_{s_t\sim \pi}\left[ \overline{A}^{\pi,\widetilde{\pi}}(s_t) \right]\mid \\
							&\leq 2\epsilon\sum_{t=0}^{\infty}\gamma^t(1-(1-\alpha)^t) \\
							&= \frac{2\epsilon\gamma\alpha}{(1-\gamma)(1-\gamma(1-\alpha))} \\
							&\leq \frac{2\epsilon\gamma\alpha}{(1-\gamma)^2}
\end{align}$$

令$$KL_{max}(\pi,\widetilde{\pi})=\max_sKL(\pi(s,\cdot)\|\widetilde{\pi}(s,\cdot))$$，易得$$\alpha\leq KL_{max}(\pi,\widetilde{\pi})$$，于是有：

$$\eta(\widetilde{\pi}) \geq L_{\pi}(\widetilde{\pi}) - C\times KL_{max}(\pi,\widetilde{\pi})$$

其中$$C = \frac{2\epsilon\gamma}{(1-\gamma)^2}$$。

我们再令$$M_i(\pi) = L_{\pi_i}(\pi) - C \times KL_{max}(\pi_i,\pi)$$为第$i$轮迭代得到的函数，通过推导可得：

$$\eta(\pi_{i+1})\geq L_{\pi_i}(\pi_{i+1}) - C \times KL_{max}(\pi_i,\pi_{i+1}) = M_i(\pi_{i+1})$$

$$\eta(\pi_i) = L_{\pi_i}(\pi_i) =  L_{\pi_i}(\pi_i) - C \times KL_{max}(\pi_i,\pi_i) =  M_i(\pi_i)$$

根据上面两个公式可以得到：

$$\eta(\pi_{i+1}) - \eta(\pi_i) \geq M_i(\pi_{i+1}) - M_i(\pi_i)$$

只要我们取$$\pi_{i+1} = arg\max_{\pi}M_i(\pi)$$，那么上面不等式右边就是一个非负值，这样我们就能确保策略对应的期望价值非负上升。

3）优化模型

经过前面的推导，我们已经知道算法的目标函数为：

$$maxmize_{\pi} \left[L_{\pi_{old}}(\pi) - C\times KL_{max}(\pi_{old},\pi)\right]$$

在实践中，这个优化模型还有很多问题，我们将一一解决。

该优化目标中包含KL散度。虽然在理论上可以得到比较好的效果，但是在实际上更新过程过于保守，每一轮迭代更新的步长都偏小，导致优化的速度很慢。为了解决这个问题，我们将优化模型形式进行转变，得到如下有约束的优化问题：

$$\begin{align} & maxmize_{\pi}  L_{\pi_{old}}(\pi) \\
			s.t.\ & KL_{max}(\pi_{old},\pi) \leq \delta
\end{align}$$

当问题变成这个形式后，原问题中较复杂的部分都被转移到了约束中。我们需要对KL散度的上界进行约束，这实际上相当于对所有状态的KL散度进行约束，这样的约束条件多而复杂。为了简化运算，我们将上界变为均值，虽然约束条件有所放宽，但这样可以降低计算的难度，而且从实践效果看这样不会造成效果过度下降。我们令$$\overline{KL}_{\rho}(\pi_1,\pi_2) = E_{s\sim\rho}\left[KL(\pi_1(s,\cdot)\| \pi_2(s,\cdot))\right]$$，于是可将问题进一步转变成如下形式：

$$\begin{align} & maxmize_{\pi} L_{\pi_{old}}(\pi) =  \eta(\pi_{old}) + \sum_s\rho_{\pi_{old}}(s)\sum_a\pi(s,a)A^{\pi_{old}}(s,a)\\
			s.t.\ & \overline{KL}_{\rho_{\pi_{old}}}(\pi_{old},\pi) \leq \delta
\end{align}$$

目标函数的第二项可以写作如下形式：

$$\sum_s\rho_{\pi_{old}}(s)E_{a\sim\pi(s)}\left[A^{\pi_{old}}(s,a)\right]$$

如果我们采用蒙特卡洛法对动作进行采样，就需要事先知道新策略的形式，这对优化造成了阻碍。我们可以采用重要性采样方法来进行规避：

$$\sum_a\pi(s,a)A^{\pi_{old}}(s,a) = \sum_a \pi_{old}(s,a)\frac{\pi(s,a)}{\pi_{old}(s,a)}A^{\pi_{old}}(s,a) = E_{a\sim\pi_{old}(s)}\left[\frac{\pi(s,a)}{\pi_{old}(s,a)}A^{\pi_{old}}(s,a)\right]$$


4）自然梯度法求解

我们可以进一步将问题转化为可以用自然梯度法求解的形式。

回顾一下，自然梯度法模型的标准形式为

$$\begin{align} &\min_{\Delta\boldsymbol{\theta}} \ell(\boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}) \Delta\boldsymbol{\theta} \\
    s.t.  \ 	& \frac{1}{2}\Delta\boldsymbol{\theta}^T\boldsymbol{I}_{f_{\boldsymbol{\theta}}}\Delta\boldsymbol{\theta} < \epsilon
\end{align}$$

其中$$\boldsymbol{I}_{f_{\boldsymbol{\theta}}} = E_{\boldsymbol{x}\sim f_{\boldsymbol{\theta}}}\left[\nabla_{\boldsymbol{\theta}}\log f_\boldsymbol{\theta}(\boldsymbol{x})\nabla_{\boldsymbol{\theta}}\log f_\boldsymbol{\theta}(\boldsymbol{x})^T\right]$$ 为Fisher信息矩阵（Fisher Information Matrix）。

令策略$\pi$的参数为$$\boldsymbol{\theta}$$，对目标函数进行一阶泰勒展开即可得到：

$$L_{\pi_{old}}(\pi) = L_{\pi_{old}}(\pi; \boldsymbol{\theta}_{old}+\Delta\boldsymbol{\theta}) \simeq L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old}) + \nabla_{\boldsymbol{\theta}}L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old}) \Delta\boldsymbol{\theta}$$

由于约束条件为：

$$E_{s\sim\rho_{\pi_{old}}}\left[KL(\pi_{old}(s,\cdot)\| \pi(s,\cdot))\right] \leq \delta $$

可类似地通过采样转化为：

$$\frac{1}{N} \sum_{i=1}^N\frac{1}{2}\Delta\boldsymbol{\theta}^T\boldsymbol{I}_{\pi_{old}(s^{(i)})}\Delta\boldsymbol{\theta}\leq \delta$$

于是，我们得到了优化策略参数的模型：

$$\begin{align}& maxmize_{\Delta\boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}}L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old}) \Delta\boldsymbol{\theta} \\
	& \frac{1}{N} \sum_{i=1}^N\frac{1}{2}\Delta\boldsymbol{\theta}^T\boldsymbol{I}_{\pi_{old}(s^{(i)})}\Delta\boldsymbol{\theta}\leq \delta
\end{align}$$

当我们完成更新量的计算，就可以将其添加到原来的策略参数上，完成一轮优化迭代：

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}_{old} + \Delta \boldsymbol{\theta}$$

如果采用拉格朗日法求解，可得：

$$\Delta \boldsymbol{\theta} = \frac{\alpha}{N}\sum_{i=1}^N \boldsymbol{I}_{\pi_{old}(s^{(i)})}^{-1}\nabla_{\boldsymbol{\theta}}L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old})$$

如果直接采用上述公式进行计算，就需要计算Fisher信息矩阵的逆矩阵，而逆矩阵的计算量比较大，直接计算会降低模型的训练速度，因此我们需要寻找一种方法来减少这部分的计算量。TRPO算法采用了共轭梯度法来求解更新量，从而避免了Fisher信息矩阵的逆矩阵计算。

将问题转化为无约束的二次优化问题：

$$minimize_{\Delta\boldsymbol{\theta}} \frac{1}{N}\sum_{i=1}^N\frac{1}{2}\Delta\boldsymbol{\theta}^T\boldsymbol{I}_{\pi_{old}(s^{(i)})}\Delta\boldsymbol{\theta} - \nabla_{\boldsymbol{\theta}}L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old}) \Delta\boldsymbol{\theta}$$

于是可得$$\boldsymbol{Q} = \frac{1}{N}\sum_{i=1}^N\boldsymbol{I}_{\pi_{old}(s^{(i)})}$$，$$\boldsymbol{b} = [\nabla_{\boldsymbol{\theta}}L_{\pi_{old}}(\pi_{old}; \boldsymbol{\theta}_{old})]^T$$。从而可以不断得到下一个共轭方向。

在计算更新步长时，我们应该考虑原约束条件，其相当于给我们划定了一个置信区域（Trust Region），以保证我们的优化满足策略提升的要求。前面假设我们用共轭梯度法求出了更新方向$$\boldsymbol{d}$$，现在令最大步长为$\beta$，也就是说参数更新的最大值为$$\beta\boldsymbol{d}$$，于是可得到：

$$ \frac{1}{N} \sum_{i=1}^N\frac{1}{2}\beta^2\boldsymbol{d}^T\boldsymbol{I}_{\pi_{old}(s^{(i)})}\boldsymbol{d} = \delta$$

将公式整理后得到：

$$\beta = \sqrt{\frac{2\delta}{ \frac{1}{N} \sum_{i=1}^N\boldsymbol{d}^T\boldsymbol{I}_{\pi_{old}(s^{(i)})}\boldsymbol{d}}}$$

这样我们就得到了满足约束条件的最大步长，然后可以采用backtrack的线搜索方法找到满足优化条件的合适步长。具体方法为：先尝试以$\beta$为步长的情况下，策略提升是否可以满足，如果已经满足则步长选择结束；如果无法满足，则将步长减少一半再进行测试，直到满足为止。




### 2.5.2 PPO 与 DPPO

### 2.5.3 ACER

### 2.5.4 DPG 与 DDPG

DPG（Deterministic Policy Gradient）算法采用off-policy的学习方式，行动策略采用随机策略（一般使用$\epsilon$-贪心策略），保证充足的探索，而评估策略采用确定型策略，可以有效减少需要采集的数据点。该算法尤其适合动作空间连续的情况，策略网络输入为状态$s$，输出直接为动作$a$。

前面介绍的Actor-Critic算法，目标函数导数要求输入的状态$s$必须是与策略相关的，因此只能用on-policy的学习方式。当采用确定型策略时，我们可以得到：

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} V^{\pi_{\boldsymbol{\theta}}}(s), \forall s \in S\\
								&= \nabla_{\boldsymbol{\theta}} Q^{\pi_{\boldsymbol{\theta}}}(s, \pi_{\boldsymbol{\theta}}(s)) \\
								&= \nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s)\nabla_a Q^{\pi_{\boldsymbol{\theta}}}(s,a)\mid_{ a=\pi_{\boldsymbol{\theta}}(s)} (链式法则)\\
\end{align}$$

我们可以用一个$Q$网络去拟合上式中$$Q^{\pi_{\boldsymbol{\theta}}}$$函数，输入为状态$s$和连续动作$a$，输出为状态-动作值。这样，我们就可以使用off-policy的方法来进行计算了。

设行动策略表示为$$\beta_{\boldsymbol{\theta}}$$，评估策略表示为$$\mu_{\boldsymbol{\theta}}$$，用于拟合的值函数模型为$$Q_{\boldsymbol{w}}(s,a)$$，则DPG算法的具体流程如下：

1）初始化：
设定参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$；

设定初始状态$s$；

2）迭代：

基于$s$，根据策略$\beta_{\boldsymbol{\theta}}$生成一个动作$a$

基于$s$和$a$，观测奖励$r$和下一状态$s'$；

根据策略$\mu_{\boldsymbol{\theta}}$生成下一动作$a'$；

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}(r+\gamma Q_{\boldsymbol{w}}(s',a')-Q_{\boldsymbol{w}}(s,a))\nabla_{\boldsymbol{w}}Q_{\boldsymbol{w}}(s,a)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}\nabla_{\boldsymbol{\theta}}\mu_{\boldsymbol{\theta}}(s)\nabla_{\widetilde{a}} Q_{\boldsymbol{w}}(s,\widetilde{a})\mid_{ \widetilde{a}=a'}$$

更新当前状态：$$s \leftarrow s'$$。

前面曾经提到，DQN的局限性是处理连续动作空间，因为求解贪心动作$$arg\max_{\widetilde{a} \in A_s}Q_{\boldsymbol{w}}(s,\widetilde{a})$$非常困难，而DPG中不需要根据$Q$函数去求解贪心动作，因此可以将DQN的思想应用到DPG中来，既能弥补其处理连续动作空间的局限，又能发挥其降低样本关联性的优势。DDPG（Deep Deterministic Policy Gradient）算法正是将DQN中的两大利器Experience Replay 和 Fixed Q-targets应用到了DPG中，在该算法中采用深度神经网络逼近值函数$$Q_{\boldsymbol{w}}(s,a)$$和确定性策略$$\mu_{\boldsymbol{\theta}}(s)$$。

DDPG的伪代码实现如下：

01：初始化critic网络$$Q(s,a\mid \boldsymbol{w})$$和actor网络$$\mu(s\mid \boldsymbol{\theta})$$；

02：用上面的两个网络初始化对应的目标网络$$Q'\leftarrow Q$$，$$\mu'\leftarrow \mu$$

03：初始化Replay Buffer：$R$

04：for episode = 1,...,M do

05：&emsp; $$s_1 = env.reset()$$；

06：&emsp; for t = 1,...,T do

07：&emsp; &emsp; $$a_t = \mu(s_t\mid \boldsymbol{\theta}) + \mathcal{N}_t$$（其中$$\mathcal{N}_t$$为行动策略的随机探索因子）

08：&emsp; &emsp; $$s_{t+1},r_t,terminate, = env.setp(a_t)$$；

09：&emsp; &emsp; $$R.save((s_t,a_t,r_t,s_{t+1}))$$；

10：&emsp; &emsp; 从$R$中随机采样$N$个样本$$(s_i,a_i,r_i,s_{i+1})$$，并令$$y_i = r_i + \gamma Q'(s_{i+1},\mu'(s_{i+1}\mid\boldsymbol{\theta}')\mid\boldsymbol{w}')$$；

11：&emsp; &emsp; 根据critic loss来更新critic网络：$$L = \frac{1}{N}\sum_i(y_i - Q(s_i,a_i\mid \boldsymbol{w}))^2$$；

12：&emsp; &emsp; 根据梯度下降法来更新actor网络：$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) =\frac{1}{N}\sum_i \nabla_{\boldsymbol{\theta}}\mu(s_i\mid\boldsymbol{\theta})\nabla_{a} Q(s_i,a\mid\boldsymbol{w})\mid_{a=\mu(s_i\mid\boldsymbol{\theta})}$$；

13：&emsp; &emsp; 更新目标网络：$$\boldsymbol{w}' \leftarrow \tau\boldsymbol{w} + (1-\tau)\boldsymbol{w}'$$，$$\boldsymbol{\theta}' \leftarrow \tau\boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}'$$；

14：&emsp; end for

15：end for





# 3、Model-based 算法

前面我们面对环境未知的问题时采用了比较“被动”的方法：既然环境的状态转移是未知的，我们就不去关注它。但其实我们可以通过对状态转移概率进行建模，使我们可以对强化学习过程进行整体优化，这称为Model-based 算法。

