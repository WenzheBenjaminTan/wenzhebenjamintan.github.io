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

因为在函数拟合中使用的训练样本并不满足独立同分布特性，因此该算法不一定能保证收敛。

### 1.1.2 DQN

深度Q网络结构模型（Deep Q-Network，DQN）可谓是深度强化学习的开山之作，由DeepMind在NIPS 2013上发表，后又在Nature 2015上提出改进版本。

用人工神经网络来拟合Q函数有两种形式：一种形式是将状态和动作当作输入，然后通过人工神经网络分析后得到动作的Q值，这对于动作空间是连续的时尤其有效；而当动作空间是离散的时，可以采用另一种更直观的形式，即只输入状态，然后输出所有的动作Q值。这相当于先接收外部的信息，然后通过大脑加工输出每种动作的值，最后选择拥有最大值的动作当作下一步要做的动作。DQN就是采用第二种形式来拟合Q函数的。

但是采用“半梯度下降法”来训练人工神经网络，因为训练样本相互关联（当利用每条episode顺序训练时），训练过程中策略会进行剧烈的振荡，从而使收敛速度十分缓慢。该问题严重影响了深度学习在强化学习中的应用。为了降低样本关联性，DQN采用了两大利器：Experience Replay 和 Fixed Q-targes。


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

$$y^{DQN} = r + \gamma Q(s',arg\max_{a'} Q(s',a';\boldsymbol{\theta}^-); \boldsymbol{\theta}^-))$$

也就是说根据状态$s'$选择动作$a'$的过程，以及估计$Q(s',a')$使用的是同一张$Q$值表，或者说是使用的同一个网络参数，这可能会导致过高的估计值（overestimate）。而Double DQN就是用来解决这种过估计的，它的想法是引入另一个神经网络来减小这种误差。而DQN中本来就有两个神经网络，我们刚好可以利用这一点，改用MainNet来选择动作$a'$，而继续用TargetNet来估计$Q(s',a')$。于是采用Double DQN的Q-target计算公式变为：

$$y^{DoubleDQN} = r + \gamma Q(s',arg\max_{a'} Q(s',a';\boldsymbol{\theta}); \boldsymbol{\theta}^-))$$

其中$$\boldsymbol{\theta}^-$$是TargetNet的参数，$$\boldsymbol{\theta}$$是MainNet的参数。


## 1.2 V函数拟合

类似地，我们可以采用$V_{\boldsymbol{w}}(s)$（其中$\boldsymbol{w}$为参数向量）来拟合$$V^{\pi}$$函数，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t + \alpha (r_t + \gamma V_{\boldsymbol{w}_t}(s_{t+1}) -V_{\boldsymbol{w}_t}(s_t))\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}_t}(s_t)
\end{align}
$$

# 2、策略梯度法（policy gradient）

策略梯度法是另一种深度强化学习方法，它将策略看作一个基于策略参数的概率函数，通过不断计算策略期望总奖励关于策略参数的梯度来更新策略参数，最终收敛于最优策略。策略梯度法对于状态空间或者动作空间特别大甚至是连续的情况尤其有效。

策略可以表示为$\pi_{\boldsymbol{\theta}}(s,a)$，即在策略参数$\boldsymbol{\theta}$下，根据当前状态$s$选择动作$a$的概率。

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
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \nabla_{\boldsymbol{\theta}}\log \left(\prod_{t=0}^{T-1}\pi_{\boldsymbol{\theta}}(S_t,A_t)p_{S_t,A_t}(S_{t+1})\right) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) R(\tau) \nabla_{\boldsymbol{\theta}} \left(\sum_{t=0}^{T-1}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)+ \sum_{t=0}^{T-1}\log p_{S_t,A_t}(S_{t+1})\right) \\
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

有的时候也可以把$\gamma^t$省略掉，进行下面的估计替代：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}R_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \right]$$


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

我们定义$$d_{\pi_{\boldsymbol{\theta}}}(x) = \sum_{k=0}^{\infty}\gamma^k P(s\rightarrow x,k,\pi_{\boldsymbol{\theta}})$$，其表示按照策略$$\pi_{\boldsymbol{\theta}}$$从起始状态$s$到状态$x$的总可能性，并且根据步数进行了折扣加权。

策略梯度定理可以表示为如下：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

它直观地给出了性能梯度与策略梯度之间的关系。

## 2.4 行动者-评论家（Actor-Critic）算法

根据策略梯度定理，可以进而推导如下：

$$\begin{align}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= \sum_s\sum_{k=0}^{\infty} P(s_0\rightarrow s,k,\pi_{\boldsymbol{\theta}})\sum_a \gamma^k Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= E_{\pi_{\boldsymbol{\theta}}}\left(\sum_{a}\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^t\sum_{a}\pi_{\boldsymbol{\theta}}(S_t,a)Q^{\pi_{\boldsymbol{\theta}}}(S_t,a)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)}{\pi_{\boldsymbol{\theta}}(S_t,a)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,A_t)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tR_t\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tR_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right) \\
\end{align}$$


所以策略梯度法的更新公式可以写为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t R_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$


但这种方法的方差也很大，因为其更新的幅度依赖于某episode中$t$时刻到结束时刻的真实样本回报$R_t$。收敛速度也慢，如果$$R_t$$总是大于0，会使得所有行动的概率密度都向正的方向“拉拢”。所以更常见的一种做法也是引入一个基准（baseline）$b(s)$，且可以满足：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) =\sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) =\sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}\left(Q^{\pi_{\boldsymbol{\theta}}}(s,a)-b(s)\right)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

$b(s)$可以取任何常数或函数，只要不和$a$相关就不影响上式的结果。因为：

$$\sum_a b(s) \nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) = b(s) \nabla_{\boldsymbol{\theta}}\sum_a\pi_{\boldsymbol{\theta}}(s,a) = b(s)\nabla_{\boldsymbol{\theta}}1 = 0$$

于是更新公式修改为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t (R_t-b(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$

至于$b(s)$怎么设计，取决于算法，但一般的做法是取$b(s) = V_{\boldsymbol{w}}(s)$，也就是说用另一个函数来估计状态均值。容易得到，这种情况下参数的更新主要取决于在状态$s_t$下执行动作$a_t$所得总奖励相对于状态均值的优势，如果有优势，则更新后的参数会增加执行该动作的概率；如果没有优势，则更新后的参数会减少执行该动作的概率。实际上，此时$$R_t - b(s_t)$$是对动作优势函数$$A^{\pi_{\boldsymbol{\theta}}}(s_t,a_t) = Q^{\pi_{\boldsymbol{\theta}}}(s_t,a_t) - V^{\pi_{\boldsymbol{\theta}}}(s_t)$$的估计，因为$$R_t$$可以视为$$Q^{\pi_{\boldsymbol{\theta}}}(s_t,a_t)$$的估计，$$b(s_t) = V_{\boldsymbol{w}}(s_t)$$可以视为$$V^{\pi_{\boldsymbol{\theta}}}(s_t)$$的估计。

此外，为了避免off-line地求得全部回报$$R_t$$，我们采用单步的奖励和下个状态估值的和式$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来估计$R_t$（注意，这个估计是有偏的），于是参数更新公式变为：

$$\begin{align}
\boldsymbol{\theta}' &= \boldsymbol{\theta} + \alpha\gamma^t (r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})-V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t) \\
\end{align}$$

这就得到了Actor-Critic算法，它是on-line和on-policy的，采用时间差分的方式来不断更新两个模型，一个是策略模型（Actor），一个是价值模型（Critic）。

因为根据贝尔曼方程可得到：

$$V^{\pi}(s) = E\left[r_{s,\pi(s)}(s') + \gamma V^{\pi}(s')\right]$$

所以价值模型可以使用$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$与$$V_{\boldsymbol{w}}(s_t)$$之间的均方误差（MSE）作为损失函数。

Actor-Critic算法的具体流程如下：

1）初始化参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$，当前状态$s=s_0$，梯度乘子$I=1$；

2）迭代：

基于当前状态$s$，根据策略$\pi_{\boldsymbol{\theta}}$生成一个动作$a$并得到奖励$r$和下一状态$s'$，并执行如下步骤：

$$\delta \leftarrow r+\gamma V_{\boldsymbol{w}}(s')-V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\delta\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s,a)$$

$$I \leftarrow \gamma I$$

$$s \leftarrow s'$$

其中$\delta$是优势函数估计。


## A3C

## A2C

## TRPO

## PPO

## DPPO

## DPG

## DDPG



