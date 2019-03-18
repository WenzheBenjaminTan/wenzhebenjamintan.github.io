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

我们定义$$\rho_{\pi_{\boldsymbol{\theta}}}(x) = \sum_{k=0}^{\infty}\gamma^k P(s\rightarrow x,k,\pi_{\boldsymbol{\theta}})$$，其表示按照策略$$\pi_{\boldsymbol{\theta}}$$从起始状态$s$到状态$x$的总可能性，并且根据步数进行了折扣加权，称作状态$x$的加权访问概率。

策略梯度定理可以表示为如下：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

它直观地给出了性能梯度与策略梯度之间的关系。

## 2.4 行动者-评论家（Actor-Critic）算法

根据策略梯度定理，可以进而推导如下：

$$\begin{align}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= \sum_s\sum_{k=0}^{\infty} P(s_0\rightarrow s,k,\pi_{\boldsymbol{\theta}})\sum_a \gamma^k Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) \\
					&= E_{\pi_{\boldsymbol{\theta}}}\left(\sum_{a}\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^t\sum_{a}\pi_{\boldsymbol{\theta}}(S_t,a)Q^{\pi_{\boldsymbol{\theta}}}(S_t,a)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,a)}{\pi_{\boldsymbol{\theta}}(S_t,a)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tQ^{\pi_{\boldsymbol{\theta}}}(S_t,A_t)\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tR_t\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tR_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right) \\
\end{align}$$


所以策略梯度法的更新公式可以写为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t R_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$


但这种方法的方差也很大，因为其更新的幅度依赖于某episode中$t$时刻到结束时刻的真实样本回报$R_t$。所以更常见的一种做法也是引入一个基准（baseline）$b(s)$，且可以满足：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) =\sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) =\sum_{s}\rho_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}\left(Q^{\pi_{\boldsymbol{\theta}}}(s,a)-b(s)\right)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

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


## 2.5 A3C

真正将Actor-Critic应用到实际中并得到优异效果的是A3C（Asynchronous Advantage Actor-Critic）算法，从算法的名字可以看出，算法突出了异步和优势两个概念。

由于Actor-Critic算法是on-policy的（每一次模型更新都需要“新样本”），为了更快地收集样本，我们需要用并行的方法来收集。在A3C方法中，我们要同时启动$N$个线程，每个线程中有一个Agent与环境进行交互。收集完样本后，每一个线程将独立完成训练并得到参数更新量，并异步地更新到全局的模型参数中。下一次训练时，线程的模型参数和全局参数完成同步，再使用新的参数进行新的一轮训练。

前面提到Actor-Critic算法中使用TD-Error的形式$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来估计$R_t$，这个方法虽然增加了学习的稳定性（即减小了方差），但是学习的偏差也相应变大，为了更好地平衡偏差和方差，A3C方法使用$n$步回报估计法，这个方法可以在训练早期更快地提升价值模型。对应的优势函数估计公式变为：

$$\sum_{i=0}^{n-1}\gamma^ir_{t+i} + \gamma^{n}V_{\boldsymbol{w}}(s_{t+n}) - V_{\boldsymbol{w}}(s_t)$$

最后，为了增加模型的探索性，模型的目标函数中加入了策略的熵。由于熵可以衡量概率分布的不确定性，所以我们希望模型的熵尽可能大一些，这样模型就可以拥有更好的多样性。这样，完整的策略梯度更新公式就变为：

$$\begin{align}
\boldsymbol{\theta}' &= \boldsymbol{\theta} + \alpha\gamma^t (\sum_{i=0}^{n-1}\gamma^ir_{t+i} + \gamma^{n}V_{\boldsymbol{w}}(s_{t+n}) - V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t) + \beta \nabla_{\boldsymbol{\theta}} Ent(\pi_{\boldsymbol{\theta}}(s_t)) \\
\end{align}$$

其中$\beta$为策略的熵在目标中的权重。

价值模型的目标函数与DQN类似，不再赘述。于是，A3C的完整算法如下：

定义：全局时间钟为$T$；全局的价值和策略参数分别为$\boldsymbol{w}$和$\boldsymbol{\theta}$，线程内部的价值和策略参数分别为$\boldsymbol{w'}$和$\boldsymbol{\theta'}$。

1：初始化线程时间钟 $t\leftarrow 0$；

2：repeat

3：&emsp; 将梯度清零：$d\boldsymbol{w} \leftarrow 0$, $d\boldsymbol{\theta} \leftarrow 0$；

4：&emsp; 同步模型参数：$d\boldsymbol{w'} \leftarrow \boldsymbol{w}$, $d\boldsymbol{\theta'} \leftarrow \boldsymbol{\theta}$；

5：&emsp; $$t_{start} = t$$；

6：&emsp; 获取当前状态$$s_t$$；

7：&emsp; repeat

8：&emsp; &emsp; 根据当前策略$$\pi_{\boldsymbol{\theta'}}(s_t,a_t)$$执行$$a_t$$；

9：&emsp; &emsp; $t\leftarrow t+1$；

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

## 2.6 其他策略梯度法

策略梯度法有两个软肋：

1）波动性：REINFORCE算法通过采样一条或几条轨迹后来估计回报，估计波动就很大；为了减少方差对算法的影响，Actor-Critic算法增加了一个模型用于估计当前状态的价值，通过引入一定的偏差换取方差的降低；而A3C算法只是通过多步回报估计法平衡了方差和偏差。总的来说，波动性的问题依然存在。

2）样本利用率：这个问题是所有on-policy算法都要解决的。因为on-policy算法在每一次策略发生改变时，都要丢弃前面产生的样本，这将带来很大的样本浪费，我们需要考虑用off-policy的算法来进行学习。

TRPO和PPO算法主要用于解决第一个问题，而ACER和DPG算法主要用于解决第二个问题。


### 2.6.1 TRPO

TRPO是置信区域策略优化（Trust Region Policy Optimization）算法的简称，它可以确保策略模型在优化时单调提升。其主要思路是找到一种衡量策略之间优劣的计算方法，并以此为目标最大化新策略与旧策略相比的优势。

1) 策略差距的推导

我们先定义基于某个策略的期望价值：

$$\eta(\pi) = E_{s_0,a_0,...\sim \pi}\left[\sum_{t=0}^{\infty}\gamma^tr_{s_t,a_t}(s_{t+1})\right]$$

其中$$s_0\sim P(s_0\rightarrow s_0,0,\pi)$$，$$a_t\sim\pi(s_t,a_t)$$，$$s_{t+1}\sim p_{s_t,a_t}(s_{t+1})$$。

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

3）自然梯度法求解




### PPO



### ACER

### DPG

# 3、Model-based 算法

前面我们面对环境未知的问题时采用了比较“被动”的方法：既然环境的状态转移是未知的，我们就不去关注它。但其实我们可以通过对状态转移概率进行建模，使我们可以对强化学习过程进行整体优化，这称为Model-based 算法。

