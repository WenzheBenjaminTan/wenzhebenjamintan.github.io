---
layout: post
title: "深度强化学习" 
---

# 1、值函数拟合（value function approximation）

当状态空间或动作空间无穷大，甚至是连续的时，原来传统的强化学习方法不再有效。这个时候可以直接针对值函数进行学习。

## 1.1 Q函数拟合

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

更新策略$\pi(s) = arg\max_{a'' \in A_s}Q_{\boldsymbol{w}}(s,a'')$；

更新当前状态：$s=s'$。

因为在函数拟合中使用的训练样本并不满足独立同分布特性，因此该算法不一定能保证收敛。

## 1.2 V函数拟合

类似地，我们可以采用$V_{\boldsymbol{w}}(s)$（其中$\boldsymbol{w}$为参数向量）来拟合V函数，学习参数$\boldsymbol{w}$的更新过程如下：

$$
\begin{align}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t + \alpha (r_t + \gamma V_{\boldsymbol{w}_t}(s_{t+1}) -V_{\boldsymbol{w}_t}(s_t))\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}_t}(s_t)
\end{align}
$$

## 1.3 Memory Replay

Memory Replay机制是深度强化学习启蒙时期最重要的技巧之一，因为它的引入将深度强化学习推进了一大步。最早提出的深度强化学习是一个深度Q网络结构模型（Deep Q-Network，DQN）。因为训练样本相互关联（当利用每条episode顺序训练时），训练过程中策略会进行剧烈的振荡，从而使收敛速度十分缓慢。该问题严重影响了深度学习在强化学习中的应用。

Memory Replay的引入主要起如下作用：

1）打破可能陷入局部最优的可能；

2）模拟监督学习；

3）打破数据之间的关联性。

正是Memory Replay具有这样的作用才使得深度学习算法被顺利地应用在了强化学习领域。Memory Replay的具体操作步骤如下：

1）在算法执行前首先开辟一个Memory空间$D$；

2）利用Meomory空间$D$来采样样本，在每个时间步$t$，将Agent与环境交互得到的转移样本$e^{(t)}=(s_t,a_t,r_t,s_{t+1})$存储到Memory空间$$D=\{e^{(1)},e^{(2)},...,e^{(t)},...\}$$，当达到空间最大值后替换原来的采样样本；

3）从$D$中随机抽取一个批量（batch）的转移样本；

4）使用随机选取的样本，根据贝尔曼方程估算得到这些样本中存在的$(s,a)$对应的Q值；

5）通过该估值和当前网络的估计值的差量来更新网络模型的参数。

这种随机采样的方式，大大降低了样本之间的关联性，使得一个强化学习的问题变成了一个类似于监督学习的问题。

# 2、策略梯度法（policy gradient）

策略梯度法是另一种深度强化学习方法，它将策略看作一个基于策略参数的概率函数，通过不断计算策略期望总奖励关于策略参数的梯度来更新策略参数，最终收敛于最优策略。策略梯度法对于状态空间或者动作空间特别大甚至是连续的情况尤其有效。

策略可以表示为$\pi_{\boldsymbol{\theta}}(s,a)$，即在策略参数$\boldsymbol{\theta}$下，根据当前状态$s$选择动作$a$的概率。

我们定义强化学习的目标函数$J(\boldsymbol{\theta})$为在策略$\pi_{\boldsymbol{\theta}}$下的回报期望。学习过程可以表示为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

可见整个更新过程就是一个梯度上升法。

## 2.1 从似然概率角度推导策略梯度

用$\tau$表示一组随机状态-动作轨迹$$ < S_0,A_0,S_1,A_1,...,S_{T-1},A_{T-1},S_T  > $$。并令$$G(\tau) = \sum_{t=0}^{T-1}\gamma^tR_{s_t,a_t}(s_{t+1})$$表示轨迹$\tau$的回报，$$P(\tau\mid\boldsymbol{\theta})$$表示在$$\pi_{\boldsymbol{\theta}}$$作用下轨迹$\tau$出现的似然概率。则强化学习的目标函数可以表示为：

$$J(\boldsymbol{\theta}) = E\left[\sum_{t=0}^{T-1}\gamma^tR_{s_t,a_t}(s_{t+1})\mid\pi_{\boldsymbol{\theta}}\right] = \sum_{\tau} G(\tau)P(\tau\mid\boldsymbol{\theta})$$

因此，对目标函数求导可得：

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \sum_{\tau} G(\tau)P(\tau\mid\boldsymbol{\theta})\\
								&= \sum_{\tau} G(\tau) \nabla_{\boldsymbol{\theta}}P(\tau\mid\boldsymbol{\theta}) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) G(\tau) \frac{\nabla_{\boldsymbol{\theta}}P(\tau\mid\boldsymbol{\theta})}{P(\tau\mid\boldsymbol{\theta})} \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) G(\tau) \nabla_{\boldsymbol{\theta}}\log P(\tau\mid\boldsymbol{\theta}) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) G(\tau) \nabla_{\boldsymbol{\theta}}\log \left(\prod_{t=0}^{T-1}\pi_{\boldsymbol{\theta}}(s_t,a_t)P_{s_t,a_t}(s_{t+1})\right) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) G(\tau) \nabla_{\boldsymbol{\theta}} \left(\sum_{t=0}^{T-1}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)+ \sum_{t=0}^{T-1}\log P_{S_t,A_t}(S_{t+1})\right) \\
								&= \sum_{\tau}P(\tau\mid\boldsymbol{\theta}) G(\tau) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})}\left[G(\tau) \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\

								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^t R_t \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\


\end{align}$$

因此，性能梯度可以用策略梯度来表示，但该估计的方差非常大。

注意到对$$t' < t$$，有

$$E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\gamma^{t'} R_{t'} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\gamma^{t'} R_{t'} \right]E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = 0 $$

因此

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^t R_t \sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\sum_{t'=t}^{T-1}\gamma^{t'} R_{t'} \right] \\
								&= E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}\gamma^tG_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \right] \label{eq15} \\

\end{align}$$

有的时候也可以把$\gamma^t$省略掉，进行下面的估计替代：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[\sum_{t=0}^{T-1}G_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t) \right]$$


## 2.2 REINFORCE算法


当利用当前策略$$\pi_{\boldsymbol{\theta}}$$采样$m$条轨迹后，可以利用$m$条轨迹的经验平均逼近目标函数的导数：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{m}\sum_{i=1}^m \sum_{t=0}^{T-1} \gamma^t G_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

上式估计的方差受$$G_t^{(i)}$$的影响仍然会很大。我们可以在回报中引入常数基线$b$以减小方差且保持期望不变。

因为有

$$E_{\tau\sim P(\tau\mid\boldsymbol{\theta})} \left[b \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right] = 0 $$

所以容易得到修改后的估计：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{m}\sum_{i=1}^m \sum_{t=0}^{T-1} (\gamma^t G_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

我们进而可求使得估计方差最小时的基线$b$。

令$$X = \sum_{t=0}^{T-1} (\gamma^t G_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t^{(i)},A_t^{(i)})$$，则方差为：

$$Var(X) = E(X-\overline{X})^2 = EX^2 - \overline{X}^2$$

其中$$\overline{X} = EX$$与$b$无关，令$$Var(X)$$对$b$的导数为0，即

$$\frac{\partial Var(X)}{\partial b} = E\left[X\frac{\partial X}{\partial b}\right] = 0$$

求解可得：

$$b= \frac{\sum_{i=1}^m \sum_{t=0}^{T-1}\gamma^t G_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})  
}{\sum_{i=1}^m \left(\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\right)^2}$$

于是可以得到基本的REINFORCE算法（policy gradient based reinforcement learning）如下：

1）初始化参数$\boldsymbol{\theta}$和步长$\alpha > 0$；

2）迭代：

根据策略$\pi_{\boldsymbol{\theta}}$生成$m$个采样片段$ < s_0^{(i)},a_0^{(i)},r_0^{(i)},s_1^{(i)},a_1^{(i)},r_1^{(i)},...,s_{T-1}^{(i)},a_{T-1}^{(i)},r_{T-1}^{(i)},s_T^{(i)} > i=1,2,...,m$，然后计算常数$b$:

$$b= \frac{\sum_{i=1}^m \sum_{t=0}^{T-1}\gamma^t G_t^{(i)} \nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})}{\sum_{i=1}^m \left(\sum_{t=0}^{T-1}\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})\right)^2}$$

并更新$\theta$:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \frac{\alpha}{m} \sum_{i=1}^m \sum_{t=0}^{T-1} (\gamma^t G_t^{(i)} - b)\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t^{(i)},a_t^{(i)})
$$

显然，基本的REINFORCE算法需要得到每个完整的episode再更新$\boldsymbol{\theta}$，因此它是off-line的。



## 2.3 策略梯度定理

实际上，我们还可以利用贝尔曼方程来对性能梯度和策略梯度之间的关系进行推导：

$$\begin{align}\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} V^{\pi_{\boldsymbol{\theta}}}(s), \forall s \in S \\
								&= \nabla_{\boldsymbol{\theta}} \left[\sum_a \pi_{\boldsymbol{\theta}}(s,a) Q^{\pi_{\boldsymbol{\theta}}}(s,a) \right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a)\nabla_{\boldsymbol{\theta}}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a)\nabla_{\boldsymbol{\theta}} \sum_{s'}P_{s,a}(s')(R_{s,a}(s') +\gamma V^{\pi_{\boldsymbol{\theta}}}(s'))\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a) \sum_{s'}\gamma P_{s,a}(s')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s')\right] \\
								&= \sum_a \left[ Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \pi_{\boldsymbol{\theta}}(s,a) \sum_{s'}\gamma P_{s,a}(s') \sum_{a'} \left[ Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \pi_{\boldsymbol{\theta}}(s',a') \sum_{s''}\gamma P_{s',a'}(s'')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s'')\right] \right] \\
								&= \sum_a  Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) + \gamma\sum_a\pi_{\boldsymbol{\theta}}(s,a) \sum_{s'} P_{s,a}(s') \sum_{a'} Q^{\pi_{\boldsymbol{\theta}}}(s',a')\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s',a') + \gamma^2\sum_a\pi_{\boldsymbol{\theta}}(s,a) \sum_{s'} P_{s,a}(s') \sum_{a'} \pi_{\boldsymbol{\theta}}(s',a') \sum_{s''} P_{s',a'}(s'')\nabla_{\boldsymbol{\theta}}V^{\pi_{\boldsymbol{\theta}}}(s'') \\

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
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tG_t\frac{\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(S_t,A_t)}{\pi_{\boldsymbol{\theta}}(S_t,A_t)}\right) \\
					&=E_{\pi_{\boldsymbol{\theta}}}\left(\gamma^tG_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(S_t,A_t)\right) \\
\end{align}$$


所以策略梯度法的更新公式可以写为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t G_t\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$


但这种方法的方差也很大，因为其更新的幅度依赖于某episode中$t$时刻到结束时刻的真实样本回报$G_t$。收敛速度也慢，如果$$G_t$$总是大于0，会使得所有行动的概率密度都向正的方向“拉拢”。所以更常见的一种做法也是引入一个基准（baseline）$b(s)$，且可以满足：

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) =\sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}Q^{\pi_{\boldsymbol{\theta}}}(s,a)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) =\sum_{s}d_{\pi_{\boldsymbol{\theta}}}(s)\sum_{a}\left(Q^{\pi_{\boldsymbol{\theta}}}(s,a)-b(s)\right)\nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a)$$

$b(s)$可以取任何常数或函数，只要不和$a$相关就不影响上式的结果。因为：

$$\sum_a b(s) \nabla_{\boldsymbol{\theta}}\pi_{\boldsymbol{\theta}}(s,a) = b(s) \nabla_{\boldsymbol{\theta}}\sum_a\pi_{\boldsymbol{\theta}}(s,a) = b(s)\nabla_{\boldsymbol{\theta}}1 = 0$$

于是更新公式修改为：

$$\boldsymbol{\theta}' = \boldsymbol{\theta} + \alpha\gamma^t (G_t-b(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t)$$

至于$b(s)$怎么设计，取决于算法，但一般的做法是取$b(s) = V_{\boldsymbol{w}}(s)$，也就是说用另一个函数来估计状态均值。容易得到，这种情况下参数的更新主要取决于在状态$s_t$下执行动作$a_t$所得总奖励相对于状态均值的优势，如果有优势，则更新后的参数会增加执行该动作的概率；如果没有优势，则更新后的参数会减少执行该动作的概率。

此外，为了避免off-line地求得全部回报$$G_t$$，我们采用单步的奖励和下个状态估值的和式$$r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})$$来估计$G_t$（注意，这个估计是有偏的），于是参数更新公式变为：

$$\begin{align}
\boldsymbol{\theta}' &= \boldsymbol{\theta} + \alpha\gamma^t (r_{t}+\gamma V_{\boldsymbol{w}}(s_{t+1})-V_{\boldsymbol{w}}(s_t))\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s_t,a_t) \\
\end{align}$$

这就得到了Actor-Critic算法，它是on-line的，采用时间差分的方式来不断更新两个模型，一个是策略模型（Actor），一个是价值模型（Critic）。

Actor-Critic算法的具体流程如下：

1）初始化参数$\boldsymbol{w}$和$\boldsymbol{\theta}$，步长$\alpha^{\boldsymbol{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0$，当前状态$s=s_0$，梯度乘子$I=1$；

2）迭代：

基于当前状态$s$，根据策略$\pi_{\boldsymbol{\theta}}$生成一个动作$a$并得到奖励$r$和下一状态$s'$，并执行如下步骤：

$$\delta \leftarrow r+\gamma V_{\boldsymbol{w}}(s')-V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla_{\boldsymbol{w}}V_{\boldsymbol{w}}(s)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\delta\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(s,a)$$

$$I \leftarrow \gamma I$$

$$s \leftarrow s'$$



# 3、蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）

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


# 4、AlphaGo算法框架

AlphaGo完整的学习系统主要由以下四个部分组成：

1）策略网络（policy network）。又分为监督学习的策略网络和强化学习的策略网络。策略网络的作用是根据当前的棋局来预测和采样下一步走棋。

2）滚轮策略（rollout policy）。也是用于预测和采样下一步走棋，但是预测速度是策略网络的1500倍。

3）估值网络（value network）。用于估计当前棋局的价值。

4）MCTS。将策略网络、滚轮策略和估值网络融合进策略搜索的过程中，形成一个完整的走棋系统。

策略网络、滚轮策略的输入是当前棋局，输出是下步每一种走棋的概率；而估值网络的输入是当前棋局，输出是当前棋局的价值。

具体实施分为四个阶段：

在第一阶段，使用策略网络$p_{\boldsymbol{\sigma}}$来直接对来自人类专家下棋的样本数据进行学习。$p_{\boldsymbol{\sigma}}$是一个13层的深度卷积网络，具体的训练方式为梯度下降法：

$$\Delta\boldsymbol{\sigma} \propto \frac{\partial \log p_{\boldsymbol{\sigma}}(a\mid s)}{\partial\boldsymbol{\sigma}}$$

在测试集上使用所有输入特征进行训练，预测人类专家走子动作的准确率为57.0%。同时，还使用局部特征训练了一个可以迅速走子采样的滚轮策略$p_{\boldsymbol{\pi}}$，预测人类专家走子动作的准确率为24.2%。$p_{\boldsymbol{\sigma}}$每下一步棋是3毫秒，而$p_{\boldsymbol{\pi}}$是2微秒。

在第二阶段，使用$p_{\boldsymbol{\sigma}}$作为输入，通过强化学习的策略梯度方法来训练策略网络$p_{\boldsymbol{\rho}}$：

$$\Delta\boldsymbol{\rho} \propto \frac{\partial \log p_{\boldsymbol{\rho}}(a\mid s)}{\partial\boldsymbol{\rho}}z$$

其中$z$表示一局棋最终所获的收益，胜为+1，负为-1，平0。

具体训练方式是：随机选择之前迭代轮的策略网络和当前的策略网络进行对弈，并利用策略梯度法（该方法属于REINFORCE算法）来更新参数，最终得到增强的策略网络$p_{\boldsymbol{\rho}}$。$p_{\boldsymbol{\rho}}$与$p_{\boldsymbol{\sigma}}$在结构上是完全相同的。增强后的$p_{\boldsymbol{\rho}}$与$p_{\boldsymbol{\sigma}}$对抗时胜率超过了80%。

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
