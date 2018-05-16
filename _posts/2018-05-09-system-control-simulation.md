---
layout: post
title: "系统控制和仿真专题" 
---
# 1、系统的形式化描述

一个系统可以通过以下八元组来形式化描述：

$$S=(T,U,\Omega,X,x_0,f,Y,g)$$

其中，

$T$：时间集，描述系统变化的时间坐标；

$U$：输入集，代表外部环境对系统的作用；

$\Omega$：输入段集，描述某个时间段内的输入信息，是 $(U,T)$ 的一个子集；

$X$：内部状态集，表示系统内部状态属性；

$\textbf{x}_0 \in X$：系统的初始状态；

$f:X \times \Omega \rightarrow X$：状态转移函数，定义系统内部状态是如何变化的；

$Y$：输出集，系统通过它作用于环境；

$g:X \times U \times T \rightarrow Y$：输出函数，定义系统如何通过当前状态和输入得到输出。


# 2、系统的状态空间

系统在时间 $t_0$ 的内部状态必须提供所有必要信息，使得根据这些信息和输入 $\textbf{u}(t)(t \geq t_0)$ 足以得到输出 $\textbf{y}(t)(t \geq t_0)$。构成内部状态的分量又叫作状态变量（State Variable），内部状态的集合又叫状态空间（State Space）。


# 3、静态系统与动态系统

如果系统的内部状态 $\textbf{x}$ 是固定不变的，且输出函数 $g$ 与时间无关，则系统为静态系统，否则为动态系统。


# 4、时变动态系统与非时变动态系统

如果动态系统的输出函数 $g$ 是与时间无关的，则该动态系统为非时变动态系统，否则为时变动态系统。

# 5、确定型动态系统与随机型动态系统

如果动态系统的输出 $\textbf{y}(t)$ 至少有一个分量是随机变量，则称该动态系统为随机型动态系统，否则称为确定型动态系统。引起系统的随机性的有可能来自三个方面：随机初始状态 $\textbf{x}_0$、随机输入 $\textbf{u}(t)$ 或随机状态转移函数 $f$。

# 6、时间驱动与事件驱动的动态系统

## 6.1 时间驱动的动态系统

时间驱动的动态系统是指内部状态变量随时间连续变化的系统，可以表示为如下方程：

$$\dot{\textbf{x}}(t) = f(\textbf{x}(t),\textbf{u}(t),t), \textbf{x}(t_0) = \textbf{x}_0$$

$$\textbf{y}(t) = g(\textbf{x}(t),\textbf{u}(t),t)$$

经典的系统控制理论正是基于上述系统数学模型来进行研究的。

对于闭环控制系统，控制函数可以表示为：

$$\textbf{u}(t) = \gamma (\textbf{r}(t), \textbf{x}(t), t)$$

其中 $\textbf{r}(t)$ 表示系统的参考输出，即期望系统的行为。

对于最优控制问题，可以表示为如下：

$$
\begin{align}
\label{1}	\max\limits_{\textbf{u}(t) \in \Omega}\ & \int_{t_0}^{t_1} L(\textbf{x}(t),\textbf{u}(t),t) dt + \varphi (\textbf{x}(t_1), t_1) \\
\label{2}	s.t.\ & \dot{\textbf{x}}(t) = f(\textbf{x}(t), \textbf{u}(t),t), \textbf{x}(t_0) = \textbf{x}_0 \\
\label{3}	& \Psi (\textbf{x}(t_1), t_1) = \textbf{0} 
\end{align}
$$

其表示的意义为：在满足系统方程 \eqref{2} 的约束条件下，在容许的控制域 $\Omega$ 中确定一个最优控制律 $\textbf{u}^*(t)$，使系统状态 $\textbf{x}(t)$ 从已知初始状态 $$\textbf{x}_0$$ 转移到要求的目标集 \eqref{3}，并使性能指标 \eqref{1} 达到极大值。

## 6.2 事件驱动的动态系统

事件驱动的动态系统是指内部状态变量在某些离散的时间点上发生离散变化的系统。这些离散的状态转移被称作“事件”，因此事件驱动的动态系统又被称作离散事件动态系统（Discrete Event Dynamic System, DEDS）。系统仿真理论和随机过程理论正是基于该类动态系统来进行研究的。

### 6.2.1 离散事件动态系统的形式化描述

根据研究的粒度，可以从三个层面来对离散事件动态系统进行形式化描述，分别为：Automaton、Timed Automaton、Stochastic Timed Automaton。

#### 1) Automaton

An automaton is a five-tuple 

$$(X, E, f, \Gamma, x_0)$$

where

$X$: a countable state space;

$E$: a countable event set;

$f:X \times E  \rightarrow X$: a state transition function and is generally a partial function on its domain;

$\Gamma:X \rightarrow 2^E$ the feasible event function $\Gamma(x)$ is the set of all events $e$ for which $f(x,e)$ is defined and it is called the feasible event set; 

$x_0$: the initial state.

The automaton operates as follows. It starts in the initial state $x_0$ and upon the occurence of an event $e \in \Gamma(x_0)$ it will make a transition to state $f(x_0,e) \in X$. This process then continues based on the transitions for which $f$ is defined.

#### 2) Timed Automaton

A timed automaton is a six-tuple

$$(X, E, f, \Gamma, x_0, \textbf{V})$$

where $(X, E, f, \Gamma, x_0)$ is an automaton and $\textbf{V} = \\{\textbf{v}_i: i \in E\\}$ is a clock structure. 

The automaton generates a state sequence $x' = f(x,e')$ driven by an event sequence $\\{(e_1,t_1), (e_2,t_2), \ldots\\}$ generated through

$$e' = arg \min\limits_{i \in \Gamma(x)}y_i$$

with the clock values $y_i, i \in E$, defined by

$$
y'_i = 
\begin{cases} 
y_i - y^* \  \text{if} i \neq e' \text{and} i \in \Gamma(x) \\ 
v_{i,N_i + 1} \ \text{if} i = e' \text{or} i \notin \Gamma(x)
\end{cases}
\
i \in \Gamma(x')
$$

where the interevent time $y^*$ is defined as 

$$y^* = \min\limits_{i \in \Gamma(x)}y_i$$

and the event scores $N_i, i \in E$, are defined by

$$
N'_i = 
\begin{cases}
N_i + 1 \ \text{if} i = e' \text{or} i \notin \Gamma(x) \\
N_i \ \text{otherwise}
\end{cases}
$$

In addition, event times $t$ are updated through

$$t' = t + y^*$$ 

and inital conditions are:

$y_i = v_{i,1}$ and $N_i = 1$ for all $i \in \Gamma(x_0)$;

$y_i$ is undefined and $N_i = 0$ for all $i \notin \Gamma(x_0)$.

#### 3) Stochastic Timed Automaton

In order to avoid notational confusion between random variables and sets (both usually represented by upper-case letters), $\chi$ and $\epsilon$ are used to denote the state space and the event set of the underlying automaton respectively.

In addition, a random variable notation paralleling the one used for the timed automaton are adopted: $X$ is the current state; $E$ is the most recent event (causing the transition into state $X$); $T$ is the most recent event time (corresponding to event E); $N_i$ is the current score of event $i$; $Y_i$ is the current clock value of event $i$.

A stochastic timed automaton is a six-tuple 

$$(\chi, \epsilon, \Gamma, p, p_0, G)$$

where

$\chi$: a countable state space;

$\epsilon$: a countable event set;

$\Gamma(x)$: a set of feasible events, defined for all $x \in \chi$ with $\Gamma(x) \subseteq \epsilon$;

$p(x' \| x,e')$: a state transition probability, defined for all $x,x' \in \chi, e' \in \epsilon$, and such that $p(x' \| x,e') = 0$ for all $e' \notin \Gamma(x)$;

$p_0(x)$: the pmf ($P(X_0 = x), x \in \chi$) of the initial state $X_0$;

$G = \\{G_i: i \in \epsilon\\}$: a stochastic clock structure.

The automaton generates a stochastic state sequence $\\{X_0, X_1, \ldots\\}$ through a transition mechanism (based on observations $X = x, E' = e'$):

$$X' = x' \text{with probability} p(x' \| x,e')$$ 

and it is driben by a stochastic event sequence $\\{(E_1,T_1), (E_2,T_2), \ldots\\$ generated through

$$E' = arg \min\limits_{i \in \Gamma(X)}Y_i$$

with the stochastic clock values $Y_i, i \in \epsilon$, defined by

$$
Y'_i = 
\begin{cases} 
Y_i - Y^* \  \text{if} i \neq E' \text{and} i \in \Gamma(X) \\ 
V_{i,N_i + 1} \ \text{if} i = E' \text{or} i \notin \Gamma(X)
\end{cases}
\
i \in \Gamma(X')
$$

where the interevent time $Y^*$ is defined as 

$$Y^* = \min\limits_{i \in \Gamma(X)}Y_i$$

and the event scores $N_i, i \in \epsilon$, are defined by

$$
N'_i = 
\begin{cases}
N_i + 1 \ \text{if} i = E' \text{or} i \notin \Gamma(X) \\
N_i \ \text{otherwise}
\end{cases}
$$

and

$$\\{V_{i,k} \sim G_i\\}$$

In addition, event times $T$ are updated through

$$T' = T + Y^*$$ 

and inital conditions are:

$X_0 \sim p_0(x)$ and

$Y_i = V_{i,1}$ and $N_i = 1$ for all $i \in \Gamma(X_0)$;

$Y_i$ is undefined and $N_i = 0$ for all $i \notin \Gamma(X_0)$.

### 6.2.2 离散事件仿真

对于离散事件仿真的实现，主要有两种方法：通用的 Event Scheduling Scheme 和主要面向有实体流动的 Process Interaction Scheme。

#### 1) Event Scheduling Scheme

a) Initialization

The INITIALIZE function sets the STOP_CONDITION, the STATE to $x_0$, and the simulation TIME to $0$ (except in unusual circumstances when TIME may be initially set at some positive value). The RANDOM VARIATE GENERATOR provides event lifetimes for all feasible events at the initial state, and the SCHEDULED EVENT LIST ($L$) is initialized, with all entries sorted in increasing order of scheduled times.

b) Scanning SCHEDULED EVENT LIST and advancing TIME

Step 1: Remove the first entry $(e_1,t_1)$ from the SCHEDULED EVENT LIST.
	
Step 2: Update the simulation TIME by advancing it to the new event time $t_1$.

Step 3: Update the STATE according to the state transition function, $x' = f(x,e_1)$.

Step 4: Delete from the SCHEDULED EVENT LIST any entries corresponding to infeasible events in the new state, that is, delete all $(e_k, t_k) \in L$ such that $e_k \notin \Gamma(x')$.
	
Step 5: Add to the SCHEDULED EVENT LIST any feasible event which is not already scheduled (possibly including the triggering event removed in Step 1). The scheduled event time for some such $i$ is given by $(TIME + v_i)$, where TIME was set in Step 2 and $v_i$ is a lifetime obtained from the RANDOM VARIATE GENERATOR.

Step 6: Reorder the updated SCHEDULED EVENT LIST based on a smallest-scheduled-time-first scheme.

The procedure then repeats with Step 1 for the new ordered list until the STOP_CONDITION return true.

During the simulation process, DATA REGISTERS are used for collecting data for estimation purposes. And a REPORT GENERATOR is used for estimating various quantities of interest upon completion of a simulation run.

#### 2) Process Interaction Scheme

A large class of DES consists of resource contention environments, where resources must be shared among many users. In such environments, it is often convenient to think of "entities" as undergoing a PROCESS as they flow through the DES. This PROCESS is a sequence of events separated by time intervals. During such a time interval, an entity is either receiving service at some resource or waiting for service. 

In the process interaction scheme, the behavior of the DES is described through several such processes, one for each type of entity of interest. Each type of entity contains particular attributes and functions. Attributes are information characterizing the entity and its state. Functions are instantaneous actions or time delays experienced by entities.

In general, we present a PROCESS as a sequence of functions of a entity. A function is one of two types:

1. Logic functions: Instantaneous actions taken by the entity that triggers this function in its process. 

2. Time delay functions: The entity is held by that function for some period of time.

There are two types of time delay functions:

a. Condition delay: The delay depends on the STATE of the system.

b. Non-condition delay: The delay is fixed, usually determined by a number obtained by the RANDOM VARIATE GENERATOR. 

The procedure of the process interaction scheme is as follows:

a) Initialization

The INITIALIZE function sets the STOP_CONDITION, the STATE to $x_0$, and the simulation TIME to $0$ (except in unusual circumstances when TIME may be initially set at some positive value). The RANDOM VARIATE GENERATOR provides arrival times for all mobile entities, and the FUTURE EVENT LIST (FEL) is initialized, with all entries sorted in increasing order of activation times (i.e., arrival times).

b) Scanning FEL and advancing TIME

Step 1: Scan all records in FEL;

Step 2: Set simulation TIME = earlist_activation_time;

Step 3: Move all records with activation_time equal to TIME from FEL to Current EVENT LIST (CEL);

c) Scanning CEL

Step 4: Scan all records in CEL in order of their priority levels. If some entity's condition satified, proceed its functions (and update the STATE) as many as possible. If some entity non-conditionally delayed, file a new record with the next activation time for the entity in FEL and remove the old record rom CEL.   

The procedure then repeats with Step 1 for the new FEL until the STOP_CONDITION return true.
