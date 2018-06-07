---
layout: post
title: "生产调度模型" 
---
# 1、调度的概念

调度是一种进行系列决策的过程，即在一段时间内如何将资源分配给要执行的任务，使得某个或多个目标达到最优。

几乎所有调度问题都可以用生产调度模型来表示。在生产调度模型中，资源被表示为机器（machine）,要执行的任务被表示为工作（job）。由于资源的稀缺性，机器的数量是有限的。而我们在考虑的调度问题中，需要被调度的工作也被假设成有限的。调度的目标正是要优化执行这些工作的指标性能，如完成时间、超时率等。

如果跟工作执行相关的参数（如加工时间、提交日期等）是不确定的，则该调度问题称为随机型调度问题，否则称为确定型调度问题。


# 2、确定型调度模型

在生产调度模型中，机器的数量记作 $m$，工作的数量记作 $n$。通常，下标 $i$ 指一台机器，而下标 $j$ 指一项工作。如果一项工作要求多个加工环节，则数对 $(i,j)$ 指的是工作 $j$ 在机器 $i$ 上的加工环节。

与工作 $j$ 相关的参数主要有：

$p_{ij}$：加工时间（processing time），表示工作 $j$ 在机器 $i$ 上需要的处理时间。用 $p_j$ 表示工作 $j$ 的总工作量，$v_{ij}$ 表示工作 $j$ 在机器 $i$ 上的处理速度，则工作 $j$ 在机器 $i$ 上的处理时间可以表示为 $p_{ij} = p_j / v_{ij}$。如果机器 $i$ 的加工速度独立于工作 $j$，则可以省略下标 $j$。如果所有机器的加工速度都相同，加工时间也可以直接用 $p_j$ 表示；

$r_j$：提交日期（release date），指工作 $j$ 可以开始加工的最早时间（到达系统的时间）；

$d_j$：工期（due date），指工作 $j$ 承诺的完成时间。一项工作可以在工期之后完成，但超时会受到惩罚。如果该工期必须满足，则称为最后期限（deadline），表示为 $\bar{d_j}$；

$w_j$：权重（weight），表示工作 $j$ 相对于其他工作的重要性，一般来说，$\sum\limits_{j}w_j = 1$。

一个调度问题可以表示为如下形式：

$$\alpha | \beta | \gamma$$

其中：

$\alpha$ 域描述机器环境，且只包含单一实例；

$\beta$ 域描述加工约束，可以为空或者包含一个甚至多个实例；

$\gamma$ 域描述优化目标，常常包含一个实例，也可以包含多个实例。

## 2.1 机器环境

在 $\alpha$ 域中可规定的机器环境实例有：

1) $\textbf{Single Machine} (1)$: The case of single machine is the simplest of all possible machine environments and is a special case of all other more complicated machine environments.

2) $\textbf{Identical Machines in Parallel} (Pm)$: There are $m$ identical machines in parallel. Job $j$ requires a single operation and may be processed on any one of the $m$ machines or any one that belongs to a given subset. If job $j$ cannot be processed on just any machine, but only on any one belonging to a specific subset $M_j$, then the entry $M_j$ appears in the $\beta$ field.

3) $\textbf{Machines in Parallel with Different Speeds} (Qm)$: There are $m$ machines in parallel with different speeds. The speed of machine $i$ is denoted by $v_i$. The time $p_{ij}$ that job $j$ spends on machine $i$ is equal to $p_j/v_i$ (assuming job $j$ receives all its processing from machine $i$). If all machines have the same speed, i.e., $v_i = 1$ for all $i$ and $p_{ij} = p_j$, then the environment is identical to $Pm$.

4) $\textbf{Job Related Machines in Parallel} (Rm)$: There are $m$ different machines in parallel. Machine $i$ can process job $j$ at speed $v_{ij}$ (related to job $j$). The time $p_{ij}$ that job $j$ spends on machine $i$ is equal to $p_j/v_{ij}$ (again assuming job $j$ receives all its processing from machine $i$). If the speeds of the machines are independent of the jobs, i.e., $v_{ij} = v_i$ for all $i$ and $j$, then the environment is identical to $Qm$.

5) $\textbf{Flow Shop} (Fm)$: There are $m$ machines in series. Each job has to be processed on each one of the $m$ machines. All jobs have to follow the same route, i.e., they have to be prcessed first on machine 1, then on machine 2, and so on. After completion on one machine a job joins the queue at the next machine. Usually, all queues are assumed to operate under the First In First Out (FIFO) discipline, that is, a job cannot "pass" another while waiting in a queue. If the FIFO displine is in effect the flow shop is referred to as a permutation flow shop and the $\beta$ field includes the entry $prmu$.

6) $\textbf{Flexible Flow Shop} (FFc)$: A flexible flow shop is a generalization of the flow shop and the parallel machine environments. Instead of $m$ machines in series there are $c$ stages in series with at each stage a number of machines in parallel. Each job has to be processed first at stage 1, then at stage 2, and so on. At each stage job $j$ requires processing on only one machine and any machine can do. The queues between the various stages may or may not operate according to the FIFO discipline.

7) $\textbf{Job Shop} (Jm)$: In a job shop with $m$ machines each job has its own predetermined route to follow. A distinction is made between job shops in which each job visits each machine at most once and job shops in which a job may visit each machine more than once. In the latter case the $\beta$ field contains the entry $rcrc$ for recirculation.

8) $\textbf{Flexible Job Shop} (FJc)$: A flexible job shop is a generalization of the job shop and the parallel machine environments. In stead of $m$ machines in series there are $c$ work centers at each work center a number of machines in parallel. Each job has its own route to follow through the shop; job $j$ requires processing at each work center on only one machine and any machine can do. If a job on its route through the shop may visit a work center more than once, then the $\beta$ field contains the entry $rcrc$ for recirculation.

9) $\textbf{Open Shop} (Om)$: There are $m$ machines. Each job has to be processed on each one of the $m$ machines. However, some of these processing times may be zero. There are no restrictions with regard to the routing of each job through the machine environment. The scheduler is allowed to determine a route for each job and different jobs may have different routes.

## 2.2 加工约束

$\beta$ 域中的加工约束可能包含的实例有：

1) $\textbf{Release Dates} (r_j)$: If this symbol appears in the $\beta$ field, then job $j$ cannot start its processing before its release data $r_j$. If $r_j$ does not appear in the $\beta$ field, the processing of job $j$ may start at any time. In contract to release dates, due dates $d_j$ are not specified in this field. The type of objective function gives sufficient indication whether or not there are due dates.

2) $\textbf{Preemptions} (prmp)$: Preemptions imply that it is not necessary to keep a job on a machine, once started, until its completion. The scheduler is allowed to interrupt the processing of a job (preempt) at any point in time and put a different job on the machine instead. The amount of processing a preempted job already has received is not lost. When a preempted job is afterwards put back on the machine (or on another machine in the case of parallel machines), it only needs the machine for its remaining processing time. When preemptions are allowed, $prmp$ is included in the $\beta$ field; when $prmp$ is not included, preemptions are not allowed.

3) $\textbf{Sequence Dependent Setup Times} (s_{jk})$: The $s_{jk}$ represents the sequence dependent setup time that is incurred between the processing of jobs $j$ and $k$; $s_{0k}$ denotes the setup time for job $k$ if job $k$ is first in the sequence and $s_{j0}$ the clean-up time after job $j$ if job $j$ is last in the sequence (of cource, $s_{0k}$ and $s_{j0}$ may be zero). If the setup time between jobs $j$ and $k$ depends on the machine, then the subscript $i$ is included, i.e., $s_{ijk}$. If no $s_jk$ appears in the $\beta$ field, all setup times are assumed to be 0 or sequence independent, in which case they are simply included in the processing times.

4) $\textbf{Job Families} (fmls)$: In this case, the $n$ jobs belong to different job families. Jobs from the same family may have different processing times, but they can be prcessed on a machine one after another without requiring any setup in between. However, if the machine switches over from one family to another, say from family $g$ to family $h$, then a setup is required. If this setup time depends on both families $g$ and $h$ and is sequence dependent, then it is denoted by $s_{gh}$. If this setup time depends only on the family about to start, i.e., family $h$, then it is denoted by $s_h$. If it does not depend on either family, it is denoted by $s$.

5) $\textbf{Batch Processing} (batch(b))$: A machine may be able to process a number of jobs, say $b$, simultaneously; that is, it can process a batch of up to $b$ jobs at the same time. The processing times of the jobs in a batch may not be all the same and the entire batch is finished only when the last job of the batch has been completed, implying that the completion time of the entire batch is determined by the job with the longest processing time. If $b = 1$, then the problem reduces to a conventional scheduling environment. Another special case that is of interest is $b = \infty$, i.e., there is no limit on the number of jobs the machine can handle at any time.

6) $\textbf{Breakdowns} (brkdwn)$: Machine breakdowns imply that a machine may not be continuously available. The periods that a machine is not available are usually assumed to be fixed (e.g., due to shifts or scheduled maintenance). If there are a number of identical machines in parallel, the number of machines available at any point in time is a function of time, i.e., $m(t)$. Machine breakdowns are at times also referred to as machine availability constraints.

7) $\textbf{Precedence Constraints} (prec)$: Precedence constraints may appear in a single machine or in a parallel machine environment, requiring that one or more jobs may have to be completed before another job is allowed to start its processing. There are several special forms of precedence constraints: if each job has at most one predecessor and at most one successor, the constraints are referred to as chains. If each job has at most one successor, the constraints are referred to as an intree. If each job has at most one predecessor the constraints are referred to as an outtree. If no $prec$ appears in the $\beta$ field, the jobs are not subject to precedence constraints.

8) $\textbf{Machine Eligibility Restrictions} (M_j)$: The $M_j$ symbol may appear in the $\beta$ field when the machine environment is $m$ machines in parallel. When the $M_j$ is present, not all $m$ machines are capable of processing job $j$. Set $M_j$ denotes the set of machines that can process job $j$. If the $\beta$ field does not contain $M_j$, job $j$ may be processed on any one of the $m$ machines.

9) $\textbf{Permutation} (prmu)$: A constraint that may appear in the flow shop environment is that the queues in front of each machine operate according to the FIFO discipline. This implies that the order (or permutation) in which the jobs go through the first machine is mantained throughout the system.

10) $\textbf{Blocking} (block)$: Blocking is a phenomenon that may occur in flow shops. If a flow shop has a limied buffer in between two successive machines, then it may happen that when the buffer is full the upstream machine is not allowed to release a completed job. Blocking implies that the completed job has to remain on the upstream machine preventing (i.e., blocking) that machine from working on the next job. In the models with blocking, it is usually assumed that the machines operate according to FIFO. That is, $block$ implies $prmu$.

11) $\textbf{No-wait} (nwt)$: The no-wait requirement is another phenomenon that may occur in flow shops. Jobs are not allowed to wait between two successive machines. This implies that the starting time of a job at the first machine has to be delayed to ensure that the job can go through the flow shop without having to wait for any machine. An example of such an operation is a steel rolling mill in which a slab of steel is not allowed to wait as it would cool off during a wait. It is clear that under no-wait the machines also operate according to the FIFO disipline.

12) $\textbf{Recirculation} (rcrc)$: Recirculation may occur in a job shop or flexible job shop when a job may visit a machine or work center more than once.

Any other entry that may appear in the $\beta$ field is self explanatory. For example, $p_j = p$ implies that all processing volumes are equal and $d_j = d$ implies that all due dates are equal. 

## 2.3 优化目标

生产调度模型中的优化目标一般都是最小化所有工作完成时间的函数。工作 $j$ 在机器 $i$ 上的完成时间记作 $C_{ij}$，则工作 $j$ 的完成时间为 $C_j = \max\limits_{i} C_{ij}$，显然地，它与调度方案有关。

优化目标也可能是工期的函数。

工作 $j$ 的延迟（lateness）定义为：

$$L_j = C_j - d_j$$

当工作滞后完成的时候为正，而当工作提前完成的时候为负。

工作 $j$ 的滞后（tardness）定义为：

$$T_j = \max(C_j - d_j, 0) = \max(L_j, 0)$$

滞后和延迟的区别在于滞后永远不会是负的。

工作 $j$ 的单位惩罚（unit penalty）定义为：

$$
U_j = \begin{cases} 
	1 & C_j > d_j \\
	0 & otherwise 
\end{cases}
$$

延迟、滞后和单位惩罚是3个基本的与工期有关的惩罚函数。

常见最小化目标函数的实例有：

1）制造期（$C_{max}$）：制造期定义为$\max(C_1,C_2,...,C_n)$，即最后一项离开系统的工作完成时间。制造期越短说明机器的利用率越高。

2）最大延迟（$L_{max}$）：最大延迟定义为$\max(L_1,L_2,...,L_n)$，它度量违反工期的最差情况。

3）加权完成时间和（$\sum w_jC_j$）：加权完成时间和用来度量由调度方案所带来的总加工或库存成本。

4）带折扣的加权完成时间和（$\sum w_j(1 - e^{-rC_j})$）：这是一个比前一个更一般的成本函数，成本每过单位时间以 $r$ 的比率打折（$0 < r < 1$）。也就是说，如果工作 $j$ 没有在时间 $t$ 完成，则它在时间段 $[t, t+dt]$ 里产生的附加成本为 $w_jre^{rt}dt$。 如果工作 $j$ 在时间 $t$ 完成，则在时间段 $[0, t]$ 内产生的总成本是 $w_j(1 - e^{-rt})$。$r$ 的值通常接近于0，如0.1。

5）加权滞后和（$\sum w_jT_j$）：这也是比加权完成时间和更一般化的成本函数。

6）加权滞后工作数量（$\sum w_jU_j$）：这是一个在日常调度中常用的指标。

以上列出的所有目标函数都是 $C_1,C_2,...,C_n$ 的非减函数，称为规则绩效指标。

还有一些非规则的绩效指标，也常常被用作目标函数。比如，当工作 $j$ 具有工期 $d_j$ 时，他可能受制于一个提前惩罚，这里工作 $j$ 的提前（earliness）定义为

$$E_j = \max (d_j-C_j, 0)$$

目标函数可以表示为

$$\sum (w'_jE_j + w_jT_j)$$

与工作 $j$ 提前惩罚相关的权重 $w'_j$ 可以与其滞后惩罚相关的权重 $w_j$ 不一样。

## 2.4 调度方案

在调度问题中，可以区分调度方案和调度策略两个概念。调度方案是针对确定型模型而言的，它是指给定调度模型中的所有决策的集合，即在什么时间做什么样的决策。而调度策略是针对随机型模型而所的，调度策略为系统可能处在的任何一种状态规定了合适的行为。因此，在确定型模型中，我们仅用调度方案来表示调度问题的解。

非延迟的调度方案：如果在一个可行的调度方案中，任何有工作需要等待加工的时候均没有可用的机器是空闲的，则称这个调度方案是非延迟的。

换句话说，一个可行调度方案是非延迟的等价于系统中没有非强迫的空闲，即只要机器有活可干，就让它马上干活。

然而，对于一些无中断的调度模型，存在一些非强迫空闲反而是有利的，因为它可以让资源得到更有效的利用。因此在无中断的调度模型中，还可以定义两类调度方案：

半活跃的调度方案：如果在一个可行的无中断调度方案中，不改变任何一台机器上的加工顺序就没有一个工作可以提前完成，则称这个调度方案是半活跃的。

换句话说，一个可行的无中断调度方案是半活跃的等价于没有工作可以在保证系统可行性的情况下被推移至其所在机器的前面时间中。

活跃的调度方案：如果在一个可行的无中断半活跃调度方案中，不可能通过改变在某机器上执行工作的顺序来建立另一个调度，使得在没有一个工作推迟完成的情况下至少有一个工作可以提前完成，则称这个调度方案是活跃的。

换句话说，一个可行的无中断半活跃调度方案是活跃的等价于没有工作可以在保证系统可行性的情况下被插入其所在机器前面时间的空隙中。显然，无中断、非延迟的调度必然是活跃的，但反之则不然。而最优的无中断调度方案必然是活跃的，但不一定是非延迟的。

## 2.5 应用举例

1）$1 \| s_{jk} \| C_{max}$ 等价于旅行商问题（Travelling Salesman Problem）

一个旅行商从城市0开始出发，访问城市 $1,2,...,n$ 并返回到城市0，在此过程中使旅行的总距离最小。从城市0到城市 $k$ 的距离是 $s_{0k}$；从城市 $j$ 到城市 $k$ 的距离是 $s_{jk}$；从城市 $j$ 到城市0的距离是 $s_{j0}$。因此，该问题中确定 $n$ 个工作的执行顺序等价于旅行商问题中访问 $n$ 个城市的顺序。

2）$1 \| p_j=1 \| \sum h_j(C_j)$ 等价于指派问题（Assignment Problem）

在该问题中，机器一共有 $n$ 个位置（可以认为是指派问题中要执行的任务），$n$ 个工作（可以认为是指派问题中的待指派对象）要被分配到这 $n$ 个位置上，当将工作 $j$ 分配给位置 $k$ 时产生的成本为 $h_j(k)$（即指派问题中 $c_{jk} = h_j(k)$）。

3) $Q_m \| p_j=1 \| \sum h_j(C_j)$ 等价于运输问题（Transportation Problem）

设机器 $i$ 的加工速度为 $v_i$。当工作 $j$ 被安排为机器 $i$ 上处理的第 $k$ 项工作时，设变量 $x_{ijk}$ 为1，其余情况为0。这样变量 $x_{ijk}$ 就被联系到一个运输活动上。$n$ 项工作等价于运输问题中的 $n$ 个出发地，每个出发地的发货量为1。可以被分配的 $n \times m$ 个位置（每个机器最多可以被分配 $n$ 项工作）等价于运输问题中的 $n \times m$ 个目的地，每个目的地最多的收货量为1。操作该运输活动的单位成本为

$$c_{ijk} = h_j(C_j) = h_j(k/v_i)$$

因此，可以将该运输问题表示为如下形式：

$$
\begin{align}
\min\ 	& \sum_{i=1}^{m}\sum_{j=1}^{n}\sum_{k=1}^{n}c_{ijk}x_{ijk} \\
s.t.\ 	& \sum_{i}\sum_{k}x_{ijk} = 1 \ & j=1,2,...,n \\
	& \sum_{j}x_{ijk} \leq 1 \ & i=1,2,...,m; k=1,2,...,n \\
	& x_{ijk} \geq 0 \ & i=1,2,...,m;j=1,2,...,n;k=1,2,...,n 
\end{align}
$$

该问题同样是一个加权二部匹配问题（Weighted Bipartite Matching Problem）。

4) $R_m \|\| \sum C_j$ 等价于加权二部匹配问题（Weighted Bipartite Matching Problem）

该问题中可以将 $(i,k)$ 位置表示在机器 $i$ 上处理的并且有 $k-1$ 项工作在其之后进行的工作。此时可以认为连接工作 $j$ 和位置 $(i,k)$ 的弧具有权重 $kp_{ij}$ （$p_{ij}$ 在目标函数中总共会被计算 $k$ 次），即 $c_{ijk} = kp_{ij}$。


# 3、随机型调度模型

## 3.1 符号表示
在随机型调度模型中，我们一般假定加工时间、提交日期和工期的分布都是事先已知的，即在时间的起始点0就已知；随机加工时间的实现只有当该工作阶段结束时才称为已知；提交日期和工期的实现也只有在实际发生的时刻才成为已知。在模型描述中，随机变量一般以大写字母表示，而已实现的已知值以小写字母表示。

与工作 $j$ 相关的参数主要有：

$X_{ij}$：工作 $j$ 在机器 $i$ 上的随机加工时间；

$1/\lambda_{ij}$ 随机变量 $X_{ij}$ 的平均值或期望值；

$R_j$：工作 $j$ 的随机提交日期；

$D_j$：工作 $j$ 的随机工期；

$w_j$：工作 $j$ 的权重。

之所以用 $X_{ij}$ 来表示随机加工时间，是因为 $P$ 通常用来表示概率。与确定型模型中相似，权重 $w_j$ 基本上等于保持工作 $j$ 在系统中的每单位时间所产生的成本。在排队论模型中，一般用 $c_j$ 来表示工作 $j$ 的权重或成本，在这里 $w_j$ 和 $c_j$ 是等同的。

## 3.2 调度策略

在随机调度过程中，会逐渐得到新的信息，比如前面工作的完成情况、提交日期和工期的随机发生等。根据对这些新信息的利用情况可将调度策略分为静态策略和动态策略。

## 3.2.1 静态策略

在静态策略中，决策者在0时刻即规定好了所有工作的优先级（包括非0时刻提交的工作），这个优先级在加工过程中不会发生变化。对于不可中断的情况，每次有机器空闲的时候选择可行且优先级最高的工作进行执行。而对于可中断的情况，在任何时间点上在可加工工作集合顶端的工作都是正在机器上加工的工作。

静态策略实现起来比较简单，但是对于复杂调度效果不好。

## 3.2.2 动态策略

在动态策略中，当不可中断时，每次机器空闲后，决策者可以决定哪一项工作进行加工，甚至等待一段时间。决策者的每次决策都是基于当前可获得的所有信息（比如当前的时间、正在等待加工的工作、正在别的机器上进行加工的工作、这些工作在机器上已经接受加工的时间等）。但是，决策者不允许中断，只要一项工作开始加工，就不允许停止。

当可以中断时，决策者可以在任何时间决定哪项工作在机器上进行加工。当然，他在任何决策时刻都要考虑所有当时可获得的信息并且可能会引起中断的发生。
