---
layout: post
title: "随机理论基础" 
---
# 1、调度的概念

调度是一种进行系列决策的过程，即在一段时间内如何将资源分配给要执行的任务，使得某个或多个目标达到最优。

几乎所有调度问题都可以用生产调度模型来表示。在生产调度模型中，资源被表示为机器（machine）,要执行的任务被表示为工作（job）。由于资源的稀缺性，机器的数量是有限的。而我们在考虑的调度问题中，需要被调度的工作也被假设成有限的。调度的目标正是要优化执行这些工作的指标性能，如完成时间、超时率等。

如果跟工作执行相关的参数（如加工时间、提交日期、工期等）是不确定的，则该调度问题称为随机型调度问题，否则称为确定型调度问题。


# 2、确定型调度模型

在生产调度模型中，机器的数量记作 $m$，工作的数量记作 $n$。通常，下标 $i$ 指一台机器，而下标 $j$ 指一项工作。如果一项工作要求多个加工环节，则数对 $(i,j)$ 指的是工作 $j$ 在机器 $i$ 上的加工环节。

与工作 $j$ 相关的参数主要有：

$p_{ij}$：加工时间（processing time），表示工作 $j$ 在机器 $i$ 上面的处理时间。如果工作 $j$ 的加工时间独立于机器（即可以加工 $j$ 的机器同速），则可以省略下标 $i$；

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

## 2.3 优化目标


# 3、随机型调度模型


