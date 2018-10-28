---
layout: post
title: "聚类" 
---

# 1、混合密度模型

混合密度（mixture density）记作

$$p(\boldsymbol{x}) = \sum_{i=1}^k p(\boldsymbol{x}\mid \mathcal{G}_i)p(\mathcal{G}_i)$$

其中$$\mathcal{G}_i$$是混合分支（mixture component），也称为分组（group）或簇（cluster）。$$p(\boldsymbol{x}\mid \mathcal{G}_i)$$是支密度（component density），而$$p(\mathcal{G}_i)$$是混合比例（mixture proportion）。分支数$k$是超级参数，应当预先指定。

给定实例集和$k$，聚类任务要做的是：第一，估计给定实例所属的分支标号；第二，估计各支密度和混合比例。






