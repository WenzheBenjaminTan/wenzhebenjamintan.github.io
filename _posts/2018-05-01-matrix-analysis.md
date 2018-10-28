---
layout: post
title: "矩阵分析基础" 
---
# 1、向量

## 1.1 向量的定义

We define a column $n$-vector to be an array of $n$ numbers, denoted by

$$\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}$$

The number $a_i$ is called the $i$th componont of the vector $\mathbf{a}$. Denote by $\mathbb{R}$ the set of real numbers and by $\mathbb{R}^n$ the set of solumn $n$-vectors with real components. We call $\mathbb{R}^n$ an $n$-dimensional real vector space. We commonly denote elements of $\mathbb{R}^n$ by lowercase bold letters (e.g., $\mathbf{x}$). The compononts of $\mathbf{x} \in \mathbb{R}^n$ are denoted $x_1,x_2,...,x_n$.

A set of vectors $\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\}$ is said to be linearly independent if the equality

$$\alpha_1\mathbf{a}_1 + \alpha_2\mathbf{a}_2 + \dots + \alpha_k\mathbf{a}_k = \mathbf{0}$$

implies that all coefficients $\alpha_i (i = 1,2,...,k)$ are equal to zero. A set of the vectors $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ is linearly dependent if it is not linearly independent.

Note that the set of composed of the single vector $\mathbf{0}$ is linearly dependent, for if $\alpha \neq 0$, then $\alpha\mathbf{0} = \mathbf{0}$. In fact, any set of vectors containing the vector $\mathbf{0}$ is linearly dependent.

A vector $\mathbf{a}$ is said to be a linear combination of vectors $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$, if there are scalars $\alpha_1, \alpha_2, ... , \alpha_k$ such that

$$\mathbf{a} = \alpha_1\mathbf{a}_1 + \alpha_2\mathbf{a}_2 + \dots + \alpha_k\mathbf{a}_k$$

$\textbf{Theorem: }$ A set of vectors  $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ is linearly dependent if and only if one of the vectors from the set is a linear combination of the remaining vectors.

A subset $\mathcal{V}$ of $\mathbb{R}^n$ is called a subspace of $\mathbb{R}^n$ if $\mathcal{V}$ is closed under the operations of vector addition and scalar multiplication. That is, if $\mathbf{a}$ and $\mathbf{b}$ are vectors in $\mathcal{V}$, then the vectors $\mathbf{a} + \mathbf{b}$ and $\alpha\mathbf{a}$ are also in $\mathcal{V}$ for every scalar $\alpha$. 

Every subspace contains the zero vector $\mathbf{0}$, for if $\mathbf{a}$ is an element of the subspace, so is $(-1)\mathbf{a} = -\mathbf{a}$. Hense, $\mathbf{a} - \mathbf{a} = \mathbf{0}$ also belongs to the subspace.

Let $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ be arbitrary vectors in $\mathbb{R}^n$. The set of all their linear combinations is called the span of $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ and is denoted

$$span[\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k] = \left \{\sum\limits_{i=1}^{k}\alpha_i\mathbf{a}_i \Big| \alpha_1,\alpha_2,...,\alpha_k \in \mathbb{R}\right \} $$

The span of any set of vectors is a subspace.

Given a subspace $\mathcal{V}$, any set of linearly independent vectors $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\} \subset \mathcal{V}$ such that $\mathcal{V} = span[\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k]$ is referred to as a basis of the subspace $\mathcal{V}$. All basis of a subspace $\mathcal{V}$ contain the same number of vectors. This number is called the dimension of $\mathcal{V}$, denoted by $dim \mathcal{V}$

$\textbf{Theorem: }$ If $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ is a basis of $\mathcal{V}$, then any vector $\mathbf{a}$ of $\mathcal{V}$ can be represented uniquely as

$$\mathbf{a} = \alpha_1\mathbf{a}_1 + \alpha_2\mathbf{a}_2 + \dots + \alpha_k\mathbf{a}_k$$

whrere $\alpha_i \in \mathbb{R} (i=1,2,...,k)$.

Suppose that we are given a basis $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$ of $\mathcal{V}$ and  vector $\mathbf{a} \in \mathcal{V}$ such that 

$$\mathbf{a} = \alpha_1\mathbf{a}_1 + \alpha_2\mathbf{a}_2 + \dots + \alpha_k\mathbf{a}_k$$

The coefficients $\alpha_i (i=1,2,...,k)$ are called the coordinates of $\mathbf{a}$ with respect to the basis $\\{\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_k\\}$. 

The natural basis for $\mathbb{R}^n$ is the set of vectors

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix},
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix},
\ldots,
\mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}.
$$

We can similarly define complex vector spaces. For this, let $\mathbb{C}$ denote the set of complex numbers and $\mathbb{C}^n$ the set of column $n$-vectors with complex components. The set $\mathbb{C}^n$ has properties similar to thoes of $\mathbb{R}^n$, where scalars can take complex values.


## 1.2 向量的内积与范数

For $\mathbf{x},\mathbf{y} \in \mathbb{R}^n$, we define the Euclidean inner product by 

$$\mathbf{x} \bullet \mathbf{y} = \sum\limits_{i=1}^n x_iy_i = \mathbf{x}^T\mathbf{y}$$

The inner product is a real-valued function: $\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$ having the following properties:

1) Positiveity: $\mathbf{x} \bullet \mathbf{x} \geq 0$, $\mathbf{x} \bullet \mathbf{x} = 0$ if and only if $\mathbf{x} = \mathbf{0}$.

2) Symmetry: $\mathbf{x} \bullet \mathbf{y} = \mathbf{y} \bullet \mathbf{x}$.

3) Additivity: $(\mathbf{x} + \mathbf{y}) \bullet \mathbf{z} = \mathbf{x} \bullet \mathbf{z} + \mathbf{y} \bullet \mathbf{z}$.

4) Homogeneity: $(r\mathbf{x}) \bullet \mathbf{y} = r(\mathbf{x} \bullet \mathbf{y})$ for every $r \in \mathbb{R}$.

It is possible to define other real-valued functions on $\mathbb{R}^n \times \mathbb{R}^n$ that satisfy properties 1 to 4 above. Many results involving the Euclidean inner product also hold for these other forms of inner products.

The vectors $\mathbf{x}$ and $\mathbf{y}$ are said to be orthogonal if $\mathbf{x} \bullet \mathbf{y} = 0$.

The Euclidean norm of a vector $\mathbf{x}$ is defined as

$$\|\mathbf{x}\| = \sqrt{\mathbf{x} \bullet \mathbf{x}} = \sqrt{\mathbf{x}^T\mathbf{x}}$$

$\textbf{Theorem: Cauchy-Schwarz Inequality. }$ For any two vectors $\mathbf{x}$ and $\mathbf{y}$ in $\mathbb{R}^n$, the Cauchy-Schwarz inequality

$$|\mathbf{x} \bullet \mathbf{y}| \leq \|\mathbf{x}\|\|\mathbf{y}\|$$

holds. Furthermore, equality holds if and only if $\mathbf{x} = \alpha\mathbf{y}$ for some $\alpha \in \mathbb{R}$.

The Euclidean norm of a vector has the following properties:

1) Positivity: $\\|\mathbf{x}\\| \geq 0$, $\\|\mathbf{x}\\| = 0$ if and only if $\mathbf{x} = \mathbf{0}$.

2) Homogeneity: $\\|r\mathbf{x}\\| = \|r\|\\|\mathbf{x}\\|$ for every $r \in \mathbb{R}$.

3) Triangle inequality: $\\|\mathbf{x} + \mathbf{y}\\| \leq \\|\mathbf{x}\\| + \\|\mathbf{y}\\|$, equality holds if and only if $\mathbf{x} = \alpha\mathbf{y}$ for some $\alpha \geq 0$.

The Euclidean norm is an example of a general vector norm, which is any function satisfying the three properties of positivity, homogeneity, and triangle inequality. Other examples of vector norms on $\mathbb{R}^n$ include the 1-norm, defined by 

$$\|\mathbf{x}\|_1 = |x_1| + |x_2| + \dots + |x_n|$$

and the $\infty$-norm, defined by 

$$\|\mathbf{x}\|_\infty = max_i |x_i|$$

where the notation $max_i$ represents the largest over all the possible index value of $i$.

The Euclidean norm is often referred to as the 2-norm, and denoted $\\|\mathbf{x}\\|_2$. 

The norms above are special cases of the $p$-norm, given by

$$\|\mathbf{x}\|_p = \begin{cases} 
(|x_1|^p + |x_2|^p + \dots + |x_n|^p)^{1/p} & \text{if } 1 \leq p < \infty \\
max\{|x_1|,|x_2|,...,|x_n|\} & \text{if } p = \infty
\end{cases}$$

 
# 2、矩阵

## 2.1 矩阵的定义

A matrix is a rectangular array of numbers, commonly denoted by uppercase bold letters (e.g., $\mathbf{A}$). A matrix with $m$ rows and $n$ columns is called an $m \times n$ matrix, and writen as

$$\mathbf{A} = \begin{bmatrix} 	a_{11} & a_{12} & \dots & a_{1n}\\ 
				a_{21} & a_{22} & \dots & a_{2n} \\ 
				\vdots & \vdots & \ddots & \vdots \\ 
				a_{m1} & a_{m2} & \dots & a_{mn} 
		\end{bmatrix}
$$

The real number $a_{ij}$ located in the $i$th row and $j$th column is called the $(i,j)$th entry. We can think of $\mathbf{A}$ in terms of its $n$ columns, each of which is a column vector in $\mathbb{R}^m$. Let us denote the $k$th column of $\mathbf{A}$ by $\mathbf{a}_k$:

$$\mathbf{a}_k = \begin{bmatrix} a_{1k} \\ a_{2k} \\ \vdots \\ a_{mk} \end{bmatrix}$$

The maximal number of linearly independent columns of $\mathbf{A}$ is called the rank of the matrix $\mathbf{A}$, denoted $rank \mathbf{A}$. Note that $rank \mathbf{A}$ is the dimension of $span[\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_n]$.

$\textbf{Theorem: }$ The rank of a matrix $\mathbf{A}$ is invariant under the following operations:

1) Muliplication of the columns of $\mathbf{A}$ by nonzero scalars.

2) Interchange of the columns.

3) Addition to a given column a linear combination of other columns.

A matrix $\mathbf{A}$ is said to be square if the number of its rows is equal to the number of its columns (i.e., it is $n \times n$). Associated with each square matrix $\mathbf{A}$ is a scalar called the derminant of the matrix $\mathbf{A}$, denoted $det \mathbf{A}$ or $\\|\mathbf{A}\\|$. The Leibniz formula for the determinant of an $n \times n$ matrix $\mathbf{A}$ is

$$det \mathbf{A} = \sum_{\sigma \in S_n}\left(sgn(\sigma)\prod_{i=1}^n a_{i,\sigma_i}\right)$$

where the sum is computed over all permutations $\sigma$ of the set $\\{1,2,...,n\\}$. The value in the $i$th position of permuation $\sigma$ is denoted by $\sigma_i$. The set of all such permutations is denoted by $S_n$. For each permuation $\sigma$, $sgn(\sigma)$ denotes the signature of $\sigma$, a value that is +1 whenever the reordering given by $\sigma$ can be achieved by successively interchanging two entries an even number of times, and -1 whenever it can be achieved by an odd number of such interchanges.

The determinant of a square matrix has the following properties:

1) The determinant of the matrix $\mathbf{A} = [\mathbf{a}_1,\mathbf{a}_2,...,\mathbf{a}_n]$ is a linear function of each column; that is,

$$det [\mathbf{a}_1,...,\mathbf{a}_{k-1},\alpha\mathbf{a}^{(1)}_k+\beta\mathbf{a}^{(2)}_k,\mathbf{a}_{k+1},...,\mathbf{a}_n] \\
	= \alpha \cdot det [\mathbf{a}_1,...,\mathbf{a}_{k-1},\mathbf{a}^{(1)}_k,\mathbf{a}_{k+1},...,\mathbf{a}_n] + \beta \cdot det [\mathbf{a}_1,...,\mathbf{a}_{k-1},\mathbf{a}^{(2)}_k,\mathbf{a}_{k+1},...,\mathbf{a}_n] 
$$

for each $\alpha,\beta \in \mathbb{R}, \mathbf{a}^{(1)}_k,\mathbf{a}^{(2)}_k \in \mathbb{R}^n$.

2) If $\mathbf{a}_i=\mathbf{a}_j (i \neq j, i,j \in \\{1,2,...,n\\})$, then $det \mathbf{A} = 0$.

3) Let 

$$\mathbf{I}_n = [\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n]$$

where $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$ is the natural basis for $\mathbb{R}^n$. Then $det \mathbf{I}_n = 1$.  

A $p$th-order minor of an $m \times n$ matrix $\mathbf{A}$, with $p \leq \min\\{m,n\\}$, is the determinant of a $p \times p$ matrix obtained from $\mathbf{A}$ by deleting $m-p$ rows and $n-p$ columns.

We can use minors to investigate the rank of a matrix. In particular, we have the following theorem.

$\textbf{Theorem: }$ If an $m \times n (m \geq n)$ matrix $\mathbf{A}$ has nonzero $n$th-order minor, then the columns of $\mathbf{A}$ are linearly independent; that is, $rank \mathbf{A} = n$.

If a matrix $\mathbf{A}$ has an $r$th-order minor $\\|\mathbf{M}\\|$ with the properties (i) $\\|\mathbf{M}\\| \neq 0$ and (ii) any minor of $\mathbf{A}$ that is formed by adding a row and a column of $\mathbf{A}$ to $\mathbf{M}$ is zero, then

$$rank \mathbf{A} = r$$

Thus, the rank of a matrix is equal to the highest order of its nonzero minor(s).

A nonsingular (or invertible) matrix is a square matrix whose determinant is nonzero. Suppose that $\mathbf{A}$ is an $n \times n$ square matrix. Then, $\mathbf{A}$ is nonsingular if and only if there is another $n \times n$ matrix $\mathbf{B}$ such that

$$\mathbf{AB} = \mathbf{BA} = \mathbf{I}_n$$

We call the matrix $\mathbf{B}$ above the inverse matrix of $\mathbf{A}$, and write $\mathbf{B} = \mathbf{A}^{-1}$.


## 2.2 线性变换

A funcation $\mathcal{L}: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is called a linear transformation if:

1) $\mathcal{L}(a\mathbf{x}) = a\mathcal{L}(\mathbf{x})$ for every $\mathbf{x} \in \mathbb{R}^n$ and $a \in \mathbb{R}$.

2) $\mathcal{L}(\mathbf{x} + \mathbf{y}) = \mathcal{L}(\mathbf{x}) + \mathcal{L}(\mathbf{y})$ for every $\mathbf{x},\mathbf{y} \in \mathbb{R}^n$.

If we fix the based for $\mathbb{R}^n$ and $\mathbb{R}^m$, then the linear transformation $\mathcal{L}$ can be represented by a matrix. Specifically, there exists $\mathbf{A} \in \mathbb{R}^{m \times n}$ such that the following representation holds. Suppose that $\mathbf{x} \in \mathbb{R}^n$ is a given vector, and $\mathbf{x'}$ is the representation of $\mathbf{x}$ with the respect to the given basis for $\mathbb{R}^n$. If $\mathbf{y} = \mathcal{L}(\mathbf{x})$, and $\mathbf{y'}$ is the representation of $\mathbf{y}$ with respect to the given basis for $\mathbb{R}^m$, then

$$\mathbf{y'} = \mathbf{Ax'}$$

We call $\mathbf{A}$ the matrix representation of $\mathcal{L}$ with respect to the given bases for $\mathbb{R}^n$ and $\mathbb{R}^m$. In the special case where we assume the natural bases for $\mathbb{R}^n$ and $\mathbb{R}^m$, the matrix representation $\mathbf{A}$ satisfies

$$\mathcal{L}(\mathbf{x}) = \mathbf{Ax}$$

Let $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$ and $\\{\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n\\}$ be two bases for $\mathbb{R}^n$. Define the matrix 

$$\mathbf{T} = [\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n]^{-1}[\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n]$$

It is clear that

$$[\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n] = [\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n]\mathbf{T}$$

The $i$th column of $\mathbf{T}$ is the vector of coordinates of $\mathbf{e}_i$ with respect to the basis $\\{\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n\\}$.

Fix a vector in $\mathbb{R}^n$. Let $\mathbf{x}$ be the column of the coordinates of the vector with respect to $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$ and $\mathbf{x'}$ the coordinates of the same vector with respect to $\\{\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n\\}$. Then $\mathbf{x'} = \mathbf{Tx}$.

We call $\mathbf{T}$ the tranformation matrix from $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$ to $\\{\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n\\}$.

Consider a linear transformation

$$\mathcal{L}: \mathbb{R}^n \rightarrow \mathbb{R}^n$$

and let $\mathbf{A}$ be its representation with respect to  $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$ and $\mathbf{B}$ its representation with respect to $\\{\mathbf{e'}_1,\mathbf{e'}_2,...,\mathbf{e'}_n\\}$. Let $\mathbf{y} = \mathbf{Ax}$ and $\mathbf{y'} = \mathbf{Bx'}$. Therefore, $\mathbf{y'} = \mathbf{Ty} = \mathbf{TAx} = \mathbf{Bx'} = \mathbf{BTx}$, and hence $\mathbf{TA} = \mathbf{BT}$, or $\mathbf{B} = \mathbf{TAT}^{-1}$, or $\mathbf{A} = \mathbf{T}^{-1}\mathbf{BT}$.

Two $n \times n$ matrices $\mathbf{A}$ and $\mathbf{B}$ are similar if there exists a nonsingular matrix $\mathbf{T}$ such that $\mathbf{A} = \mathbf{T}^{-1}\mathbf{BT}$.

In conclusion, similar matrices correspond to the same linear transformation with respect to different bases.


## 2.3 特征值与特征向量（Eigenvalues and Eigenvectors）

Consider a linear transformation

$$\mathcal{L}: \mathbb{R}^n \rightarrow \mathbb{R}^n$$

an eigenvector $\mathbf{v}$ is a non-zero vector that only changes by a scalar factor when $\mathcal{L}$ is applied to it. This condition can be written as 

$$\mathcal{L}(\mathbf{v}) = \lambda\mathbf{v}$$

where $\lambda$ is a scalar in the field $\mathbb{R}$, konwn as the eigenvalue assocated with the eigenvector $\mathbf{v}$.  


If $\mathbf{A}$ is $\mathcal{L}$'s representation and $\mathbf{v'}$ $\mathbf{v}$'s representation with respect to $\\{\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n\\}$, it can be written by

$$\mathbf{Av'} = \lambda\mathbf{v'}$$

The scalar $\lambda$ and nonzero vector $\mathbf{v'}$ are said to be, respectively, an eigenvalue and an eigenvector of matrix $\mathbf{A}$.

For $\lambda$ to be an eigenvalue it is necessary and sufficient for the matrix $\lambda\mathbf{I}-\mathbf{A}$ to be singular; that is, $det [\lambda\mathbf{I}-\mathbf{A}] = 0$, where $\mathbf{I}$ is the $n \times n$ identity matrix. This leads to an $n$th-order polynomial equation

$$det [\lambda\mathbf{I}-\mathbf{A}] = \lambda^n + \alpha_{n-1}\lambda^{n-1} + \dots + \alpha_1\lambda + \alpha_0 = 0$$

We call the polynomial $det [\lambda\mathbf{I}-\mathbf{A}]$ the characteristic polynomial of the matrix $\mathbf{A}$, and the equation above the characteristic equation. 

According to the fundamental theorem of algebra, the characteristic equation must have $n$ (possibly nondistinct) roots that are the eigenvalues of $\mathbf{A}$. 

$\textbf{Theorem: }$ Suppose that the characteristic equation $det [\lambda\mathbf{I}-\mathbf{A}] = 0$ has $n$ distinct roots $\lambda_1,\lambda_2,...,\lambda_n$. Then, there exist $n$ linearly independent vectors $\mathbf{v}_1,\mathbf{v}_2,..,\mathbf{v}_n$ such that

$$\mathbf{Av}_i = \lambda_i\mathbf{v}_i (i = 1,2,...,n)$$

Consider a basis formed by a linearly independent set of eigenvectors $\\{ \mathbf{v}_1,\mathbf{v}_2,...,\mathbf{v}_n \\}$. With respect to this basis, the matrix $\mathbf{A}$ is diagonal (i.e., if $$a_{ij}$$ is the element of $\mathbf{A}$, then $$a_{ij} = 0$$ for all $i \neq j$). Indeed, let


$$\mathbf{T} = [\mathbf{v}_1,\mathbf{v}_2,..,\mathbf{v}_n]^{-1}[\mathbf{e}_1,\mathbf{e}_2,...,\mathbf{e}_n]$$

Then,

$$\mathbf{TAT}^{-1} = 
\begin{bmatrix} 	
\lambda_{1} &  &  & 0\\ 
 & \lambda_{2} &  &  \\ 
 & & \ddots & \\ 
0 & & & \lambda_{n} 
\end{bmatrix}
$$

$\textbf{Theorem: }$ All eigenvalues of a real symmetric matrix (a matrix $\mathbf{A}$ is symmetric if $\mathbf{A} = \mathbf{A}^T$) are real.

$\textbf{Theorem: }$ Any real symmetric $n \times n$ matrix has a set of $n$ eigenvectors that are mutually orthogonal.

If $\mathbf{A}$ is real symmetric, then a set of its eigenvectors forms an orthogonal basis for $\mathbb{R}^n$. If the basis $\\{ \mathbf{v}_1,\mathbf{v}_2,...,\mathbf{v}_n \\}$ is normalized so that each element has norm of unity, then defining the matrix

$$\mathbf{T} = [\mathbf{v}_1,\mathbf{v}_2,...,\mathbf{v}_n]$$

we have

$$\mathbf{T}^T\mathbf{T} = \mathbf{I}$$

and hence

$$\mathbf{T}^T = \mathbf{T}^{-1}$$

A matrix whose transpose is its inverse is said to be an orthogonal matrix.


## 2.4 正交投影

If $\mathcal{V}$ is a subspace of $\mathbb{R}^n$, then the orthogonal complement of $\mathcal{V}$, denoted $\mathcal{V}^{\bot}$, consists of all vectors that are orthogonal to every vector in $\mathcal{V}$. Thus,

$$\mathcal{V}^{\bot} = \{\mathbf{x} | \mathbf{v}^T\mathbf{x} = 0 \text{ for all } \mathbf{v} \in \mathcal{V} \}$$

$\mathcal{V}^{\bot}$ is also a subspace. Together, $\mathcal{V}$ and $\mathcal{V}^{\bot}$ span $\mathbb{R}^n$ in the sense that every vector $\mathbf{x} \in \mathbb{R}^n$ can be represented uniquely as 

$$\mathbf{x} = \mathbf{x}_1 + \mathbf{x}_2$$

where $\mathbf{x}_1 \in \mathcal{V}$ and $\mathbf{x}_2 \in \mathcal{V}^\bot$. We call the representation above the orthogonal decomposition of $\mathbf{x}$. We say that $\mathbf{x}_1$ and $\mathbf{x}_2$ are orthogonal projections of $\mathbf{x}$ onto the subspaces $\mathcal{V}$ and $\mathcal{V}^{\bot}$. We say that a linear transformation $\mathbf{P}$ is an orthogonal prjector onto $\mathcal{V}$ if for all $\mathbf{x} \in \mathbb{R}^n$, we have $\mathbf{Px} \in \mathcal{V}$ and $\mathbf{x} - \mathbf{Px} \in \mathcal{V}^{\bot}$.

In the subsequent discussion we use the following notation. Let $\mathbf{A} \in \mathbb{R}^{m \times n}$. Let the rangespace, or image, of $\mathbf{A}$ be denoted

$$\mathcal{R}(\mathbf{A}) \triangleq \{\mathbf{Ax} | \mathbf{x} \in \mathbb{R}^n\}$$

and the nullspace, or kernel, of $\mathbf{A}$ be denoted

$$\mathcal{N}(\mathbf{A}) \triangleq \{\mathbf{x} | \mathbf{Ax} = \mathbf{0}, \mathbf{x} \in \mathbb{R}^n\}$$

$\textbf{Theorem: }$ Let $\mathbf{A}$ be a given matrix. Then, $\mathcal{R}(\mathbf{A})^{\bot} = \mathcal{N}(\mathbf{A}^T)$ and $\mathcal{N}(\mathbf{A})^{\bot} = \mathcal{R}(\mathbf{A}^T)$.

There is also a fact that for any subspace $\mathcal{V}$, we have $(\mathcal{V}^{\bot})^{\bot} = \mathcal{V}$.

Note that if $\mathbf{P}$ is an orthogonal projector onto $\mathcal{V}$, then $\mathcal{R}(\mathbf{P}) = \mathcal{V}$ and $\mathbf{Px} = \mathbf{x}$ for all $\mathbf{x} \in \mathcal{V}$.

$\textbf{Theorem: }$ A matrix $\mathbf{P}$ is an orthogonal projector (onto the subspace $\mathcal{V} = \mathcal{R}(\mathbf{P})$) if and only if $\mathbf{P}^2 = \mathbf{P} = \mathbf{P}^T$.


## 2.5 二次型函数

A quadratic form $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a function

$$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Qx}$$

where $\mathbf{Q}$ is an $n \times n$ real matrix. There is no loss of generality in assuming $\mathbf{Q}$ to be symmetric. For if it is not symmetric, we can always replace it with the symmetric matrix $\mathbf{Q}_a$, say that

$$\mathbf{x}^T\mathbf{Qx} = \mathbf{x}^T\mathbf{Q}_a\mathbf{x} = \mathbf{x}^T(\frac{1}{2}\mathbf{Q} + \frac{1}{2}\mathbf{Q}^T)\mathbf{x}$$

A quadratic form $\mathbf{x}^T\mathbf{Qx} (\mathbf{Q}=\mathbf{Q}^T)$ is said to be positive definite, or positive semidefinite, if $\mathbf{x}^T\mathbf{Qx} > 0$, or $\mathbf{x}^T\mathbf{Qx} \geq 0$, for all nozero vectors $\mathbf{x}$. Similarly, we define the quadratic form to be negative definite, or negative semidefinite, if $\mathbf{x}^T\mathbf{Qx} < 0$, or $\mathbf{x}^T\mathbf{Qx} \leq 0$, for all nozero vectors $\mathbf{x}$.

Recall that the minors of a matrix $\mathbf{Q}$ are the determinants of the matrices obtained by successively removing rows and columns from $\mathbf{Q}$. The principal minors are $det \mathbf{Q}$ itself and the determinants of matrices obtained by successively removing an $i$th row and an $i$th column. The leading principal minors are $det \mathbf{Q}$ itself and the determinants of matrices obtained by successively removing the last row and the last column. That is, the leading pricipal minors are

$$
\Delta_1 = q_{11},
\Delta_2 = det \begin{bmatrix} q_{11} & q_{12} \\ q_{21} & q_{22} \end{bmatrix},
\Delta_3 = det \begin{bmatrix} q_{11} & q_{12} & q_{13} \\ q_{21} & q_{22} & q_{23} \\ q_{31} & q_{32} & q_{33} \end{bmatrix},
\ldots, 
\Delta_n = det \mathbf{Q}
$$

$\textbf{Theorem: Sylvester's Criterion.}$ A quadratic form $\mathbf{x}^T\mathbf{Qx} (\mathbf{Q}=\mathbf{Q}^T)$ is positive definite if and only if the leading principal minors of $\mathbf{Q}$ are positive.

Note that if $\mathbf{Q}$ is not symmetric, Sylvester's criterion cannot be used to check positive definiteness of the quadratic form $\mathbf{x}^T\mathbf{Qx}$.

A necessary condition for a real quadratic form to be positive semidefinite is that the leading principal minors be nonnegative. However, this is not a sufficient condition. In fact, a real quadratic form is positive semidefinite if and only if all principal minors are nonnegative.

A symmetric matrix $\mathbf{Q}$ is said to be positive definite if the quadratic form $\mathbf{x}^T\mathbf{Qx}$ is positive definite. Similarly, we define a symmetric matrix $\mathbf{Q}$ to be positive semidefinite, negative definite, and negative semidefinite if the corresponding quadratic froms have the respective properties. The symmetric matrix $\mathbf{Q}$ is indefinite if it is neither positive semidefinite nor negative semidefinite. Note that the symmetric matrix $\mathbf{Q}$ is positive definite (semidefinite) if and only if the matrix $-\mathbf{Q}$ is negative definite (semidefinite).

Sylvester's criterion provides a way of checking the definiteness of a quadratic form, or equivalently, a symmetric matrix. An alternative method involves checking the eigenvalues of $\mathbf{Q}$, as stated below.

$\textbf{Theorem: }$ A symmetric matrix $\mathbf{Q}$ is positive definite (or positive semidefinite) if and only if all eigenvalues of $\mathbf{Q}$ are positive (or nonnegative).

$\textbf{Theorem: Rayleigh's Inequaliteis.}$ If an $n \times n$ matrix $\mathbf{P}$ is a real symmetric positive definite, then 

$$\lambda_{min}(\mathbf{P})\|\mathbf{x}\|^2 \leq \mathbf{x}^T\mathbf{Px} \leq \lambda_{max}(\mathbf{P})\|\mathbf{x}\|^2$$

where $\lambda_{min}(\mathbf{P})$ denotes the smallest eigenvalue of $\mathbf{P}$, and $\lambda_{max}(\mathbf{P})$ denotes the largest eigenvalue of $\mathbf{P}$.

In summary, we have presented two tests for definiteness of quadratic forms and symmetric matrices. We point out again that nonnegativity of leading principal minors is a necessary but not a sufficient condition for positive semidefiniteness.


## 2.6 矩阵范数

The norm of a matrix may be chosen in a variety of ways. Because the set of a matrices $\mathbb{R}^{m \times n}$ can be viewed as the real vector space $\mathbb{R}^{mn}$, matrix norms should be no different form regular vector norms. Therefore, we define the norm of a matrix $\mathbf{A}$, denoted $\\|\mathbf{A}\\|$, to be any function $\\|\cdot\\|$ that satisfies the following conditions:

1) $\\|\mathbf{A}\\| > 0$ if $\mathbf{A} \neq \mathbf{O}$, and $\\|\mathbf{O}\\| = 0$, where $\mathbf{O}$ is a matrix with all entries equal to zero.

2) $\\|c\mathbf{A}\\| = \|c\|\\|\mathbf{A}\\|$ for any $c \in \mathbb{R}$.

3) $\\|\mathbf{A} + \mathbf{B}\\| \leq \\|\mathbf{A}\\| + \\|\mathbf{B}\\|$.

An example of a matrix norm is the Frobenius norm, defined as

$$\|\mathbf{A}\|_F = \left(\sum_{i=1}^m\sum_{j=1}^n a_{ij}^2\right)^{\frac{1}{2}}$$

where $\mathbf{A} \in \mathbb{R}^{m \times n}$. Note that the Frobenius norm is equivalent to the Euclidean norm on $\mathbb{R}^{mn}$.

We consider only matrix norms that satisfy the following additional condition:

4) $\\|\mathbf{AB}\\| \leq \\|\mathbf{A}\\|\\|\mathbf{B}\\|$.

It turns out that the Frobenius norm satisfies condition 4 as well.

In many problems, both matrices and vectors appear simultaneously. Therefore, it is convenient to construct the norm of a matrix in such a way that it will be related to vector norms. To this end we consider a special class of matrix norms, called induced norms. Let $$\|\cdot\|_{(n)}$$ and $$\|\cdot\|_{(m)}$$ be vector norms on $\mathbb{R}^n$ and $\mathbb{R}^m$, respectively. We say that the matrix norm is induced by, or is compatible with, the given vector norms if if for any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and any vector $\mathbf{x} \in \mathbb{R}^n$, the following inequality is satisfied:

$$\|\mathbf{Ax}\|_{(m)} \leq \|\mathbf{A}\|\|\mathbf{x}\|_{(n)}$$

We can define an induced matrix norm as 

$$\|\mathbf{A}\| = \max_{\|\mathbf{x}\|_{(n)}=1}\|\mathbf{Ax}\|_{(m)}$$

that is, $\\|\mathbf{A}\\|$ is the maximum of the norms of the vectors $\mathbf{Ax}$ where the vector $\mathbf{x}$ runs over the set of all vectors with unit norm. 

This induced matrix norm also satisfies conditions 1 to 4. 

$\textbf{Theorem: }$ Let

$$\|\mathbf{x}\| = \left(\sum_{k=1}^n|x_k|^2\right)^{1/2} = \sqrt{\mathbf{x} \bullet \mathbf{x}}$$

The matrix norm induced by this vector norm is 

$$\|\mathbf{A}\| = \sqrt{\lambda_{max}(\mathbf{A}^T\mathbf{A})}$$

where $\lambda_{max}(\mathbf{A}^T\mathbf{A})$ is the largest eigenvalue of the matrix $\mathbf{A}^T\mathbf{A}$.


# 3、线性几何

## 3.1 线段（Line Segments）

In the following analysis we concern ourselves only with $\mathbb{R}^n$. The elements of this space are the $n$-component vectors $\mathbf{x} = [x_1,x_2,...,x_n]^T$.

The line segment between two points $\mathbf{x}$ and $\mathbf{y}$ in $\mathbb{R}^n$ is the set of points on the straight line joining points $\mathbf{x}$ and $\mathbf{y}$. Note that if $\mathbf{z}$ lies on the line segment between $\mathbf{x}$ and $\mathbf{y}$, then

$$\mathbf{z} - \mathbf{y} = \alpha (\mathbf{x}-\mathbf{y})$$

where $\alpha$ is a real number from the interval $[0,1]$. The equation above can be rewritten as $\mathbf{z} = \alpha\mathbf{x} + (1 - \alpha)\mathbf{y}$. Hence, the line segment between $\mathbf{x}$ and $\mathbf{y}$ can be represented as 

$$\{\alpha\mathbf{x} + (1 - \alpha)\mathbf{y} | \alpha \in [0,1]\}$$


## 3.2 超平面与线性簇（Hyperplanes and Linear Varieties）

Let $u_1,u_2,...,u_n,v \in \mathbb{R}$, where at least one of the $u_i$ is nonzero. The set of all points $\mathbf{x} = [x_1,x_2,...,x_n]^T$ that satisfy the linear equation

$$u_1x_1 + u_2x_2 + \dots + u_nx_n = v$$

is called a hyperplane of the space $\mathbb{R}^n$. We may describe the hyperplane by 

$$H = \{\mathbf{x} \in \mathbb{R}^n | \mathbf{u}^T\mathbf{x} = v\}$$

where $\mathbf{u} = [u_1,u_2,...,u_n]^T$.

A hyperplane is not necessarily a subspace of $\mathbb{R}^n$ since, in general, it does not contain the origin. By translating a hyperplane so that it contains the origin of $\mathbb{R}^n$, it becomes a subspace of $\mathbb{R}^n$. Because the dimension of this subspace is $n-1$, we say that the hyperplane has dimension $n-1$.

The hyperplane divides $\mathbb{R}^n$ into two half-spaces $$H_{+}$$ and $$H_{-}$$, denoted

$$
H_+ = \{\mathbf{x} \in \mathbb{R}^n | \mathbf{u}^T\mathbf{x} \geq v\} \\
H_- = \{\mathbf{x} \in \mathbb{R}^n | \mathbf{u}^T\mathbf{x} \leq v\}
$$

The half-space $$H_{+}$$ is called the positive half-space, and the half-space $$H_{-}$$ is called the negative half-space.

Let $\mathbf{a} = [a_1,a_2,...,a_n]^T$ be an arbitrary point of the hyperplane $H$. Note that the hyperplane $H$ consists of the point $\mathbf{x}$ for which $\mathbf{u} \bullet (\mathbf{x}-\mathbf{a}) = 0$. In other words, the hyperplane $H$ consists of points $\mathbf{x}$ for which the vectors $\mathbf{u}$ and $\mathbf{x}-\mathbf{a}$ are orthogonal. We call the vector $\mathbf{u}$ is __normal__ to the hyperplane $H$. The set $$H_+$$ consists of those points $\mathbf{x}$ for which $\mathbf{u} \bullet (\mathbf{x}-\mathbf{a}) \geq 0$, and $$H_-$$ consists of those points $\mathbf{x}$ for which $\mathbf{u} \bullet (\mathbf{x}-\mathbf{a}) \leq 0$.

A linear veriety is a set of the form

$$\{\mathbf{x} \in \mathbb{R}^n | \mathbf{A}\mathbf{x} = \mathbf{b}\}$$

for some matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and vector $\mathbf{b} \in \mathbb{R}^m$. A linear variety is a subspace if and only if $\mathbf{b} = \mathbf{0}$. If $\mathbf{A} = \mathbf{O}$, the linear variety is $\mathbb{R}^n$. If $dim\mathcal{N}(\mathbf{A}) = r$, we say that the linear variety has dimension $r$. If the dimension of the linear variety is less than $n$, then it is the intersection of a finite number of hyperplanes.

## 3.3 凸集与凸函数

### 3.3.1 凸集

A set $\Theta \in \mathbb{R}^n$ is convex if for all $\mathbf{u},\mathbf{v} \in \Theta$, the line segment between $\mathbf{u}$ and $\mathbf{v}$ is in $\Theta$. Note that $\Theta$ is convex if and only if $\alpha\mathbf{u} + (1 - \alpha)\mathbf{v} \in \Theta$ for all $\mathbf{u},\mathbf{v} \in \Theta$ and $\alpha \in (0,1)$.

$\textbf{Theorem: }$ Convex subsets of $\mathbb{R}^n$ have the following properties:

1) If $\Theta$ is a convex set and $\beta$ is a real number, then the set

$$\beta\Theta = \{\beta\mathbf{v} | \mathbf{v} \in \Theta\}$$

is also convex.

2) If $\Theta_1$ and $\Theta_2$ are convex sets, then the set

$$\Theta_1 + \Theta_2 = \{\mathbf{v}_1 + \mathbf{v}_2 | \mathbf{v}_1 \in \Theta_1, \mathbf{v}_2 \in \Theta_2\}$$

is also convex.

3) The intersection of any collection of convex sets is convex.

A point $\mathbf{x}$ in a convex set $\Theta$ is said to be an extreme point of $\Theta$ if there are no two distinct points $\mathbf{u}$ and $\mathbf{v}$ in $\Theta$ such that $\mathbf{x} = \alpha\mathbf{u} + (1 - \alpha)\mathbf{v}$ for some $\alpha \in (0,1)$.

### 3.3.2 凸函数

__Definition:__ The epigraph of a function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, denoted $epi(f)$, is the set of points in $\Omega\times\mathbb{R}$ given by 

$$epi(f) = \left\{ \begin{bmatrix}\mathbf{x} \\ \beta\end{bmatrix} \Big| \mathbf{x}\in\Omega, \beta\in\mathbb{R}, \beta\geq f(\mathbf{x}) \right\}$$

The epigraph of a function $f$ is simply the set of points in $\Omega\times\mathbb{R}$ on or above the graph of $f$.

__Definition:__ A function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, is convex on $\Omega$ if its epigraph is a convex set. 

__Theorem:__ If a function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, is convex on $\Omega$, then $\Omega$ must be a convex set.

__Theorem:__ If a function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, is convex on $\Omega$ if and only if for all $\mathbf{x,y}\in\Omega$ and all $\alpha \in (0,1)$, we have

$$f(\alpha\mathbf{x}+(1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})$$

__Theorem:__ Suppose that $$f_1$$ and $$f_2$$ are convex funcitons. Then, for any $a\geq 0$, the function $$af_1$$ is a convex. Moreover, $$f_1+f_2$$ is convex.

__Definition:__ A function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, is strictly convex on convex set $\Omega$ if and only if for all $\mathbf{x,y}\in\Omega$ and all $\alpha \in (0,1)$, we have

$$f(\alpha\mathbf{x}+(1-\alpha)\mathbf{y}) < \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})$$

__Definition:__ A function $f: \Omega\rightarrow\mathbb{R}\ (\Omega\subset\mathbb{R}^n)$, is (strictly) concave on convex set $\Omega$ if $-f$ is (strictly) convex.


## 3.4 邻域

A neighborhood of a point $\mathbf{x} \in \mathbb{R}^n$ is the set

$$\{\mathbf{y} \in \mathbb{R}^n | \|\mathbf{y} - \mathbf{x}\|<\epsilon\}$$

where $\epsilon$ is some positive number. The neighborhood is also called a ball with radius $\epsilon$ and center $\mathbf{x}$.

A point $\mathbf{x} \in S$ is said to be an interior point of the set $S$ if the set $S$ contains some neighborhood of $\mathbf{x}$; that is, if all points within some neighborhood of $\mathbf{x}$ are also in S. The set of all interior points of $S$ is called the interior of $S$.

A point $\mathbf{x}$ is said to be a boundary point of the set $S$ if every neighborhood of $\mathbf{x}$ contains a point in $S$ and a point not in $S$. Note that a boundary point of $S$ may or may not be an element of $S$. The set of all boundary points of $S$ is called the boundary of $S$.

A set $S$ is said to be open if it contains a neighborhood of each of its points; that is, if each of its points is an interior point, or equivalently, if $S$ contains no boundary points.

A set $S$ is said to be closed if it contains its boundary. We can see that a set is closed if and only if its complement is open.

A set that is contained in a ball of finite radius is said to be bounded. A set is compact if it is both closed and bounded. 

Compact sets are important in optimization problems.

$\textbf{Theorem: Theorem of Weierstrass.}$ Let $f: \Omega \rightarrow \mathbb{R}$ be a continuous function, where $\Omega \subset \mathbb{R}^n$ is a compact set. Then, there exists $\mathbf{x}_0,\mathbf{x}_1 \in \Omega$ such that $f(\mathbf{x}_0) \leq f(\mathbf{x}) \leq f(\mathbf{x}_1)$ for all $\mathbf{x} \in \Omega$.


## 3.5 多面体与多胞体（Polytopes and Polyhedra）

Let $\Theta$ be a convex set, and suppose that $\mathbf{y}$ is a boundary point of $\Theta$. A hyperplane passing through $\mathbf{y}$ is called a hyperplane of support (or supporting hyperplane) of the set $\Theta$ if the entire set $\Theta$ lies completely in one of the two half-spaces into which this hyperplane divides the space $\mathbb{R}^n$.

Recall that the intersection of any number of convex sets is convex. In what follows we are concerned with the intersection of a finite number of half-spaces. Because every half-space $$H_+$$ or $$H_-$$ is convex in $\mathbb{R}^n$, the intersection of any number of half-spaces is a convex set.

A set that can be expressed as the intersection of a finite number of half-spaces is called a comvex polytope.

A nonempty bounded polytope is called a polyhedron.

For every convex polyhedron $\Theta \subset \mathbb{R}^n$, there exists a nonnegative integer $k \leq n$ such that $\Theta$ is contained in a linear variety of dimension k, but is not entirely contained in any $(k-1)$-dimensional linear variety of $\mathbb{R}^n$. Furthermore, there exists only one $k$-dimensional linear variety containing $\Theta$, called the carrier of the polyhedron $\Theta$, and $k$ is called the dimension of $\Theta$. For example, a zero-dimensional polyhedron is a point of $\mathbb{R}^n$, and its carrier is itself. A one-dimensional polyhedron is a segment, and its carrier is the straight line on which it lies. The boundary of any $k$-dimensional polyhedron, $k > 0$, consists of a finite number of $k-1$-dimensional polyhedra. For example, the boundary of a one-dimensional polyhedron consists of two points that are the endpoints of the segment.

The $(k-1)$-dimensional polyhedra forming the boundary of a $k$-dimensional polyhedron are called the faces of the polyhedron. Each of these faces has, in turn, $(k-1)$-dimensional faces. We also consider each of these $(k-2)$-dimensional faces to be faces of the original $k$-dimensional polyhedron. Thus, every $k$-dimensional polyhedron has faces of dimensions $k-1,k-2,...,1,0$. A zero-dimensional face of a polyhedron is called a vertex, and a one-dimensional face is called an edge.


