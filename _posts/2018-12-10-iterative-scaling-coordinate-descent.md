---
layout: post
title: Iterative Scaling and Coordinate Descent
tags: ["machine-learning","optimization","maxent"]
mathjax: true
---

Recently, I was reading a paper on language model adaptation, which used an optimization technique called Generalized Iterative Scaling (GIS). Having no idea what the method was, I sought out [the first paper](https://www.jstor.org/stable/2240069?seq=1#metadata_info_tab_contents) which proposed it, but since the paper is from 1972, and I am not a pure math guy, I found it difficult to follow. After some more looking around, I chanced upon this lucid JMLR'10 paper from [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/): [Iterative Scaling and Coordinate Descent Methods for Maximum Entropy Models](http://www.jmlr.org/papers/volume11/huang10a/huang10a.pdf). In this post, I will summarize the ideas in the paper, primarily the discussion about a unified framework for **Iterative Scaling** (IS) and **Coordinate Descent** (CD) methods, and how each particular technique is derived from this general framework.

## The General Framework


Iterative Scaling (IS) and Coordinate Descent (CD) are methods used to optimize maximum entropy (maxent) models. *What is a maxent model?* Given a sequence $x$, a maxent model predicts the label sequence $y$ with maximal probability. It is discriminatively trained by modeling the conditional probability

$$ P_{\mathbf{w}}(y|x) = \frac{S_{\mathbf{w}}(x,y)}{T_{\mathbf{w}}(x)}, $$

where $S_{\mathbf{w}}(x,y) = \exp(\sum_t w_t f_t(x,y))$ and $T_{\mathbf{w}}(x) = \sum_y S_{\mathbf{w}}(x,y)$.

Note that each of the $f_t$ are features which can be defined arbitrarily with the sole constraint that they must be non-negative. Each $f_t$ has a corresponding weight $w_t$ which needs to be estimated. IS and CD methods do this estimation by iterating over all the $w_t$'s, either sequentially or in parallel. Based on the above conditional probability, we can define an objective function by taking the log of the probability and adding an L2-regularization term to it as

$$ \text{min}_{\mathbf{w}} L(\mathbf{w}) \equiv \text{min}_{\mathbf{w}} \sum_x \tilde{P}(x) \log T_{\mathbf{w}}(x) - \sum_t w_t \tilde{P}(f_t) + \frac{1}{2\sigma^2}\sum_t w_t^2. $$

Here, $\tilde{P}(x) = \sum_y \tilde{P}(x,y),$ where $\tilde{P}(x,y)$ is the empirical distribution, and $\tilde{P}(f_t)$ is the expected value of $f_t(x,y)$. The log-likelihood itself (without regularization) is convex, but adding the regularization term makes it strictly convex, and it can also be shown that this objective function has a unique global minima. 

If we update our weights (either in parallel or in sequence), after one such iteration of updation, we change our objective function from $L(\mathbf{w})$ to $L(\mathbf{w}+\mathbf{z})$, where $\mathbf{z}$ si the update made to the weights. Each such iteration can be written as a subproblem which we need to solve, i.e.

$$ A(\mathbf{z}) \leq L(\mathbf{w}+\mathbf{z}) - L(\mathbf{w}).$$

In addition, if we have $A(0) = 0$, this implies that $L$ decreases with every update. Let us now expand the RHS in the above equation. We have

$$ \begin{align} L(\mathbf{w}+\mathbf{z}) - L(\mathbf{w}) &= \sum_x \tilde{P}(x) \log T_{\mathbf{w}+\mathbf{z}}(x) - \sum_t w_t \tilde{P}(f_t) + \frac{1}{2\sigma^2}\sum_t (w_t+z_t)^2 \\ & - \sum_x \tilde{P}(x) \log T_{\mathbf{w}}(x) + \sum_t w_t \tilde{P}(f_t) - \frac{1}{2\sigma^2}\sum_t w_t^2 \\ &= \sum_x \tilde{P}(x) \log \frac{T_{\mathbf{w}+\mathbf{z}}(x)}{T_{\mathbf{w}}(x)} + \sum_t Q_t (z_t) \end{align} $$

where $Q_t(z_t) \equiv \frac{2w_tz_t + z_t^2}{2\sigma^2} - z_t \tilde{P}(f_t)$. Further, the ratio in the log term can be simplified as

$$ \frac{T_{\mathbf{w}+\mathbf{z}}(x)}{T_{\mathbf{w}}(x)} = \sum_y P_{\mathbf{w}}(y|x)e^{\sum_t z_t f_t(x,y)}. $$

This is the general overview of the problem that all IS and CD methods solve. The difference is in how this function is minimized. Let us look at each of the methods and how they build upon this general framework.

### Coordinate Descent

[CD](https://link.springer.com/article/10.1007%2Fs10107-015-0892-3) solves the exact problem without any approximation, i.e., the subproblem is

$$ A(\mathbf{z}) = L(\mathbf{w}+\mathbf{z}) - L(\mathbf{w}) $$

This then leads to the subproblem be exactly equal to as derived above. This has an advantage and a limitation.

* Since $A(\mathbf{z})$ here is the maximum possible decrement in any iteration, the convergence requires the least number of steps out of all possible approximations of $A(\mathbf{z})$.

* Because of the presence of the log term in the objective function, there is no closed form solution, and so every iteration must solve an optimization problem using the Newton method. 

In practice, the Newton optimization at each step overshadows any gain due to fewer iterations till convergence, so that CD takes more time to converge than IS methods which approximate $A(\mathbf{z})$.

### Generalized IS (GIS) and Sequential Conditional GIS (SC-GIS)

[GIS](https://www.jstor.org/stable/2240069?seq=1#metadata_info_tab_contents) and [SC-GIS](http://www.aclweb.org/anthology/P02-1002) use the approximation $\log \alpha \leq \alpha -1$ to get

$$ \begin{align} L(\mathbf{w}+\mathbf{z}) - L(\mathbf{w}) &\leq \sum_t Q_t (z_t) + \sum_x \tilde{P}(x)       (\sum_y P_{\mathbf{w}}(y|x) e^{\sum_t z_t f_t(x,y)} - 1) \\ &= \sum_t Q_t (z_t) + \sum_{x,y} \tilde{P}(x) (P_{\mathbf{w}}(y|x)e^{\sum_t z_t f_t(x,y)} - 1)   \end{align} $$

Define $f^{\\#}(x,y) = \sum_t f_t(x,y)$ and $f^{\\#}=\text{max}\_{x,y}(f^{\\#}(x,y))$. We can then use Jensen's inequality to upper bound the exponential term in the above inequality. GIS is a parallel update method, i.e., all the $w_t$'s are updated simultaneously, which means that we can use $f^{\\#}$ to bound the exponential terms. On the contrary, SC-GIS is a sequential method, which means we can only use $f_t^{\\#}$ to get this bound, where $f_t^{\\#} \equiv \text{max}\_{x,y}f_t(x,y)$. Finally, the subproblems can be written as

$$ A_t^{GIS}(z_t) = Q_t (z_t) + \frac{e^{z_t f^{\#}}-1}{f^{\#}}\sum_{x,y} \tilde{P}(x) P_{\mathbf{w}}(y|x)f_t(x,y)  $$

$$ A_t^{SC-GIS}(z_t) = Q_t (z_t) + \frac{e^{z_t f_t^{\#}}-1}{f_t^{\#}}\sum_{x,y} \tilde{P}(x) P_{\mathbf{w}}(y|x)f_t(x,y) $$

### Improved IS (IIS)

A problem with bounding in terms of $f^{\\#}$ as done in GIS is that $f^{\\#}$ can be too large even if one of the $(x,y)$ pairs has a large value of $f^{\\#}(x,y)$. This would cause the subproblem to be very small, similar to the issue of small learning rates in gradient-based optimization. To remedy this, we can bound in terms of $f^{\\#}(x,y)$, although in that case we the term cannot be taken out of the summation. This is what is done in IIS, and this gives the following definition of the subproblem.

$$ A_t^{IIS}(z_t) = Q_t (z_t) + \sum_{x,y} \tilde{P}(x) P_{\mathbf{w}}(y|x)f_t(x,y)\frac{e^{z_t f^{\#}(x,y)}-1}{f^{\#}(x,y)} $$

## Key points

1. Iterative scaling and coordinate descent methods have provably linear convergence.

2. However, the time complexity of solving each subproblem is key in choosing which method to use for optimization.

3. GIS and SC-GIS have closed form solutions for the subproblems, which makes it $\mathcal{O}(1)$ to solve each iteration.

4. Although CD and IIS need Newton optimization for each subproblem, the authors propose a fast CD method which performs only 1 update in later iterations. This is because it is empirically observed that a single update is enough to update the weight sufficiently in later stages.

