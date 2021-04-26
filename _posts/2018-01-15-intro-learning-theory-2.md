---
layout: post
title: Introduction to Learning Theory - Part 2
tags: ["learning theory"]
mathjax: true
---

In the [first part]({% post_url 2018-01-15-intro-learning-theory-1 %}) of this series on learning theory, we looked only at the case of finite hypothesis sets, and derived some generalization bounds using the PAC learning framework. However, in most practical cases, the hypothesis class is usually infinite. To measure the complexity of the class in such cases, 3 different measures are often used — Rademacher complexity, growth function, and VC dimension. In this article, I will discuss all of these.

#### Rademacher complexity

Given a family of functions, one of the ways to measure its complexity is to see how well it can fit a random assignment of labels. A more complex hypothesis set would be able to fit a random noise better, and vice versa. For this purpose, we define $m$ random variables $\sigma_i$, called Rademacher variables. We then define the *empirical* Rademacher complexity as

$$ \hat{\mathcal{R}_S}(G) = \mathbb{E}_{\sigma}[\text{sup}_{g\in G}\frac{1}{m}\sigma_i g(z_i)] $$

Here the summation term is essentially the inner product of the vector of noise (Rademacher variables) and the labels with some $g \in G$. Intuitively, this term can be taken to represent the correlation between the actual assignment and the random assignment. On taking the supremum over all $g \in G$, we are computing how well the function class $G$ correlates with random noise on $S$. The expectation of this term over all random noise distributions measures the average correlation.

Therefore, a higher Rademacher complexity would imply that the function class $G$ is able to fit a random assignment of labels well, and vice versa. This is because the more complex a class $G$ is, higher is the probability that it would have some $g$ which correlates well with random noise.

However, this is just the empirical R.C. since we are computing the mean on the given sample set. The actual R.C. is obtained by taking the expectation of this value by sampling $S$ from a distribution $D$ consisting of sample sets of size $m$. Having thus defined the R.C., we can obtain an upper bound on the expected value of an error function $g$ taken from a family of functions $G$.

$$ \mathbb{E}[g(z)] \leq \frac{1}{m} \sum_{i=1}^m g(z_i) + 2\mathcal{R}_m(G) + \sqrt{\frac{\log \frac{1}{\delta}}{2m}} $$

Note that if we take the first term on RHS to LHS, the LHS becomes the maximum difference between the empirical and general loss (function value if function is binary-valued). We have access to the empirical values, but not the expectation. So we take 2 sample sets A and B which differ at only 1 point, so that we can use the McDiarmid’s inequality.

> The McDiarmid’s inequality bounds the probability that the actual mean and expected mean of a function differ by more than a fixed quantity, given that the function does not deviate by a large amount on perturbing a single element.

The actual proof then becomes simply manipulating the expectation and supremum using Jensen’s inequality (function of an expectation is at most expectation of the function, if the function itself is convex). I do not go into the details of the proof here since it is readily available.

Till now, we have only computed the bounds on the expectation of the set of loss functions $G$. We actually need to compute bounds on the general loss on the hypothesis class $H$, which assigns binary values to given samples. For this, we use the following lemma which is simple to prove.

$$ \hat{\mathcal{R}_S} (G) = \frac{1}{2}\hat{\mathcal{R}_{S_X}}(G) $$

From this and the earlier result, we easily arrive at an upper bound on the generalization error of the hypothesis class in terms of its Rademacher complexity.

$$ R(h) \leq \hat{R}(h) + \mathcal{R}_m(H) + \sqrt{\frac{\log \frac{1}{\delta}}{2m}} $$

Here, computing the empirical loss is simple, but computing the R.C. for some hypothesis sets may be hard (since it is equivalent to an empirical risk minimization problem). Therefore, we need some complexity measures which are easier to compute.

#### Growth function

The growth function of a hypothesis class $H$ for sample size $m$ denotes the number of distinct ways that $H$ can classify the sample. A more complex hypothesis class would be able to have a larger number of possible combinations for any sample size $m$. However, unlike R.C., this measure is purely combinatorial, and independent of the underlying distributions in $H$.

The Rademacher complexity and the growth function are related by Massart’s lemma as

$$ \mathcal{R}_m(G) \leq \sqrt{\frac{2\log \prod_G (m) }{m}} $$

> The Massart’s lemma bounds the expected correlation of a given vector taken from a set with a vector of random noise, in terms of the size of the set, dimensionality of the set, and the maximum L2-norm of the set.

As soon as we see “expected correlation,” we should think of the Rademacher complexity. To introduce the growth function, we use the term for the size of the set, since it essentially denotes the size of set containing all possible assignments for a sample.

Using this relation in the earlier obtained upper bound, we can bound the generalization error in terms of the growth function.

Although it is a combinatorial quantity, the growth function still depends on the sample size $m$, and thus would require repeated calculations for all values $m>1$. Instead, we turn to the third and most popular complexity measure for hypothesis sets.

#### VC-dimension

The VC-dimension of a hypothesis class is the size of the largest set that can be fully shattered by it. By shattering, we mean that $H$ can classify the given set in all possible ways. Formally,

$$ VCdim(H) = \max\{ m:\prod_H (m) = 2^m \} $$

It is important to understand 2 things:

1.  If $VCdim(H) = d$, then there exists a set of size $d$ that can be fully shattered. This does not mean that all sets of size $d$ or less are fully shattered by $H$.
2.  Also, in this case, no set of size greater than $d$ can ever be shattered by $H$.

To relate VC-dimension with the growth function, we use the Sauer’s lemma:

$$ \prod_H(m) \leq \sum_{i=0}^m {m\choose i} $$

Here, the LHS, which is the growth function, represents the number of possible behaviors that $H$ can have on a set of size $m$. The RHS is the number of small subsets that are completely shattered by $H$. For a detailed proof, I highly recommend [this lecture](https://www.youtube.com/watch?v=LHIwWeQhhk4) (Actually, I would highly recommend the entire course).

Using some manipulations on the combinatorial, we arrive at

$$ \prod_H(m) \leq  \left( \frac{em}{d} \right)^d = \mathcal{O}(m^d) $$

Now we can use this relation with the earlier results to bound the generalization error in terms of the VC-dimension of the hypothesis class.

$$ R(h) \leq \hat{R}(h) + \mathcal{O}\left( \sqrt{\frac{\log(m/d)}{m/d}} \right) $$

where $m$ is the sample size and $d$ is the VC-dimension.

*****

Here is a quick recap:

* Rademacher complexity — ability to fit random labels (using correlation)
* Growth function — number of distinct behaviors on $m$
* VC-dimension — largest set size that can be fully shattered

This blog post is loosely based on notes made from Chapter 3 “Rademacher complexity and VC-Dimension” of *Foundations of Machine Learning.*
