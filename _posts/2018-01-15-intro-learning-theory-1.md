---
layout: post
title: Introduction to Learning Theory - Part 1
tags: ["learning theory"]
mathjax: true
---

One of the most significant take-aways from NIPS 2017 was the ["alchemy" debate](https://syncedreview.com/2017/12/12/lecun-vs-rahimi-has-machine-learning-become-alchemy/) spearheaded by [Ali Rahimi](https://www.linkedin.com/in/ali-rahimi-a85104/). In the wake of the event, I have been trying to learn more about statistical learning theory, even though the concepts may not be readily applicable to deep neural networks.

One of the most important concepts in this regard is to measure the complexity of a hypothesis class $H$. In any machine learning model, the end goal is to find a hypothesis class that achieves a high accuracy on the training set, and has low generalization error on the test set. For this, we require the hypothesis class $H$ to approximate the concept class $C$ which determines the labels for the distribution $D$. Since both $C$ and $D$ are unknown, we try to model $H$ based on the known sample set $S$ and its labels.

**Generalization error:** The generalization error of a hypothesis $h$ is the expectation of the error on a sample $x$ picked from the distribution $D$.

**Empirical error:** This is the mean of the error of hypothesis $h$ on the sample $S$ of size $m$.

Having defined the generalization error and empirical error thus, we can state the objective of learning as follows.

> The objective of learning is to have the empirical error approximate the generalization error with high probability.

This kind of a learning framework is known as **PAC-learning** (Probably Approximately Correct). Formally, a concept class $C$ is PAC-learnable if there is some algorithm A for which the generalization error on a sample $S$ derived from the distribution $D$ is very low (less than $\epsilon$) with high probability (greater than $1- \delta$). In other words, we can say that for a PAC-learnable class, the accuracy is high with good confidence.

### Guarantees for finite hypothesis sets

The PAC-learning framework provides strong guarantees for finite hypothesis sets (i.e., where the size of $H$ is finite). Again, this falls in two categories — the consistent case, and the inconsistent case. A hypothesis class is said to be *consistent* if it admits no error on the training sample, i.e., the training accuracy is 100%.

#### Consistent hypothesis

Let us consider a finite hypothesis set $H$. We want the generalization error to be less than some $\epsilon$, so we will take a consistent hypothesis $h \in H$, and bound the probability that its error is more than $\epsilon$, i.e., we are calculating the probability that there exists some $h \in H$, such that $h$ is consistent and its generalization error is more than $\epsilon$. This is simply the union of all $h \in H$ such that it follows the said constraints. By the union bound, this
probability will be less than the sum of the individual probabilities i.e.,

$$ \sum_{h\in H}Pr[\hat{R}(h)=0 \wedge R(h) > \epsilon] $$

From the definition of conditional probability, we can write

$$ Pr(A \cap B) = Pr(A|B)Pr(B) \leq Pr(A|B) $$

which bounds the required probability $P$ as

$$ P \leq \sum_{h\in H} Pr[\hat{R}(h) =0| R(h) > \epsilon] $$

The condition says that the expectation of error of $h$ on any sample is at least $\epsilon$, so it would correctly classify a sample with probability at most $1-\epsilon$. Hence, to correctly classify $m$ training samples with $\vert H\vert$ hypotheses, the total probability is given as

$$ P \leq |H|(1-\epsilon)^m \leq |H|\exp(-m\epsilon) $$

On setting the RHS of the inequality to $\delta$, we obtain the generalization bound of the finite, consistent hypothesis class as

$$ R(h_S) \leq \frac{1}{m}\left( \log |H| + \log \frac{1}{\delta} \right) $$

As expected, the generalization error decreases with a larger training set. However, to arrive at a consistent algorithm, we may have to increase the size of the hypothesis class, which results in an increase in generalization error.

#### Inconsistent hypothesis

In practical scenarios, it is very restrictive to always require a consistent hypothesis class to bound the generalization error. In this section, we look at a more general case where empirical error is non-zero. For this derivation, we use the **Hoeffding’s inequality** which provides an upper bound on the probability that the mean of independent variables in an interval $[0,1]$ deviates from its expected value by more than a certain amount.

$$ P(\bar{X} - \mathbb{E}\bar{X} \geq t) \leq \exp(-2nt^2) $$

If we take the errors as the random variable, their mean is the empirical error and the expectation is the generalization error. We can then get an upper bound for the generalization error of a single hypothesis $h$ as

$$ R(h) \leq \hat{R}(h) + \sqrt{\frac{\log \frac{2}{\delta}}{2m}} $$

However, this is still not the general case since the hypothesis $h$ returned by the learning algorithm is not fixed. Similar to the consistent case, we will try to obtain an upper bound on the generalization error for an inconsistent (but finite) hypothesis, i.e., we need to compute the probability that there exists some hypothesis $h \in H$ such that the generalization error of $h$ differs from its empirical error by a value greater than $\epsilon$. Again, using the union bound, we get

$$ P \leq \sum_{h \in H}Pr[|\hat{R}(h)-R(h)|>\epsilon] $$

Using the Hoeffdieng’s inequality, this becomes

$$ P \leq 2|H|\exp(-2m\epsilon^2) $$

Now equating the RHS with $\delta$, we can arrive at the result

$$ R(h) \leq \hat{R}(h) + \sqrt{\frac{\log |H| + \log \frac{2}{\delta}}{2m}} $$

Here it is interesting to note that for a fixed $\vert H\vert$, to attain the same guarantee as in the consistent case, a quadratically larger labeled sample is required. Let us now analyze the role of the size of hypothesis class. If we have a smaller $H$, the second term is reduced but the empirical error may increase, and vice versa. However, for the same empirical error, it is always better to go with the smaller hypothesis class, i.e., the famous *Occam’s Razor* principle.

*****

In this article, we looked at some generalization bounds in case of a finite hypothesis, using the PAC learning framework. In the next part, I will discuss some measures for infinite hypotheses, namely the Rademacher complexity, growth function, and the VC dimension.

This blog post is loosely based on notes made from Chapter 2 "The PAC Learning Framework" of *Foundations of Machine Learning*.