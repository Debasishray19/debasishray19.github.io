---
layout: post
title: Acoustic analysis using the finite difference time domain method
tags: ["computational acoustic","finite difference time domain method","FDTD"]
mathjax: true
---

The traditional finite difference time domain (FDTD) method has been used for modelling computational electrodynamics (i.e., finding approximate solutions to Maxwell's equations). The method uses staggered Yee grids (named after the applied mathematician [Kane S. Yee](https://en.wikipedia.org/wiki/Kane_S._Yee)[1]) to discretize the electric and magnetic field components in space and time. It then employs the finite-difference operator to numerically solve these field components for every grid cell at each step. This approach can also be extended to solve problems in other domains. Here, we will discuss the implementation of the FDTD scheme to approximate acoustic wave propagation in an open space.


### References
[1] Yee, K. (1966). Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media. IEEE Transactions on antennas and propagation, 14(3), 302-307.


### [BLEU (Bilingual Evaluation Understudy)](http://aclweb.org/anthology/P/P02/P02-1040.pdf)

This is by far the most popular metric for evaluating machine translation system. In BLEU, precision and recall are approximated by *modified n-gram precision* and *best match length*, respectively.

**Modified n-gram precision**: First, an n-gram precision is the fraction of n-grams in the candidate text which are present in any of the reference texts. From the example above, the unigram precision of A is 100%. However, just using this value presents a problem. For example, consider the two candidates:

> (i) He works on machine learning.

> (ii) He works on on machine machine learning learning.

Candidate (i) has a unigram precision of 60% while for (ii) it is 75%. However, it is obvious that (ii) is not a better candidate than (i) in any way. To solve this problem, we use a “modified” n-gram precision. It matches the candidate’s n-grams only as many times as they are present in any of the reference texts. So in the above example, (ii)’s unigrams ‘on’, ‘machine’, and ‘learning’ are matched only once, and the unigram precision becomes 37.5%.

Finally, to include all the n-gram precision scores in our final precision, we take their geometric mean. This is done because it has been found that precision decreases exponentially with *n*, and as such, we would require logarithmic averaging to represent all values fairly.

$$ \text{Precision} = \exp\left( \sum_{i=1}^N w_n \log p_n \right), ~~~~ \text{where}~~ w_n = \frac{1}{n} $$

**Best match length:** While precision calculation was relatively simple, the problem with recall is that there may be many reference texts. So it is difficult to calculate the sensitivity of the candidate with respect to a general reference. However, it is intuitive to think that a longer candidate text is more likely to contain a larger fraction of some reference than a shorter candidate. At the same time, we have already ensured that candidate texts are not arbitrarily long, since then their precision score would be low.

Therefore, we can introduce recall by just penalizing brevity in candidate texts. This is done by adding a multiplicative factor *BP* with the modified n-gram precision as follows.

$$ \text{BP} = \begin{cases} 1, &\text{if } c > r, \\ \exp(1-\frac{r}{c}),&\text{otherwise}.\end{cases} $$

Here, $c$ is the total length of candidate translation corpus, and $r$ is the effective reference length of corpus, i.e., average length of all references. The lengths are taken as average over the entire corpus to avoid harshly punishing the length deviations on short sentences. As the candidate length decreases, the ratio $\frac{r}{c}$ increases, and the BP decreases exponentially.

*****

### [ROUGE (Recall Oriented Understudy for Gisting Evaluation)](http://www.aclweb.org/anthology/W/W04/W04-1013.pdf)

As is clear from its name, ROUGE is based only on recall, and is mostly used for summary evaluation. Depending on the feature used for calculating recall, ROUGE may be of many types, namely ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S. Here, we describe the idea behind one of these, and then give a quick run-down of the
others.

**ROUGE-N:** This is based on n-grams. For example, ROUGE-1 counts recall based on matching unigrams, and so on. For any $n$, we count the total number of n-grams across all the reference summaries, and find out how many of them are present in the candidate summary. This fraction is the required metric value.

ROUGE-L/W/S are based on: longest common subsequence (LCS), weighted LCS, and skip-bigram co-occurence statistics, respectively. Instead of using only recall, these use an F-score which is the harmonic mean of precision and recall values. These are in turn, calculated as follows for ROUGE-L.

Suppose A and B are candidate and reference summaries of lengths $m$ and $n$ respectively. Then, we have

$$ P = \frac{LCS(A,B)}{m} ~~~~\text{and}~~~~ R = \frac{LCS(A,B)}{n}. $$

$F$ is then calculated as the weighted harmonic mean of P and R, as

$$ F = \frac{(1+b^2)RP}{R+b^2P}. $$

Similarly, in ROUGE-W, for calculating weighted LCS, we also track the lengths of the consecutive matches, in addition to the length of longest common subsequence (since there may be non-matching words in the middle). In ROUGE-S, a skip-bigram refers to any pair of words in sentence order allowing for arbitrary gaps. The precision and recall, in this case, are computed as a ratio of the total number of possible bigrams, i.e., ${n \choose 2}$.

*****

### [METEOR (Metric for Evaluation for Translation with Explicit Ordering)](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)

METEOR is another metric for machine translation evaluation, and it claims to have better correlation with human judgement.

So why do we need a new metric when BLEU is already available? The problem with BLEU is that since the *BP* value uses lengths which are averaged over the entire corpus, so the scores of individual sentences take a hit.

To solve this problem, METEOR modifies the precision and recall computations, replacing them with a weighted F-score based on mapping unigrams and a penalty function for incorrect word order.

**Weighted F-score:** First, we try to find the largest subset of mappings that can form an alignment between the candidate and reference translations. For this, we look at exact matches, followed by matches after Porter stemming, and finally using WordNet synonymy. After such an alignment is found, suppose $m$ is
the number of mapped unigrams between the two texts. Then, precision and recall are given as $\frac{m}{c}$ and $\frac{m}{r}$, where $c$ and $r$ are candidate and reference lengths, respectively. F is calculated as

$$ F = \frac{PR}{\alpha P + (1-\alpha)R}. $$

**Penalty function:** To account for the word order in the candidate, we introduce a penalty function as

$$ p = \gamma \left( \frac{c}{m} \right)^{\beta},~~~~ \text{where}~~ 0 \leq \gamma \leq 1. $$

Here, $c$ is the number of matching chunks and $m$ is the total number of matches. As such, if most of the matches are contiguous, the number of chunks is lower and the penalty decreases. Finally, the METEOR score is given as $(1-p)F$.

*****

*The links to the original papers for the methods described here are in the section headings. Readers are advised to refer to them for details. I have tried to outline the main ideas here for a quick review.*
