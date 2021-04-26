---
layout: post
title: The Best Papers at ICLR 2017
tags: ["deep learning","conference summary"]
mathjax: true
---

The International Conference on Learning Representations (ICLR) has evolved into *the* deep learning conference over the last few years, and with its open review system, it is not difficult to understand why. I was recently going through some of the papers accepted at this year’s ICLR, especially the 3 that were awarded the Best Paper award. In this article, I will try to summarize these 3 papers in simple words, and hopefully get an idea about what’s hot in deep learning.

The 3 best papers are:

1.  [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf)
2.  [Making neural programming architectures generalize via recursion](https://arxiv.org/pdf/1704.06611.pdf)
3.  [Semi-supervised knowledge knowledge transfer for deep learning from private training data](https://arxiv.org/pdf/1610.05755.pdf)

Statisticians always like saying that deep learning is a black box and eveything that happens is a result of hyperparameter tuning. You cannot say why you have obtained a good result, let alone providing a guarantee for a result. Well, not anymore. The 3 best papers were all about providing proveable guarantees, and it looks like the deep learning community is all set to move past its “black box” days.

*****

#### Understanding deep learning requires rethinking generalization

This paper from Google Brain is very readable and discusses some very common occurences. The authors make 2 very straightforward observations: (i) Deep neural networks easily fit random labels, and (ii) Explicit regularization is neither necessary nor sufficient for controlling generalization error.

Essentially, they evaluate well-known deep architectures on several popular datasets, with some randomness added, such as random labels, or Gaussian noise-added labels, or shifted input features, in an attempt to show that even though training error is still close to negligible, generalization error increases alarmingly even in the presence of explicit regularizers such as weight decay, dropout, and data augmentation, and for implicit regularizers such as early stopping. This serves as a wake-up call for people who study the performance of neural networks.

The most interesting (and PROVABLE) guarantee that the paper contains is the following theorem: *There exists a two-layer neural network with ReLU activations and 2n+d weights that can represent any function on a sample of size n in d dimensions.* While I will not go into the detailed proof here, it is essentially based on solving the system of linear equations based on the ReLU activation function. For the system to have a solution, the coefficient matrix should be full-ranked, which the authors show is indeed the case. If the proof in the paper is too formal (read: succinct) for you, you can find a more detailed one [here](https://danieltakeshi.github.io/2017/05/19/understanding-deep-learning-requires-rethinking-generalization-my-thoughts-and-notes). In addition, they also show ways in which we can reduce the width of the network
at each layer by increasing its depth.

*****

#### Making neural programming architectures generalize via recursion

Understanding this paper from researchers at UC Berkeley requires a little background of neural programmer-interpreter (NPI) architectures, which can be found in the paper as well as in the [ICLR ’16 paper](https://arxiv.org/pdf/1511.06279.pdf) in which they were introduced. Basically, an NPI framework consists of a controller (such as an LSTM), which takes as input the environment state and arguments, and returns the next program pointer to be executed. In this way, given a set of sequences that an algorithm must follow to get to the output, the NPI can learn the algorithm itself. In the original paper, the authors learned to perform tasks such as adding, sorting,
etc. using NPIs.

In this paper, the concept of *recursion* is added to the existing NPI framework, and this makes it capable of performing much more complex tasks such as quick sort. Formally, a function exhibits recursive behavior when it possesses two properties: (1) Base cases — terminating scenarios that do not use recursion to produce answers; (2) A set of rules that reduces all other problems toward the base cases. In the paper, the author describe how they construct NPI training traces so as to make them contain recursive elements and thus enable NPI to learn recursive programs.

Furthermore, the authors show *provably perfect generalization* for their new architecture. The theorem states that for the same sequence of step inputs, the model produces the exact same step output as the target program it aims to learn.

To prove this, we again go to the notions of base case and recursive case. For example, in the addition task, the base case is always a set of small, fixed size step input sequences during which the LSTM state remains constant. So the base case is trivially true. The key in proving the recursive step is to construct the verification set well, so that the step inputs are neither too large so as to be outside the scope of evaluation, nor too small so that the semantics of the problem are not well defined. The actual verification process is simple and can be read in the paper.

*****

#### Semi-supervised knowledge transfer for deep learning from private training data

Another paper from Google Brain, this deals with the important subject of building a model which learns from sensitive data while also keeping it private. A new model called PATE (Private Aggregation of Teacher Ensembles) is introduced which basically trains in a 2-step strategy:

* An ensemble of teacher models is trained on disjoint subsets of the sensitive dataset.
* A student model is trained on the aggregate output of the ensemble.

The aggregation is “private” because the number of times the student can access the teacher model is limited, and the top vote of the ensemble is revealed only after adding random noise. Due to these restrictions, no amount of querying can get hold of the private training data used to train the teacher models. Furthermore, in the Laplacian noise added to the teacher aggregation, the noise
parameter can be used to tune the privacy-accuracy tradeoff for the student model. For example, if the noise is large, privacy is high at the cost of reduced accuracy, and vice versa.

For the transfer of knowledge from the ensemble to the student model, the authors experimented with various techniques and finally used semi-supervised learning with GANs. The student is trained on nonsensitive data (which may be labeled or unlabeled). The discriminator is a multi-class classifier which is trained such that it classifies the labeled data into the correct class, the unlabeled (true) data into any of the *k* classes, and the generated data (from the generator) into an extra class.

Again, the authors use the notion of “differentiable privacy guarantee” to come up with a lower bound for the privacy guarantee for their model. (The derivation is a little involved and I skipped it since I don’t have the prerequisites of security and privacy.)

*****

To sum up, all the papers seem to provide some generalization guarantee rather than just proposing a “good” model. Looks like sunny days for deep learning!