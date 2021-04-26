---
layout: post
title: Understanding Word Vectors
tags: ["deep learning","natural language processing","representation learning"]
mathjax: true
---

*This article is a formal representation of my understanding of vector semantics, from course notes and reading reference papers and chapters from Jurafsky’s SLP book. I will be talking about sparse and dense vector semantics, including SVD, skip-gram, and GloVe. In many places, I will try to explain the ideas in language rather than equations (but I’ll provide links to derivations and stuff wherever it is absolutely essential, which is actually everywhere!).*

*****

> “You shall know a word by the company it keeps.”

In vision, images are represented by the corresponding RGB values (or values obtained from other filters), so they are essentially matrices of integers. Language was more arbitrary because traditionally there was no formal method (or globally accepted standard) for representing words with numerical values. Well, not until **word embeddings** came into the picture (no pun intended)!

What are embeddings, though? They are called so because words are essentially transformed into vectors by “embedding” them into a vector space. For this, we make use of the hypothesis that words which occur in similar context tend to have similar meaning, i.e., the meaning of a word can be inferred from the distribution around it. For this reason, these methods are also called “distributional” methods.

Word vectors may be sparse or dense. I’ll begin with sparse vectors and then describe dense ones.

*****

### Sparse vectors

#### Term-document and term-term matrix

Suppose we have a set of 1000 documents, consisting of a total of 5000 unique words. In a very naive fashion, we can simply count the number of occurrences of each word in every document, and then represent each word by this 1000-dimensional vector of counts. This is exactly what a **term-document matrix** does.

Similarly, consider a large corpus of text with 5000 unique words. Now take a window of some fixed size and for each word pair, we count the number of times it occurs in the window. These counts form a **term-term matrix**, also called a **co-occurrence matrix** which in this case will be a 5000x5000 matrix (with most cells 0 if the window size is relatively small).

#### Pointwise Mutual Information (PMI)

The co-occurrence matrix is not the best measure of similarity between 2 words since it is based on the raw frequency, and hence is very skewed. Instead, it would be desirable to have a quantity which measures how much more likely is it for 2 words to occur in a window, compared with pure chance. This is exactly what PMI measures.

$$ \text{PMI}(x,y) = \log \left( \frac{P(x,y)}{P(x)P(y)} \right) $$

If PMI is positive, the ($x$,$y$) pair is more likely to occur together than pure chance, and vice versa. However, a negative value is unreliable since it is unlikely to get many co-occurrences of a word pair in a small corpus. To solve this problem, we define a Positive PMI (PPMI) as

$$ \text{PPMI}(x,y) = \max (\text{PMI}(x,y),0). $$

#### TF-IDF (Term frequency — inverse document frequency)

This is composed of 2 parts: TF, which denotes the count of the word in a document, and IDF, which is a weight component that gives higher weight to words occurring only in a few documents (and hence are more representative of the documents they are present in, in contrast to words like ‘the’ which are present in large number of documents).

$$ idf_i = \log \left( \frac{N}{df_i} \right) $$

Here, $N$ is the total number of documents and $df_i$ is the number of documents in which word $i$ occurs.

*****

### Dense vectors

The problem with sparse vectors is the curse of dimensionality, which makes computation and storage infeasible. For this reason, we prefer dense vectors, with real-valued elements. Dense vector semantics fall into 2 categories: matrix factorization, and neural embeddings.

*****

### Matrix Factorization

#### Singular vector decomposition (SVD)

This is basically a dimensionality reduction technique where we find the dimensions with the highest variances. Suppose we have the co-occurence matrix A of size $m \times n$, then it is possible to factorize A into:

$$ A_{m \times n} = U_{m\times r}S_{r\times r}V_{r\times n}^T $$

where $r$ is the rank of matrix $A$ (i.e. $r$ = maximum number of linearly independent vectors that can be used to form $A$). Also, $U$ is a matrix of the eigenvectors of $AA^T)$ and $S$ is a diagonal matrix comprising its eigenvalues. If we rearrange the columns in $U$ to correspond with a decreasing order of eigenvalues, we can keep the first $k$ columns which will represent the dimensions in the latent space which have the highest variance. These will give us a $k$-dimensional representation for each of the $m$ words in the vocabulary.

But why do we want to perform this truncation?

* First, removing the lower variance dimensions filters the noise component from the word embeddings.
* More importantly, having a lower number of parameters leads to better generalization. It is found that 300-dimensional word embeddings perform much better than, say, 3000-dimensional ones.

However, this approach is still constrained since the matrix factorization of $A$, which in itself may be a large matrix, is computationally complex.

*****

### Neural embeddings

The idea is simple. We can treat each element in the vector as a parameter to be updated while training a neural network model. We start with a randomly initialized vector and update it at each iteration. This update is based on the vectors of the context (window) words. The hypothesis is that such an update would ultimately result in similar words having vectors which are closer to each other in the vector space.

Here, I will describe the 2 most popular neural models — Word2Vec and GloVe.

#### Word2Vec

Word2Vec is actually the name of a tool which internally uses skip-gram or CBOW (continuous bag-of-words) with negative sampling. The objectives for both these models are quite similar, except a subtle distinction. In skip-gram, we predict the context words given the target word, and in CBOW, we predict the target word given the context words. In this article, I will limit my discussion to *skip-gram with negative sampling* (SGNS).

Suppose we have a context window where $w$ is the target word and $c$ is one of the context words. Then, skip-gram’s objective is to compute $P(c\vert w)$, which is given as

$$ p(c|w;\theta) = \frac{\exp(v_c\cdot v_w)}{\sum_{c^{\prime}\in C}\exp(v_{c^{\prime}}\cdot v_w)}. $$

Basically, it is just a softmax probability distribution over all the word-context pairs in the corpus, directed by the cosine similarity. However, the denominator term here is very expensive to compute since there may be a very large number of context words. To solve this problem, negative sampling is used.

Goldberg and Levy have explained the derivation for the objective function in SGNS very clearly in their [note](https://arxiv.org/pdf/1402.3722.pdf). I will try to provide a little intuition here.

For the word $w$, we are trying to predict the context word $c$. Since we are using softmax, this is essentially like a multi-class classification problem, where we are trying to classify the next word into one of $N$ classes (where $N$ is the number of words in the dictionary). Since $N$ may be quite large, this is a very difficult problem.

What SGNS does is that it converts this multi-classification problem into binary classification. The new objective is to predict, for any given word-context pair ($w$,$c$), whether the pair is in the window or not. For this, we try to increase the probability of a “positive” pair ($w$,$c$), while at the same time reducing the probability of $k$ randomly chosen “negative samples” ($w$,$s$) where $s$ is a word not found in $w$’s context. This leads to the following objective function which we try to maximize in SGNS:

$$ J = \log \sigma(c\cdot w) + \sum_{i=1}^k \mathbb{E}_{w_i \sim p(w)}[\log \sigma (-w_i \cdot w)]  $$

#### GloVe (Global Vectors)

One grievance with skip-gram and CBOW is that since they are both window-based models, the co-occurrence statistics of the corpus are not used efficiently, thereby resulting in suboptimal embeddings. The GloVe model proposed by Pennington et al. seeks to solve this problem by formulating an objective function from probability statistics.

Again, the original [paper](https://nlp.stanford.edu/pubs/glove.pdf) is very pleasant to read (section 3 describes their model in detail), and it is interesting to note the derivation for the objective function:

$$ J = \sum_{i,j=1}^V f(X_{ij})(w_i^Tw_j + b_i + b_j - \log X_{ij})^2 $$

Here, $X_{ij}$ is the count of the word pair ($i$,$j$) in the corpus. The weight function $f(x)$ has 3 requirements:

* $f(0) = 0$, so that the entire term does not tend to $\infty$.
* It should be non-decreasing to assign low weights to rare occurrences.
* It should be relatively small for large values of $x$.

Again, please read the paper for details.

*****

Although the matrix factorization approach and the neural embedding method may initially come off as completely independent, Levy and Goldberg (again!) ingeniously showed in a [NIPS 2014 paper](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) that even the SGNS method implicitly factorizes a word-context matrix where the cells are the PMI (pointwise mutual information) of the respective word-context pairs, shifted by a global context. They derive this in Section 3.1 of the paper, and I urge you to go to the link and read it. It’s a delight! The derivation is really simple and I would have done it here, except that I would only be reproducing the exact proof.

Very recently, Richard Socher’s group at Salesforce Research have proposed a new kind of embeddings called CoVe (Contextualized Word Vectors) in their paper. The idea is again borrowed from vision, where transfer learning has been used for a long time. Basically, models with various objectives are trained on a large dataset such as ImageNet, and then these weights are used to initialize model parameters for various vision tasks. Similarly, CoVe uses parameters trained on a attentional Seq2Seq machine translation task, and then uses it for various other tasks, including question-answering, where it has shown state-of-the-art performance on the SQuAD dataset. I have only skimmed through the paper, but I suppose such a deep transfer learning is naturally the next step towards improving word embeddings.

*****

As an aside, there is a series of blog posts by Sanjeev Arora that analyzes the theory of semantic embeddings in great detail. There are 3 posts in the series:

1.  [Semantic word embeddings](https://www.offconvex.org/2015/12/12/word-embeddings-1/)
2.  [Word Embeddings: Explaining their properties](https://www.offconvex.org/2016/02/14/word-embeddings-2/)
3.  [Linear algebraic structure of word embeddings](https://www.offconvex.org/2016/07/10/embeddingspolysemy/)

These provide great insight into the mathematics behind word vectors, and are beautifully written (which is no surprise since Prof. Arora is one of the authors of the famous and notoriously advanced book on Computational Complexity).