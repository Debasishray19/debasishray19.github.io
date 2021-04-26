---
layout: post
title: Beyond Euclidean Embeddings
tags: ["representation learning"]
mathjax: true
---

Representation learning, as the name suggests, seeks to learn representations for structures such as images, videos, words, sentencences, graphs, etc., which may then be used for several objectives. Arguably the most important representations used nowadays are word embeddings, usually learnt using the distributional semantics methods such as skip-gram or GloVe. I have previously written about these methods [here]({% post_url 2017-09-28-understanding-word-vectors %}).

Two assumptions are inherent while using these methods to learn word vectors:

1.  That words are best visualized as points in the $n$-dimensional space.
2.  That the Euclidean distance or the Euclidean dot product are the best measures of similarity between words (or other structures for which the embeddings have been learnt).

Over the last couple years, researchers have sought to challenge both of these assumptions by proposing several new non-Euclidean representations for words and graphs. Especially in the case of learning relational embeddings, the model should be able to learn all combinations of properties, namely reflexivity/irreflexivity, symmetry/anti-symmetry, and transitivity. Euclidean dot products are limited in that they cannot handle anti-symmetry, since dot products are commutative.

In this post, I will discuss 4 non-Euclidean embeddings: Gaussian, Holographic, Complex, and Poincare.

*****

#### Word representations via Gaussian embeddings

The key idea in this ICLR ’15 paper[^1] is to map words to a density instead of a point. Density here is represented by a “potential function,” such as a Gaussian. The authors provide a nice recap of energy functions as a tool for learning word representations.

Essentially, any representation learning involves an energy function $E(x,y)$ which scores pairs of inputs and outputs. A loss function is then uses this energy function to quantify the difference between actual output and predicted output. In the case of skip-gram models, the energy function used is a dot product, and the loss function is a logistic regression. In this paper, the authors propose 2 kinds of energy functions (for symmetric and asymmetric similarity), and the loss function used is max margin as follows.

$$ L_m(w,c_p,c_n) = \max(0,m-E(w,c_p)+E(w,c_n)) $$

For a Gaussian distribution to model any word, a baseline approach may involve using the distribution around the word to compute and mean and variance. If a word $w$ occurs $N$ times in the corpus, the covariance of the distribution around $w$ is given as

$$ \sum_w = \frac{1}{NW}\sum_i^N \sum_j^W (c(w)_{ij})(c(w)_{ij}-w)^T $$

where W is the window size, and $w$ is the assumed mean. However, the distributions learned using this empirical approach do not possess some desired properties such as unsupervised entailment represented as inclusion between ellipsoids. To solve this, 2 energy functions are proposed.

**Method 1: Symmetric similarity**

This method just computes the inner product between the two distributions. It has been shown that the inner product of two normal distributions is again a normal distribution. Furthermore, we take the log of this value for two reasons. First, since we are dealing with ranking loss, taking the logarithm converts absolute values into relative values, which is easier to interpret. Second, it is numerically easier to deal with.

Furthermore, the energy function is shown to be of the form **log det A + const**. We can interpret the constant term as a regularizer that prevents us from decreasing the distance by only increasing joint variance. This combination pushes the means together while encouraging them to have more concentrated, sharply peaked distributions in order to have high energy.

**Method 2: Asymmetric similarity**

This method computes the energy function as the negative of the KL-divergence between the 2 distributions (negative because the KL-divergence returns a distance value and hence needs to be minimized to increase similarity). A low KL divergence from $x$ to $y$ indicates that we can encode $y$ easily as $x$, implying that $y$ entails (logically follows from) $x$.

The authors have further computed the gradients for each of the two energy functions, and they are easily expressible in terms of existing means and covariances.

*****

#### Poincare embeddings for hierarchical representations

This paper[^2] proposes embeddings in hyperbolic spaces, such as the Poincare sphere. Before we get into the method itself, I think it would be best to give a brief overview of hyperbolic geometry itself.

**Hyperbolic geometry**

In his book *Elements*, Euclid provided a rigourous framework for axioms, theorems and postulates for all geometrical knowledge at the time. He stated 5 axioms which were to be assumed true. The first 4 were quite self-evident, and were:

1. Any two points can be connected by a line. 
2. Any line segment can be extended indefinitely. 
3. Given a line segment, a circle can be drawn with center at one of the endpoints and radius equal to the length of the segment.
4. Any two right angles are congruent.

However, the fifth axiom, also known as Playfair’s axiom, is much less obvious.

*Playfair’s axiom*: Given a line L and a point P, there exists at most one line through P that is parallel to L.

Euclid himself wasn’t very fond of this axiom and his first 28 postulates depended only on the first 4 axioms, which are the “core” of Euclidean geometry. Even 2000 years after his death, mathematicians tried to derive the fifth axiom from the first 4. While using “proof by contradiction” for this purpose, they assumed the negation of the fifth axiom (Given a line L and a point P not on L, there are at least two distinct lines that can be drawn through P that are parallel to L) and tried to arrive at a contradiction. However, while the derived results were strange and very different from those in Euclidean geometry, they were consistent within themselves. This was a turning point in mathematics as such a bifurcation in geometry had never been expected before. The geometry that arose from these explorations is known as *hyperbolic geometry*.

With this knowledge, let us now look at how embeddings may be computed in this new model.

The Poincare sphere model of hyperbolic space is particularly suitable for representing hierarchies. Consider a knowledge base which can be visualized as a tree. For any branching factor *b*, the number of leaf nodes increases exponentially as the number of levels increases. If we try to replicate this construction in a Euclidean disk(sphere), it would not be possible since the area(volume) of a disk(sphere) increases only quadratically(cubically) with increase in radius. This requires that we increase the number of dimensions exponentially.

However, the Poincare sphere embeds such hierarchies easily: nodes that are exactly $l$ levels below the root are placed on a sphere in hyperbolic space with radius $r \propto l$ and nodes that are less than $l$ levels below the root are located within this sphere. This type of construction is possible as hyperbolic disc area and circle length grow exponentially with their radius. In the paper, the authors used a sphere instead of disk since more degrees of freedom implies better representation of latent hierarchies.

Distances in the hyperbolic space are given as

$$ d(u,v) = arcosh\left( 1 + 2\frac{\lVert u-v \rVert^2}{(1-\lVert u \rVert)^2(1-\lVert v \rVert)^2} \right) $$

Here, hierarchy is represented using the norm of the embedding, while similarity is mirrored in the norm of vector difference. Furthermore, the function is differentiable, which is good for gradient descent.

For optimization, the update term is the learning rate times the Riemannian gradient of the parameter. The Riemannian gradient itself is computed by taking the product of the Poincare ball matrix inverse (which is trivial to compute) with the Euclidean gradient (which depends on the gradients of the distance function). The loss function used in the paper is a softmax with negative sampling.

*****

#### Holographic embeddings for knowledge graphs

This and the next method seek to learn embeddings for relations within knowledge graphs, and the motivation for both is to have embeddings that allow asymmetric relations to be sufficiently represented. To achieve said objective, this AAAI ’16 paper[^3] employs circular correlations, while the next paper from ICML ’16[^4] uses complex embeddings.

Before describing the method, I will first describe the task. Given a set $E$ of entities and a set $P$ of relation types, the objective is to learn a characteristic function for each relation type that determines whether that relation exists between any two elements in $E$. The entities are referred to as the *subject* and the *object*.

The general approach is to approximate the characteristic function using a function that takes as input the relation vector, and the vectors corresponding to the subject and the object. Using a loss function such as log likelihood minimization with negative sampling, we can tune the parameters that describe the entity vectors and the relation type vector. This is similar to our earlier discussion on energy function optimization.

The catch here is that the characteristic function is supposed to output a scalar score (the probability of the relation), but the inputs to it are vectors. To convert the input to a scalar, the entity vectors are combined using a composition operator **o**(more on this later), and its dot product is taken with the relation type vector.

So the problem boils down to the choice of a good compositional operator. In the past, three different approaches have been taken for this problem.

1.  *Tensor product*: Take the outer product of the entity vectors. However, the resulting vector contains the square of the initial number of parameters, which may cause problems such as overfitting down the line. 
2.  *Concatenation, projection, and non-linearity*: The projection matrix is learned during training. However, due to the absence of interaction between features, the representation learnt is not rich enough, even though non-linearity is added.
3.  *Non-compositional methods*: In these approaches, the score is computed as the distance of the difference vector with the relation vector (e.g., TransE). 

Essentially, we want an operator which has cross-feature interactions without having the number of parameters explode. To this end, the authors propose the circular correlation operator, which is given as

$$ [a\cdot b]_k = \sum_{i=1}^{d-1}a_i b_{(k+i)\text{mod}d}. $$

The output contains as many parameters as the input vectors, while also capturing the interaction between the features. The function measures the covariance between embeddings at different dimension shifts, and the asymmetry stems from this circular correlation.

At this point, you may be wondering why a simple convolutional operator would not suffice. The answer is that convolution is a commutative function, while correlation is not. Again, the key lies in symmetry (or the lack of it)!

*****

#### Complex embeddings for link prediction

In the objective of predicting relations described earlier, we can think of the characteristic function as a function which takes as input a latent matrix **X** of scores and outputs the corresponding probability. This latent matrix is an $E \times E$ matrix since it contains the scores for every possible pair of entities. However, since the number of entitites may be very large, the problem we want to solve is that of matrix factorization.

This is similar to the singular value decomposition method for learning word vectors that I discussed in an earlier blog post. If we assume that an entity has only one unique representation, regardless of whether it occurs as subject or object, the matrix X can be factorized as

$$ X = EWE^{-1} $$

Since the entity vectors are complex in nature ($u$ = Re($u$) + $i$Im($u$)), the matrix factorization of $X$ may be either real or complex. But since the characteristic function returns a real output, we define $X$ as the Real part of the factorization. Now, our original objective is to learn $P(Y=1)$ for every $s-o$ pair, and we are trying to approximate this using the latent matrix $X$. In the case of binary relations (yes/no), $Y$ is essentially a sign matrix, and hence it is safe to assume that its “sign-rank” is low. 

*But what is a “sign-rank”?* It refers to the smallest rank of a real matrix having the same sign pattern as $Y$. The authors showed in an earlier paper that if the sign rank of $Y$ is low, the rank of Re($EWE^T$) is at most twice that of $Y$. While this is a good upper bound, the actual rank is often much lower than the rank of $Y$.

In the case of multi-relational data, each relation has a representation $w$ associated with it. The characteristic function then takes as input the relation type along with the subject and object, and computes the score based on a novel scoring function. This function has the following property: if $w$ is real, the characteristic function is symmetric, and if $w$ is imaginary, then it is anti-symmetric.

*****

While Euclidean embeddings are popular, they are in no way sufficient to represent all the complexities and hierarchies in language. These methods suggest that looking at non-Euclidean spaces for representation learning may be the way to go.


[^1]: Vilnis, Luke, and Andrew McCallum. “[Word representations via gaussian embedding](https://arxiv.org/pdf/1412.6623.pdf).” *arXiv preprint arXiv:1412.6623*(2014).

[^2]: Nickel, Maximilian, and Douwe Kiela. “[Poincare Embeddings for Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039.pdf).” *arXiv preprint arXiv:1705.08039* (2017).

[^3]: Nickel, Maximilian, Lorenzo Rosasco, and Tomaso A. Poggio. “[Holographic Embeddings of Knowledge Graphs](https://arxiv.org/pdf/1510.04935.pdf).” *AAAI*. 2016.

[^4]: Trouillon, Théo, et al. “[Complex embeddings for simple link prediction](http://proceedings.mlr.press/v48/trouillon16.pdf).” *International Conference on Machine Learning*. 2016.