---
layout: post
title: Irony Detection in Tweets
tags: ["natural language processing","representation learning"]
mathjax: true
---

There was a [SemEval 2018 Shared Task](https://github.com/Cyvhee/SemEval2018-Task3) on “irony detection in tweets” that ended recently. As a fun personal project, I thought of giving it a shot, just to implement some new ideas. In this post, I will describe my approach for the problem along with some code.

#### Problem description

The task itself was divided into two subtasks:

1.  *Task A: Binary classification*. Given a tweet, detect whether it has irony or not.
2.  *Task B: Multi-label classification*. Given a tweet and a set of labels: i) verbal irony realized through a polarity contrast, ii) verbal irony without such a polarity contrast (i.e., other verbal irony), iii) descriptions of situational irony, iv) non-irony, find the correct irony type.

While the task appears to be a simple text classification job, there are several nuances that make it challenging. Irony is often context-dependent or derived from world knowledge. In sentiment analysis, the semantics of the sentences are sufficient to judge whether the sentence has been spoken in a positive or negative manner. However, irony, by definition, almost always exists when the literal meaning of the sentence is dramatically different from what has been implied. Sample this:

> Just great when you’re (sic) mobile bill arrives by text.

From a sentiment analysis perspective, the presence of the phrase “just great” would adjudge this sentence strongly positive. However, from our world knowledge, we know the nuances of the interplay between a “mobile bill” and “text.” As a human, then, we can judge that the sentence is spoken in irony.

The problem is: how can we have an automated system understand this?

#### Circular correlation between text and hashtags

The first idea of a solution came from how the dataset was generated in the first place. To mine tweets containing irony, those tweets were selected which contained the hashtag **#not**. The idea was that a lot of people explicitly declare their intent at irony through hashtags. For instance, consider the following tweet:

> Physical therapy at 8 am is just what I want to be doing with my Friday #iwanttosleep

In this example, let us breakdown the tweet into 2 components:

1.  *Text*: Physical therapy at 8 am is just what I want to be doing with my Friday.
2.  *Hashtag*: I want to sleep

It is obvious from the semantics of the 2 components that they imply very different things. As such, it may help to model the interaction between the “text” and “hashtag” components of the tweet and then use the resulting embedding for classification. In this regard, we are essentially treating the problem as that of relation classification, where the entities are the 2 components and we need to identify whether there exists a relation between them (task A), and if yes, of which type (task B).

The problem, now, is reduced to the issue of how to model the two components and their interaction. This is where deep learning comes into the picture.

#### Modeling embeddings and interaction

The embeddings to represent the components are obtained simply by passing their pretrained word vectors through a bidirectional LSTM layer. This is fairly simple for the text component.

However, in the hashtag component, a single hashtag almost always consists of multiple words concatenated into a single string. Therefore, we first perform word segmentation on the hashtag and use the resulting segments to obtain the embedding.

```python
import wordsegment as ws
ws.load()
hashtag = “ “.join(ws.segment(temp))
## Here, 'temp' is the original hashtag
```

Once the embeddings for the two components have been obtained, we use the circular cross-correlation technique (which I have earlier described in [this blog post]({% post_url 2017-12-06-beyond-euclidean-embeddings %}) to model their interaction.  Essentially, the operator is defined as

$$ [a\cdot b]_k = \sum_{i=1}^{d-1}a_i b_{(k+i)\text{mod}d}. $$

In Tensorflow, this is implemented as follows:

```python
import tensorflow as tf

def holographic_merge(inp):
    [a, b] = inp
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')
```

The output of this merge is then passed to an XGBoost classifier (whose implementation was used out-of-the-box from the corresponding Python package).

This model resulted in a validation accuracy of ~62%, compared to ~59% for a simple LSTM model. Time to analyze where it was failing!

#### World knowledge for irony detection

The problem with this idea was that although it performed well for samples similar to the example given above, such samples constituted only about 20% of the dataset. For a majority of the tweets containing irony, there was no hashtag, and as such, modeling interactions was useless.

In such cases, we have to solely rely upon the text component to detect hashtag, for e.g.

> The fun part about 4 am drives in the winter, is no one has cleaned the snow yet

If an automated system has to understand that the above sentence contains irony, it needs to know that there is nothing fun about driving on a road covered in snow. This knowledge cannot be gained from learning on a few thousand tweets. We now turn to **transfer learning**!

MIT researchers recently built an unsupervised system called [DeepMoji](https://deepmoji.mit.edu/) for emoji prediction in tweets. According to the website, "DeepMoji has learned to understand emotions and sarcasm based on millions of emojis. We hypothesize that if we use this pretrained model to extract features from the text component, it may then be used to predict whether the text contains irony. In a way, we are transfering world knowledge to our model (assuming that the million tweets on which DeepMoji was trained is our world!).

As expected, concatenating the DeepMoji features with the holographic embeddings resulted in a validation accuracy of $\sim69\%$, i.e., a jump of almost 7%. This reinforces our hypothesis that world knowledge is indeed an important ingredient in any kind of irony detection.

#### Summary

In essence, we identified 2 aspects that were essential to identify irony in
tweets:

1.  Semantic interaction between text and hashtags, modeled using holographic embeddings
2.  World knowledge about irony in text, obtained through transfer learning from DeepMoji

The code for the project is available [here](https://github.com/desh2608/tweet-irony-detection).

**Disclaimer:** In the final test phase, the results were disappointing (~50% for task A) especially given the high performance on validation set. This could likely have been due to some implementation error on the test set, and we are waiting for the gold labels to be released to analyze our mistake.