---
layout: post
title: A note on MFCCs and delta features
tags: ["speech-recognition","kaldi","feature extraction"]
mathjax: true
---

### What are MFCCs and how are they computed?

Feature extraction is the first step in any automatic speech recognition (ASR) pipeline. The objective is to compute features from speech waveforms which contain relevant information about the linguistic content of the speech, and ignore information about the background noise, emotions, etc. This problem has been extensively studied since early days in ASR research, and several feature extraction methods have been proposed. Among these, the most well-known and widely used are Mel Frequency Cepstral Coefficients (MFCCs).

Since MFCCs are very well known, I will only briefly describe their computation in this post. Most of this is taken from [this blog](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/), which explains them in some detail. The key steps for computing MFCCs are described below.

1. First, the entire waveform is divided into shorter segments of 20-40 ms each. The assumption is that in this short segment, the signal is statistically stationary, and so features can be assumed to be constant inside this window. In Kaldi and most major ASR systems, windows are 25 ms in length and at 10 ms intervals apart, i.e., they are overlapping.

2. In order to recognize the frequencies present in this short segment, the power spectrum (or the [periodogram](https://en.wikipedia.org/wiki/Periodogram) estimate) is computed. This is done using discrete-time Fourier transforms.

3. It is difficult to distinguish individual frequencies in the raw power spectrum, especially in the high frequency range. To solve this problem, the spectrum is convolved with several (20-40, in general) triangular Mel filters, called a filterbank. These filters are narrow at low frequency and get wider as frequency increases, in accordance with the human cochlea. Furthermore, a log transform is applied since humans don't perceive loudness on a linear scale. 

4. Since filterbank energies are correlated and cannot be used directly with a Gaussian mixture with diagonal covariance, we apply a [discrete cosine transform (DCT)](https://en.wikipedia.org/wiki/Discrete_cosine_transform) to decorrelate them.

There is some debate in the community regarding the use of the DCT, instead of directly using the log Mel fiterbank features, particularly for deep neural network based acoustic models. Some research groups, like Google, use filterbanks (fbanks) while Kaldi mostly uses MFCCs, especially in its TDNN chain models. Here is [Dan Povey](https://www.danielpovey.com/)'s take on this:

> The reason we use MFCC is because they are more easily compressible, being decorrelated; we dump them to disk
with compression to 1 byte per coefficient.  But we dump all the coefficients, so it's equivalent to filterbanks times a full-rank matrix, no information is lost. 

> (Source: [kaldi-help](https://groups.google.com/forum/#!topic/kaldi-help/_7hB74HKhC4))

### Delta and delta-delta features

The idea behind using delta (differential) and delta-delta (acceleration) coefficients is that in order to recognize speech better, we need to understand the dynamics of the power spectrum, i.e., the trajectories of MFCCs over time. The delta coeffients are computed using the following formula.

$$ d_t = \frac{\sum_{n=1}^N n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^N n^2}, $$
where $d_t$ is a delta coefficient from frame $t$ computed in terms of the static coefficients $c_{t-n}$ to $c_{t+n}$. $n$ is usually taken to be 2. The acceleration coefficients are computed similarly, but using the differential instead of the static coefficients.

### The LDA transform in Kaldi

> For a comprehensive reference on LDA, readers are advised to refer to [this post](https://sebastianraschka.com/Articles/2014_python_lda.html#what-is-a-good-feature-subspace).

The latest TDNN-based chain models in Kaldi (see, for example, [this recipe](https://github.com/kaldi-asr/kaldi/blob/06442e1870996486cb052fdd89d63aac44144b87/egs/wsj/s5/local/chain/tuning/run_tdnn_1g.sh#L188)) do not use differential and acceleration features (hereby refered to as "delta features" for convenience). Instead, they employ an LDA-like transformation which is essentially an affine transformation of the spliced input. Here is a sample from the xconfig of a typical Kaldi TDNN model:

```bash
input dim=100 name=ivector
input dim=40 name=input
# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
```


This splicing can be over 1 or 2 frames on either side of the central frame, i.e. `Append(-1,0,1)` or `Append(-2,-1,0,1,2)`. Additionally, [i-vectors](https://ieeexplore.ieee.org/document/5545402) are appended with the spliced input before the LDA. Although Kaldi itself has an [implementation of the LDA transform](https://kaldi-asr.org/doc/transform.html#transform_lda) available, the transformation here simply multiplies the spliced input with a full-rank matrix. This is why this is called an "LDA-like", and not an LDA transform.

### Some new results

In some sense, this LDA-like transform is a generalization of using the delta features, since it can apply arbitrary scaling to each coefficient, and this matrix is learned in the training stage. However, this means having to additionally learn $(k \times n+d)^2$ parameters, where $k$ is the splicing window, $n$ is the MFCC size, and $d$ is the i-vector dimensionality. For typical values of $k$, $n$, and $d$, this is in the range of 50000 to 90000 parameters. While this is not a "huge" number compared to the size of modern deep networks (a typical TDNN model in Kaldi may have up to 10 million parameters), we would still like to see if this is disposable.

I replaced the LDA transform with simple delta features. In the context of our input, the differential is simply $c_{t+1} - c_{t-1}$, and the acceleration is $c_{t-2} + c_{t+2} - 2\times c_t$. This is implemented using a new `xconfig` layer called `delta-layer` as follows.

```python
class XconfigDeltaLayer(XconfigLayerBase):
    """This class is for parsing lines like
     'delta-layer name=delta input=idct'
    which appends the central frame with the delta features
    (i.e. -1,0,1 since scale equals 1) and delta-delta features 
    (i.e. 1,0,-2,0,1), and then applies batchnorm to it.
    Parameters of the class, and their defaults:
      input='[-1]'             [Descriptor giving the input of the layer]
    """
    def __init__(self, first_token, key_to_value, prev_names=None):
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input': '[-1]'}

    def check_configs(self):
        pass

    def output_name(self, auxiliary_output=None):
        assert auxiliary_output is None
        return self.name

    def output_dim(self, auxiliary_output=None):
        assert auxiliary_output is None
        input_dim = self.descriptors['input']['dim']
        return (3*input_dim)

    def get_full_config(self):
        ans = []
        config_lines = self._generate_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in this layer
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    def _generate_config(self):
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.output_dim()

        configs = []
        line = ('dim-range-node name={0}_copy1 input-node={0} dim={1} dim-offset=0'.format(
            input_desc, input_dim))
        configs.append(line)
        line = ('dim-range-node name={0}_copy2 input-node={0} dim={1} dim-offset=0'.format(
            input_desc, input_dim))
        configs.append(line)

        line = ('component name={0}_2 type=NoOpComponent dim={1}'.format(
            input_desc, output_dim))
        configs.append(line)
        line = ('component-node name={0}_2 component={0}_2 input=Append(Offset({0},0),'
            ' Sum(Offset(Scale(-1.0,{0}_copy1),-1), Offset({0},1)), Sum(Offset({0},-2), Offset({0},2),' 
            ' Offset(Scale(-2.0,{0}_copy2),0)))'.format(input_desc))
        configs.append(line)

        line = ('component name={0} type=BatchNormComponent dim={1}'.format(
            self.name, output_dim))
        configs.append(line)
        line = ('component-node name={0} component={0} input={1}_2'.format(
            self.name, input_desc))
        configs.append(line)
        return configs
```

The following are some experimental results on `mini_librispeech`, `wsj` (Wall Street Journal), and `swbd` (Switchboard). The i-vector scale was reduced for `mini_librispeech` since the delta features are computed on top of a [SpecAugment](https://arxiv.org/abs/1904.08779) layer, which itself includes batch normalization. Therefore, using an i-vector scale of 1.0 would overpower the MFCCs.

| Setup            | Test set   | IDCT | SpecAugment | i-vector scale | LDA  | Delta |
|------------------|------------|------|-------------|----------------|------|-------|
| mini_librispeech | dev_clean2 | Y    | Y           | 0.5            | 7.54 | 7.66  |
| wsj              | eval92     | Y    | N           | 1.0            | 2.39 | 2.41  |
| swbd             | rt03       | N    | N           | 1.0            | 15.0 | 15.0  |

These results are for a particular test set for these setups, and for a specific decoder, but the general trend of results is found to be the same across all test set and decoder combinations. Without significant loss in performance, we can eliminate the need of an LDA transform in the network. Work on a [pull request](https://github.com/kaldi-asr/kaldi/pull/3490/files) for this setup is in progress. 