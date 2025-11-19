---
output-file: common.modules.html
title: NN Modules
---


## 1. MLP

Multi-Layer Perceptron

::: neuralforecast.common._modules.MLP
    options:
      members: []
      heading_level: 3

## 2. Temporal Convolutions

For long time in deep learning, sequence modelling was synonymous with
recurrent networks, yet several papers have shown that simple
convolutional architectures can outperform canonical recurrent networks
like LSTMs by demonstrating longer effective memory.

**References**

-[van den Oord, A., Dieleman, S., Zen, H., Simonyan,
K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A. W., &
Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio.
Computing Research Repository, abs/1609.03499. URL:
http://arxiv.org/abs/1609.03499.
arXiv:1609.03499.](https://arxiv.org/abs/1609.03499)

-[Shaojie Bai,
Zico Kolter, Vladlen Koltun. (2018). An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling. Computing
Research Repository, abs/1803.01271. URL:
https://arxiv.org/abs/1803.01271.](https://arxiv.org/abs/1803.01271)

::: neuralforecast.common._modules.Chomp1d
    options:
      members: []
      heading_level: 3

### CausalConv1d

::: neuralforecast.common._modules.CausalConv1d
    options:
      members: []
      heading_level: 3

### TemporalConvolutionEncoder

::: neuralforecast.common._modules.TemporalConlutionEncoder
    options:
      members: []
      heading_level: 3


## 3. Transformers

**References**

- [Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai
Zhang, Jianxin Li, Hui Xiong, Wancai Zhang. “Informer: Beyond Efficient
Transformer for Long Sequence Time-Series
Forecasting”](https://arxiv.org/abs/2012.07436)

- [Haixu Wu, Jiehui
Xu, Jianmin Wang, Mingsheng Long.](https://arxiv.org/abs/2106.13008)

::: neuralforecast.common._modules.TransEncoder
    options:
      heading_level: 3

::: neuralforecast.common._modules.TransEncoderLayer
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.TransDecoder
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.TransDecoderLayer
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.AttentionLayer
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.FullAttention
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.TriangularCausalMask
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.DataEmbedding_inverted
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.DataEmbedding
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.TemporalEmbedding
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.FixedEmbedding
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.TimeFeatureEmbedding
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.PositionalEmbedding
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.SeriesDecomp
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.MovingAvg
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.RevIN
    options:
      members: []
      heading_level: 3

::: neuralforecast.common._modules.RevINMultivariate
    options:
      members: []
      heading_level: 3
