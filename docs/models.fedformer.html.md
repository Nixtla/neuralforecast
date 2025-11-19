---
output-file: models.fedformer.html
title: FEDformer
---

The FEDformer model tackles the challenge of finding reliable
dependencies on intricate temporal patterns of long-horizon forecasting.

The architecture has the following distinctive features: - In-built
progressive decomposition in trend and seasonal components based on a
moving average filter. - Frequency Enhanced Block and Frequency Enhanced
Attention to perform attention in the sparse representation on basis
such as Fourier transform. - Classic encoder-decoder proposed by Vaswani
et al. (2017) with a multi-head attention mechanism.

The FEDformer model utilizes a three-component approach to define its
embedding: - It employs encoded autoregressive features obtained from a
convolution network. - Absolute positional embeddings obtained from
calendar features are utilized.

**References**
 - [Zhou, Tian, Ziqing Ma, Qingsong Wen, Xue Wang,
Liang Sun, and Rong Jin.. “FEDformer: Frequency enhanced decomposed
transformer for long-term series
forecasting”](https://proceedings.mlr.press/v162/zhou22g.html)


![Figure 1. FEDformer Architecture.](imgs_models/fedformer.png)
*Figure 1. FEDformer
Architecture.*

## FEDformer

::: neuralforecast.models.fedformer.FEDformer
    options:
      members:
        - fit
        - predict
      heading_level: 3
