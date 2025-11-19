---
output-file: models.patchtst.html
title: PatchTST
---

The PatchTST model is an efficient Transformer-based model for
multivariate time series forecasting.

It is based on two key components: - segmentation of time series into
windows (patches) which are served as input tokens to Transformer -
channel-independence. where each channel contains a single univariate
time series.

**References**
 - [Nie, Y., Nguyen, N. H., Sinthong, P., &
Kalagnanam, J. (2022). “A Time Series is Worth 64 Words: Long-term
Forecasting with
Transformers”](https://arxiv.org/pdf/2211.14730.pdf)


![Figure 1. PatchTST.](imgs_models/patchtst.png)
*Figure 1. PatchTST.*

## PatchTST

::: neuralforecast.models.patchtst.PatchTST
    options:
      members:
        - fit
        - predict
      heading_level: 3
