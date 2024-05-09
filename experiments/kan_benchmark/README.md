# KAN for Forecasting - Benchmark on M3 and M4 datasets

[Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs) is an alternative to the multilayer perceptron (MLP). In this experiment, we assess the performance of KANs in forecasting time series.

We use the M3 and M4 datasets, which represents more than 102 000 unique time series covering yearly, quarterly, monthly, weekly, daily and hourly frequencies.

While KANs reduce the number of parameters by 38% to 92% compared to the MLP, it also rarely performs better than the MLP in time series forecasting tasks. In this benchmark, N-BEATS still performs best across the vast majority of datasets.

The detailed results are shown in the table below.

| Dataset       | Model  | MAE      | sMAPE     |
|---------------|--------|----------|-----------|
| M3 - Yearly   | KAN    | 1206     | 9.70%     |
|               | MLP    | <u>1111</u>     | <u>8.70%</u>     |
|               | NBEATS | **1027** | **8.40%** |
| M3 - Quarterly| KAN    | 565      | 5.20%     |
|               | MLP    | **540**| **5.00%** |
|               | NBEATS | <u>542</u>  | **5.00%** |
| M3 - Monthly  | KAN    | 676      | 7.50%     |
|               | MLP    | <u>652</u>| <u>7.20%</u>|
|               | NBEATS | **636**  | **7.10%** |
| M4 - Yearly   | KAN    | <u>875</u>| <u>7.20%</u>|
|               | MLP    | 921      | 7.30%     |
|               | NBEATS | **854**  | **6.90%** |
| M4 - Quarterly| KAN    | <u>602</u>| <u>5.30%</u>|
|               | MLP    | <u>602</u>| <u>5.30%</u>|
|               | NBEATS | **588**  | **5.10%** |
| M4 - Monthly  | KAN    | 607| 7.00%     |
|               | MLP    | <u>596</u>      | <u>6.80%</u>|
|               | NBEATS | **584**  | **6.70%** |
| M4 - Weekly   | KAN    | <u>341</u>| <u>4.70%</u>|
|               | MLP    | 375      | 5.00%     |
|               | NBEATS | **313**  | **4.00%** |
| M4 - Daily    | KAN    | 194      | 1.60%|
|               | MLP    | <u>189</u>      | <u>1.60%</u>|
|               | NBEATS | **176**  | **1.50%** |
| M4 - Hourly   | KAN    | **267**  | **7.10%**     |
|               | MLP    | 315      | 7.80%     |
|               | NBEATS | <u>280</u>  | <u>7.40%</u> |
<br>

## Reproducibility

1. Create a conda environment `kan_benchmark` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate long_horizon
  ```

4. Run the experiments for each dataset and each model using the notebook `kan-experiment.ipynb`.
<br>

## References
-[Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark - "KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756)