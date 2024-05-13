# KAN for Forecasting - Benchmark on M3 and M4 datasets

[Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs) is an alternative to the multilayer perceptron (MLP). In this experiment, we assess the performance of KANs in forecasting time series.

We use the M3 and M4 datasets, which represents more than 102 000 unique time series covering yearly, quarterly, monthly, weekly, daily and hourly frequencies.

While KANs reduce the number of parameters by 38% to 92% compared to the MLP, it also rarely performs better than the MLP in time series forecasting tasks. In this benchmark, N-BEATS still performs best across the vast majority of datasets.

The detailed results are shown in the table below.

| Dataset       | Model  | MAE         | sMAPE (%)      | time (s)  |
|---------------|--------|-------------|----------------|-----------|
| M3 - Yearly   | KAN    | 1206        | 9.74           | 23        |
|               | MLP    | 1111        | 8.68           | 9         |
|               | NBEATS | **1027**    | **8.35**       | 11        |
|               | NHITS  | <u>1087</u> | <u>8.36</u>    | 14        |
| M3 - Quarterly| KAN    | 565         | 5.19           | 45        |
|               | MLP    | **540**     | <u>4.99</u>    | 10        |
|               | NBEATS | <u>542</u>  | **4.97**       | 26        |
|               | NHITS  | 573         | 5.29           | 26        |
| M3 - Monthly  | KAN    | 676         | 7.55           | 38        |
|               | MLP    | <u>653</u>  | 7.19           | 20        |
|               | NBEATS | **637**     | <u>7.11</u>    | 24        |
|               | NHITS  | **637**     | **7.08**       | 35        |
| M4 - Yearly   | KAN    | 875         | 7.20           | 132       |
|               | MLP    | 921         | 7.37           | 51        |
|               | NBEATS | <u>855</u>  | **6.87**       | 62        |
|               | NHITS  | **852**     | <u>6.88</u>    | 73        |
| M4 - Quarterly| KAN    | 603         | 5.36           | 121       |
|               | MLP    | 602         | 5.35           | 40        |
|               | NBEATS | **588**     | **5.15**       | 49        |
|               | NHITS  | <u>591</u>  | <u>5.19</u>    | 61        |
| M4 - Monthly  | KAN    | 607         | 7.00           | 215       |
|               | MLP    | <u>594</u>  | <u>6.80</u>    | 150       |
|               | NBEATS | **584**     | **6.70%**      | 131       |
|               | NHITS  | **584**     | **6.70**       | 173       |
| M4 - Weekly   | KAN    | 341         | 4.70           | 34        |
|               | MLP    | 375         | 5.00%          | 22        |
|               | NBEATS | **313**     | **4.00**       | 18        |
|               | NHITS  | <u>329</u>  | <u>4.40</u>    | 21        |
| M4 - Daily    | KAN    | 194         | 1.60           | 53        |
|               | MLP    | <u>189</u>  | <u>1.60</u>    | 24        |
|               | NBEATS | **176**     | **1.50**       | 43        |
|               | NHITS  | **176**     | **1.50**       | 51        |
| M4 - Hourly   | KAN    | **267**     | **7.10**       | 33        | 
|               | MLP    | 315         | 7.80           | 10        |
|               | NBEATS | <u>280</u>  | <u>7.40</u>    | 18        |
|               | NHITS  | 302         | 6.95           | 23        |
<br>

## Reproducibility

1. Create a conda environment `kan_benchmark` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate kan_benchmark
  ```

4. Run the experiments using:<br>
- `--dataset` parameter in `['M3-yearly', 'M3-quarterly', 'M3-monthly', 'M4-yearly', 'M4-quarterly', 'M4-monthly', 'M4-weekly', 'M4-daily', 'M4-hourly']`<br>

```shell
python run_experiment.py --dataset 
```
<br>

## References
-[Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark - "KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756)