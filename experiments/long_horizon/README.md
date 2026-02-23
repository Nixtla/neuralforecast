# Long Horizon Forecasting Experiments with NHITS

In these experiments we use `NHITS` on the [ETTh1, ETTh2, ETTm1, ETTm2](https://github.com/zhouhaoyi/ETDataset) benchmark datasets.

| Dataset  | Horizon  | NHITS-MSE  | NHITS-MAE  | TIDE-MSE   | TIDE-MAE   |
|----------|----------|------------|------------|------------|------------|
| ETTh1    | 96       | 0.378      | 0.393      | 0.375      | 0.398      |
| ETTh1    | 192      | 0.427      | 0.436      | 0.412      | 0.422      |
| ETTh1    | 336      | 0.458      | 0.484      | 0.435      | 0.433      |
| ETTh1    | 720      | 0.561      | 0.501      | 0.454      | 0.465      |
|----------|----------|------------|------------|------------|------------|
| ETTh2    | 96       | 0.274      | 0.345      | 0.270      | 0.336      |
| ETTh2    | 192      | 0.353      | 0.401      | 0.332      | 0.380      |
| ETTh2    | 336      | 0.382      | 0.425      | 0.360      | 0.407      |
| ETTh2    | 720      | 0.625      | 0.557      | 0.419      | 0.451      |
|----------|----------|------------|------------|------------|------------|
| ETTm1    | 96       | 0.302      | 0.35       | 0.306      | 0.349      |
| ETTm1    | 192      | 0.347      | 0.383      | 0.335      | 0.366      |
| ETTm1    | 336      | 0.369      | 0.402      | 0.364      | 0.384      |
| ETTm1    | 720      | 0.431      | 0.441      | 0.413      | 0.413      |
|----------|----------|------------|------------|------------|------------|
| ETTm2    | 96       | 0.176      | 0.255      | 0.161      | 0.251      |
| ETTm2    | 192      | 0.245      | 0.305      | 0.215      | 0.289      |
| ETTm2    | 336      | 0.295      | 0.346      | 0.267      | 0.326      |
| ETTm2    | 720      | 0.401      | 0.413      | 0.352      | 0.383      |
|----------|----------|------------|------------|------------|------------|
<br>

## Reproducibility

1. Create a conda environment `long_horizon` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate long_horizon
  ```

Alternatively simply installing neuralforecast and datasetsforecast with pip may suffice:
```
pip install git+https://github.com/Nixtla/datasetsforecast.git
pip install git+https://github.com/Nixtla/neuralforecast.git
```

4. Run the experiments for each dataset and each model using with 
- `--horizon` parameter in `[96, 192, 336, 720]`
- `--dataset` parameter in `['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']`
<br>

```shell
python run_nhits.py --dataset 'ETTh1' --horizon 96 --num_samples 20
```

You can access the final forecasts from the `./data/{dataset}/{horizon}_forecasts.csv` file. Example: `./data/ETTh1/96_forecasts.csv`.
<br><br>

## References
-[Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski (2023). "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting". Accepted at the Thirty-Seventh AAAI Conference on Artificial Intelligence.](https://arxiv.org/abs/2201.12886)