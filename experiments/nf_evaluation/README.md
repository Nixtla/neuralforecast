# Comprehensive Evaluation of Neuralforecast models

In this experiment, we tested all available models in Neuralforecast on benchmark datasets to evaluate their speed and forecasting performance.

The datasets used for this benchmark are:
- M4 (yearly)
- M4 (quarterly)
- M4 (monthly)
- M4 (daily)
- Ettm2 (15 min)
- Electricity (hourly)
- Weather (10 min)
- Traffic (hourly)
- ILI (weekly)

Each model went through hyperparameter optimization. The test was completed locally on CPU.

The table below summarizes the results
*Table will be updated as results are obtained*

<br>

## Reproducibility

1. Create a conda environment `nf_evaluation` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate nf_evaluation
  ```

Alternatively simply installing neuralforecast and datasetsforecast with pip may suffice:
```
pip install git+https://github.com/Nixtla/datasetsforecast.git
pip install git+https://github.com/Nixtla/neuralforecast.git
```

4. Run the experiments for each dataset and each model using with 
- `--dataset` parameter in `[M4-yearly, M4-quarterly, M4-monthly, M4-daily, Ettm2, Electricity, Weather, Traffic, ILI]`
- `--model` parameter in `['AutoLSTM', 'AutoRNN', 'AutoGRU', 'AutoDilatedRNN', 'AutoDeepAR', 'AutoTCN', 'AutoMLP', 'AutoNBEATS', 'AutoNHITS', 'AutoDLinear', 'AutoTFT', 'AutoVanillaTransformer', 'AutoInformer', 'AutoAutoformer', 'AutoFEDformer', 'AutoTimesNet', 'AutoPatchTST']`
<br>

```shell
python run_experiments.py --dataset M4-yearly --model AutoMLP
```

5. The script creates a folder `results/<dataset>` which contains a CSV file with the metrics for the specified model