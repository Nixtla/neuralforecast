---
description: >-
  BaseAuto class for hyperparameter optimization in NeuralForecast. Integrates Optuna, HyperOpt, Dragonfly through Ray for automated model tuning with cross-validation.
output-file: common.base_auto.html
title: Hyperparameter Optimization | NeuralForecast
---

Machine Learning forecasting methods are defined by many hyperparameters that
control their behavior, with effects ranging from their speed and memory
requirements to their predictive performance. For a long time, manual
hyperparameter tuning prevailed. This approach is time-consuming, **automated
hyperparameter optimization** methods have been introduced, proving more
efficient than manual tuning, grid search, and random search.<br/><br/> The
`BaseAuto` class offers shared API connections to hyperparameter optimization
algorithms like
[Optuna](https://docs.ray.io/en/latest/tune/examples/bayesopt_example.html),
[HyperOpt](https://docs.ray.io/en/latest/tune/examples/hyperopt_example.html),
[Dragonfly](https://docs.ray.io/en/releases-2.7.0/tune/examples/dragonfly_example.html)
among others through `ray`, which gives you access to grid search, bayesian
optimization and other state-of-the-art tools like
hyperband.

Comprehending the impacts of hyperparameters is still a
precious skill, as it can help guide the design of informed hyperparameter
spaces that are faster to explore automatically.

![](imgs_models/data_splits.png)
*Figure 1. Example of dataset split (left), validation (yellow) and test (orange). The hyperparameter optimization guiding signal is obtained from the validation set.*

## 

::: neuralforecast.common._base_auto.BaseAuto
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
      heading_level: 3
      show_root_heading: true
      show_source: true

### Usage Example

```python
class RayLogLossesCallback(tune.Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        result = trial.last_result
        print(40 * '-' + 'Trial finished' + 40 * '-')
        print(f'Train loss: {result["train_loss"]:.2f}. Valid loss: {result["loss"]:.2f}')
        print(80 * '-')
```


```python
config = {
    "hidden_size": tune.choice([512]),
    "num_layers": tune.choice([3, 4]),
    "input_size": 12,
    "max_steps": 10,
    "val_check_steps": 5
}
auto = BaseAuto(h=12, loss=MAE(), valid_loss=MSE(), cls_model=MLP, config=config, num_samples=2, cpus=1, gpus=0, callbacks=[RayLogLossesCallback()])
auto.fit(dataset=dataset)
y_hat = auto.predict(dataset=dataset)
assert mae(Y_test_df['y'].values, y_hat[:, 0]) < 200
```


```python
def config_f(trial):
    return {
        "hidden_size": trial.suggest_categorical('hidden_size', [512]),
        "num_layers": trial.suggest_categorical('num_layers', [3, 4]),
        "input_size": 12,
        "max_steps": 10,
        "val_check_steps": 5
    }

class OptunaLogLossesCallback:
    def __call__(self, study, trial):
        metrics = trial.user_attrs['METRICS']
        print(40 * '-' + 'Trial finished' + 40 * '-')
        print(f'Train loss: {metrics["train_loss"]:.2f}. Valid loss: {metrics["loss"]:.2f}')
        print(80 * '-')
```


```python
auto2 = BaseAuto(h=12, loss=MAE(), valid_loss=MSE(), cls_model=MLP, config=config_f, search_alg=optuna.samplers.RandomSampler(), num_samples=2, backend='optuna', callbacks=[OptunaLogLossesCallback()])
auto2.fit(dataset=dataset)
assert isinstance(auto2.results, optuna.Study)
y_hat2 = auto2.predict(dataset=dataset)
assert mae(Y_test_df['y'].values, y_hat2[:, 0]) < 200
```

### References

-   [James Bergstra, Remi Bardenet, Yoshua Bengio, and Balazs Kegl
    (2011). “Algorithms for Hyper-Parameter Optimization”. In: Advances
    in Neural Information Processing Systems. url:
    https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
-   [Kirthevasan Kandasamy, Karun Raju Vysyaraju, Willie Neiswanger,
    Biswajit Paria, Christopher R. Collins, Jeff Schneider, Barnabas
    Poczos, Eric P. Xing (2019). “Tuning Hyperparameters without Grad
    Students: Scalable and Robust Bayesian Optimisation with Dragonfly”.
    Journal of Machine Learning Research. url:
    https://arxiv.org/abs/1903.06694](https://arxiv.org/abs/1903.06694)
-   [Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh,
    Ameet Talwalkar (2016). “Hyperband: A Novel Bandit-Based Approach to
    Hyperparameter Optimization”. Journal of Machine Learning Research.
    url:
    https://arxiv.org/abs/1603.06560](https://arxiv.org/abs/1603.06560)

