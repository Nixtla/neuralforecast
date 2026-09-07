from neuralforecast.auto import AutoiTransformer, RayOptions, iTransformer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_itransformer_model(suppress_warnings):
    check_model(iTransformer, ["airpassengers"])

def test_autoitransformer(setup_dataset):

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoiTransformer, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoiTransformer.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 16})
        return config

    model = AutoiTransformer(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoiTransformer.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['hidden_size'] = 16
    model = AutoiTransformer(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, ray_options=RayOptions(cpus=1))
    model.fit(dataset=setup_dataset)


def test_itransformer_futr_exog(suppress_warnings):
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import generate_series

    assert iTransformer.EXOGENOUS_FUTR

    # Multivariate test with future exogenous
    n_series = 2
    h = 12
    input_size = 24

    df = generate_series(
        n_series=n_series,
        n_temporal_features=2,
        seed=42,
        freq="D",
        equal_ends=True,
        max_length=60,
    )
    max_ds = df.ds.max() - pd.Timedelta(h, "D")
    train_df = df[df.ds < max_ds]
    test_df = df[df.ds >= max_ds]

    model = iTransformer(
        h=h,
        input_size=input_size,
        n_series=n_series,
        futr_exog_list=["temporal_0", "temporal_1"],
        max_steps=5,
        val_check_steps=5,
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    preds = nf.predict(futr_df=test_df)

    assert not preds.isnull().values.any()
    assert len(preds) == n_series * h

    # Univariate test with future exogenous
    df_uni = generate_series(
        n_series=1,
        n_temporal_features=1,
        seed=42,
        freq="D",
        equal_ends=True,
        max_length=60,
    )
    max_ds_uni = df_uni.ds.max() - pd.Timedelta(h, "D")
    train_uni = df_uni[df_uni.ds < max_ds_uni]
    test_uni = df_uni[df_uni.ds >= max_ds_uni]

    model_uni = iTransformer(
        h=h,
        input_size=input_size,
        n_series=1,
        futr_exog_list=["temporal_0"],
        max_steps=5,
        val_check_steps=5,
    )
    nf_uni = NeuralForecast(models=[model_uni], freq="D")
    nf_uni.fit(df=train_uni)
    preds_uni = nf_uni.predict(futr_df=test_uni)

    assert not preds_uni.isnull().values.any()
    assert len(preds_uni) == 1 * h
