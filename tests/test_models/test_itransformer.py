import pandas as pd
import pytest

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoiTransformer, RayOptions, iTransformer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.utils import generate_series

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


@pytest.mark.parametrize("n_series, n_temporal_features", [(1, 1), (2, 3)])
@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize("loss", [MAE(), MQLoss(level=[10, 90])])
def test_itransformer_futr_exog(suppress_warnings, n_series, n_temporal_features, use_norm, loss):
    assert iTransformer.EXOGENOUS_FUTR

    h = 12
    input_size = 24

    df = generate_series(
        n_series=n_series,
        n_temporal_features=n_temporal_features,
        seed=42,
        freq="D",
        equal_ends=True,
        max_length=60,
    )
    max_ds = df.ds.max() - pd.Timedelta(h, "D")
    train_df = df[df.ds < max_ds]
    test_df = df[df.ds >= max_ds]

    futr_exog_list = [f"temporal_{i}" for i in range(n_temporal_features)]

    model = iTransformer(
        h=h,
        input_size=input_size,
        n_series=n_series,
        futr_exog_list=futr_exog_list,
        use_norm=use_norm,
        loss=loss,
        max_steps=5,
        val_check_steps=5,
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    preds = nf.predict(futr_df=test_df)

    assert not preds.isnull().values.any()
    assert len(preds) == n_series * h

    # Assert predictions change when futr_df changes to ensure futr_tokens are not silently dropped
    # We loop over each temporal column, gracefully altering it to avoid silent token dropping
    for col in futr_exog_list:
        test_df_changed = test_df.assign(**{col: test_df[col].astype(float) + 1.0})
        preds_changed = nf.predict(futr_df=test_df_changed)
        assert not preds_changed.equals(preds), f"Predictions did not change when {col} was modified!"

