import numpy as np
import torch as t

from neuralforecast.losses.numpy import (
    mae,  # unscaled errors
    mape,  # percentage errors
    mase,  # scaled error
    mqloss,
    mse,
    quantile_loss,  # probabilistic errors
    rmse,
    smape,
)
from neuralforecast.losses.pytorch import (
    MAE,  # unscaled errors
    MAPE,  # percentage errors
    MASE,  # scaled error
    MSE,
    RMSE,
    SMAPE,
    MQLoss,
    QuantileLoss,  # probabilistic errors
)


# Test class for pytorch/numpy loss functions
class TestLoss:
    @classmethod
    def setup_class(self):
        self.num_quantiles = np.random.randint(3, 10)
        self.first_num = np.random.randint(1, 300)
        self.second_num = np.random.randint(1, 300)

        self.y = t.rand(self.first_num, self.second_num)
        self.y_hat = t.rand(self.first_num, self.second_num)
        self.y_hat2 = t.rand(self.first_num, self.second_num)
        self.y_hat_quantile = t.rand(self.first_num, self.second_num, self.num_quantiles)

        self.quantiles = t.rand(self.num_quantiles)
        self.q_float = np.random.random_sample()

    def test_mae(self):
        mae_numpy   = mae(self.y, self.y_hat)
        mae_pytorch = MAE()
        mae_pytorch = mae_pytorch(self.y, self.y_hat).numpy()
        np.testing.assert_allclose(mae_numpy, mae_pytorch, rtol=1e-6, atol=1e-6)

    def test_mse(self):
        mse_numpy   = mse(self.y, self.y_hat)
        mse_pytorch = MSE()
        mse_pytorch = mse_pytorch(self.y, self.y_hat).numpy()
        np.testing.assert_allclose(mse_numpy, mse_pytorch, rtol=1e-6, atol=1e-6)

    def test_rmse(self):
        rmse_numpy   = rmse(self.y, self.y_hat)
        rmse_pytorch = RMSE()
        rmse_pytorch = rmse_pytorch(self.y, self.y_hat).numpy()
        np.testing.assert_allclose(rmse_numpy, rmse_pytorch, rtol=1e-6, atol=1e-6)

    def test_mape(self):
        mape_numpy   = mape(y=self.y, y_hat=self.y_hat)
        mape_pytorch = MAPE()
        mape_pytorch = mape_pytorch(y=self.y, y_hat=self.y_hat).numpy()
        np.testing.assert_allclose(mape_numpy, mape_pytorch, rtol=1e-6, atol=1e-6)


    def test_smape(self):
        smape_numpy   = smape(self.y, self.y_hat)
        smape_pytorch = SMAPE()
        smape_pytorch = smape_pytorch(self.y, self.y_hat).numpy()
        np.testing.assert_allclose(smape_numpy, smape_pytorch, rtol=1e-6, atol=1e-6)

    # def test_mase(self):
    #    y_insample = t.rand(self.first_num, self.second_num)
    #    seasonality = 24
    #    # Hourly 24, Daily 7, Weekly 52
    #    # Monthly 12, Quarterly 4, Yearly 1
    #    mase_numpy   = mase(y=self.y, y_hat=self.y_hat,
    #                        y_insample=y_insample, seasonality=seasonality)
    #    mase_object  = MASE(seasonality=seasonality)
    #    mase_pytorch = mase_object(y=self.y, y_hat=self.y_hat,
    #                               y_insample=y_insample).numpy()
    #    np.testing.assert_array_almost_equal(mase_numpy, mase_pytorch, decimal=2)

    #def test_rmae(self):
    #    rmae_numpy   = rmae(self.y, self.y_hat, self.y_hat2)
    #    rmae_object  = RMAE()
    #    rmae_pytorch = rmae_object(self.y, self.y_hat, self.y_hat2).numpy()
    #    self.assertAlmostEqual(rmae_numpy, rmae_pytorch, places=4)

    def test_quantile(self):
        quantile_numpy = quantile_loss(self.y, self.y_hat, q = self.q_float)
        quantile_pytorch = QuantileLoss(q = self.q_float)
        quantile_pytorch = quantile_pytorch(self.y, self.y_hat).numpy()
        np.testing.assert_array_almost_equal(quantile_numpy, quantile_pytorch, decimal=6)

    # def test_mqloss(self):
    #     weights = np.ones_like(self.y)

    #     mql_np_w = mqloss(self.y, self.y_hat_quantile, self.quantiles, weights=weights)
    #     mql_np_default_w = mqloss(self.y, self.y_hat_quantile, self.quantiles)

    #     mql_object = MQLoss(quantiles=self.quantiles)
    #     mql_py_w = mql_object(y=self.y,
    #                           y_hat=self.y_hat_quantile,
    #                           mask=t.Tensor(weights)).numpy()

    #     print('self.y.shape', self.y.shape)
    #     print('self.y_hat_quantile.shape', self.y_hat_quantile.shape)
    #     mql_py_default_w = mql_object(y=self.y,
    #                                   y_hat=self.y_hat_quantile).numpy()

    #     weights[0,:] = 0
    #     mql_np_new_w = mqloss(self.y, self.y_hat_quantile, self.quantiles, weights=weights)
    #     mql_py_new_w = mql_object(y=self.y,
    #                               y_hat=self.y_hat_quantile,
    #                               mask=t.Tensor(weights)).numpy()

    #     np.testing.assert_array_almost_equal(mql_np_w,  mql_np_default_w)
    #     np.testing.assert_array_almost_equal(mql_py_w,  mql_py_default_w)
    #     np.testing.assert_array_almost_equal(mql_np_new_w,  mql_py_new_w)



