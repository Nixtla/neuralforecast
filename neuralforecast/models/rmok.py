# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.rmok.ipynb.

# %% auto 0
__all__ = ['WaveKANLayer', 'TaylorKANLayer', 'JacobiKANLayer', 'RMoK']

# %% ../../nbs/models.rmok.ipynb 6
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.pytorch import MAE
from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate
from typing import Optional

# %% ../../nbs/models.rmok.ipynb 8
class WaveKANLayer(nn.Module):
    """This is a sample code for the simulations of the paper:
    Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

    https://arxiv.org/abs/2405.12832
    and also available at:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
    We used efficient KAN notation and some part of the code:+

    """

    def __init__(
        self,
        in_features,
        out_features,
        wavelet_type="mexican_hat",
        with_bn=True,
        device="cpu",
    ):
        super(WaveKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.with_bn = with_bn

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # self.weight1 is not used; you may use it for weighting base activation and adding it like Spl-KAN paper
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == "mexican_hat":
            term1 = (x_scaled**2) - 1
            term2 = torch.exp(-0.5 * x_scaled**2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
                wavelet
            )
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == "morlet":
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled**2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
                wavelet
            )
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == "dog":
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled**2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
                wavelet
            )
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == "meyer":
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(
                    v <= 1 / 2,
                    torch.ones_like(v),
                    torch.where(
                        v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))
                    ),
                )

            def nu(t):
                return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
                wavelet
            )
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == "shannon":
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(
                x_scaled.size(-1),
                periodic=False,
                dtype=x_scaled.dtype,
                device=x_scaled.device,
            )
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
                wavelet
            )
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        # base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        # Apply batch normalization
        if self.with_bn:
            return self.bn(combined_output)
        else:
            return combined_output

# %% ../../nbs/models.rmok.ipynb 10
class TaylorKANLayer(nn.Module):
    """
    https://github.com/Muyuzhierchengse/TaylorKAN/
    """

    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded**i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y

# %% ../../nbs/models.rmok.ipynb 12
class JacobiKANLayer(nn.Module):
    """
    https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py
    """

    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )

        nn.init.normal_(
            self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if (
            self.degree > 0
        ):  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (
                (2 * i + self.a + self.b)
                * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1)
                * (self.a * self.a - self.b * self.b)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1)
                * (i + self.b - 1)
                * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[
                :, :, i - 1
            ].clone() - theta_k2 * jacobi[
                :, :, i - 2
            ].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum(
            "bid,iod->bo", jacobi, self.jacobi_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

# %% ../../nbs/models.rmok.ipynb 14
class RMoK(BaseModel):
    """Reversible Mixture of KAN


    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `n_series`: int, number of time-series.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `taylor_order`: int, order of the Taylor polynomial.<br>
    `jacobi_degree`: int, degree of the Jacobi polynomial.<br>
    `wavelet_function`: str, wavelet function to use in the WaveKAN. Choose from ["mexican_hat", "morlet", "dog", "meyer", "shannon"]<br>
    `dropout`: float, dropout rate.<br>
    `revin_affine`: bool=False, bool to use affine in RevIn.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References**<br>
    - [Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang, Zhe Wu."KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?". arXiv.](https://arxiv.org/abs/2408.11306)<br>
    """

    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        n_series: int,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        taylor_order: int = 3,
        jacobi_degree: int = 6,
        wavelet_function: str = "mexican_hat",
        dropout: float = 0.1,
        revine_affine: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs
    ):

        super(RMoK, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            stat_exog_list=None,
            futr_exog_list=None,
            hist_exog_list=None,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs
        )

        self.input_size = input_size
        self.h = h
        self.n_series = n_series
        self.dropout = nn.Dropout(dropout)
        self.revin_affine = revine_affine

        self.taylor_order = taylor_order
        self.jacobi_degree = jacobi_degree
        self.wavelet_function = wavelet_function

        self.experts = nn.ModuleList(
            [
                TaylorKANLayer(
                    self.input_size,
                    self.h * self.loss.outputsize_multiplier,
                    order=self.taylor_order,
                    addbias=True,
                ),
                JacobiKANLayer(
                    self.input_size,
                    self.h * self.loss.outputsize_multiplier,
                    degree=self.jacobi_degree,
                ),
                WaveKANLayer(
                    self.input_size,
                    self.h * self.loss.outputsize_multiplier,
                    wavelet_type=self.wavelet_function,
                ),
                nn.Linear(self.input_size, self.h * self.loss.outputsize_multiplier),
            ]
        )

        self.num_experts = len(self.experts)
        self.gate = nn.Linear(self.input_size, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.rev = RevINMultivariate(self.n_series, affine=self.revin_affine)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        B, L, N = insample_y.shape
        x = self.rev(insample_y, "norm")
        x = self.dropout(x).transpose(1, 2).reshape(B * N, L)

        score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.num_experts)], dim=-1
        )

        y_pred = (
            torch.einsum("BLE, BE -> BL", expert_outputs, score)
            .reshape(B, N, self.h * self.loss.outputsize_multiplier)
            .permute(0, 2, 1)
        )
        y_pred = self.rev(y_pred, "denorm")
        y_pred = y_pred.reshape(B, self.h, -1)

        return y_pred
