


__all__ = ['ACTIVATIONS', 'MLP', 'Chomp1d', 'CausalConv1d', 'TemporalConvolutionEncoder', 'TransEncoderLayer', 'TransEncoder',
           'TransDecoderLayer', 'TransDecoder', 'AttentionLayer', 'TriangularCausalMask', 'FullAttention',
           'PositionalEmbedding', 'TokenEmbedding', 'TimeFeatureEmbedding', 'FixedEmbedding', 'TemporalEmbedding',
           'DataEmbedding', 'DataEmbedding_inverted', 'MovingAvg', 'SeriesDecomp', 'RevIN', 'RevINMultivariate']


import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = [
    "ReLU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "PReLU",
    "Sigmoid",
    "GELU",
    "ELU",
    "SiLU",
    "Mish",
    "GLU",
    "Softsign",
    "Hardshrink",
    "Softshrink",
    "Threshold",
    "RReLU",
    "CELU",
    "LogSigmoid",
    "Hardtanh",
    "Hardswish",
    "Identity",
]


class MLP(nn.Module):
    """Multi-Layer Perceptron Class

    Args:
        in_features (int): Dimension of input.
        out_features (int): Dimension of output.
        activation (str): Activation function to use.
        hidden_size (int): Dimension of hidden layers.
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout rate.
    """

    def __init__(
        self, in_features, out_features, activation, hidden_size, num_layers, dropout
    ):
        super().__init__()
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        self.activation = getattr(nn, activation)()

        # MultiLayer Perceptron
        # Input layer
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.activation,
            nn.Dropout(dropout),
        ]
        # Hidden layers
        for i in range(num_layers - 2):
            layers += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                self.activation,
                nn.Dropout(dropout),
            ]
        # Output layer
        layers += [nn.Linear(in_features=hidden_size, out_features=out_features)]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Chomp1d(nn.Module):
    """Chomp1d

    Receives `x` input of dim [N,C,T], and trims it so that only
    'time available' information is used.
    Used by one dimensional causal convolutions `CausalConv1d`.

    Args:
        horizon (int): Length of outsample values to skip.
    """

    def __init__(self, horizon):
        super(Chomp1d, self).__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, :, : -self.horizon].contiguous()


class CausalConv1d(nn.Module):
    """Causal Convolution 1d

    Receives `x` input of dim [N,C_in,T], and computes a causal convolution
    in the time dimension. Skipping the H steps of the forecast horizon, through
    its dilation.
    Consider a batch of one element, the dilated convolution operation on the
    $t$ time step is defined:

    $\mathrm{Conv1D}(\mathbf{x},\mathbf{w})(t) = (\mathbf{x}_{[*d]} \mathbf{w})(t) = \sum^{K}_{k=1} w_{k} \mathbf{x}_{t-dk}$

    where $d$ is the dilation factor, $K$ is the kernel size, $t-dk$ is the index of
    the considered past observation. The dilation effectively applies a filter with skip
    connections. If $d=1$ one recovers a normal convolution.

    Args:
        in_channels (int): Dimension of `x` input's initial channels.
        out_channels (int): Dimension of `x` outputs's channels.
        activation (str): Identifying activations from PyTorch activations.
        select from 'ReLU','Softplus','Tanh','SELU', 'LeakyReLU','PReLU','Sigmoid'.
        padding (int): Number of zero padding used to the left.
        kernel_size (int): Convolution's kernel size.
        dilation (int): Dilation skip connections.

    Returns:
        torch.Tensor: Torch tensor of dim [N,C_out,T] activation(conv1d(inputs, kernel) + bias).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        dilation,
        activation,
        stride: int = 1,
    ):
        super(CausalConv1d, self).__init__()
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.chomp = Chomp1d(padding)
        self.activation = getattr(nn, activation)()
        self.causalconv = nn.Sequential(self.conv, self.chomp, self.activation)

    def forward(self, x):
        return self.causalconv(x)


class TemporalConvolutionEncoder(nn.Module):
    """Temporal Convolution Encoder

    Receives `x` input of dim [N,T,C_in], permutes it to  [N,C_in,T]
    applies a deep stack of exponentially dilated causal convolutions.
    The exponentially increasing dilations of the convolutions allow for
    the creation of weighted averages of exponentially large long-term memory.

    Args:
        in_channels (int): Dimension of `x` input's initial channels.
        out_channels (int): Dimension of `x` outputs's channels.
        kernel_size (int): Size of the convolving kernel.
        dilations (int list): Controls the temporal spacing between the kernel points.
        activation (str): Identifying activations from PyTorch activations.
        select from 'ReLU','Softplus','Tanh','SELU', 'LeakyReLU','PReLU','Sigmoid'.

    Returns:
        torch.Tensor: Torch tensor of dim [N,T,C_out].
    """

    # TODO: Add dilations parameter and change layers declaration to for loop
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilations,
        activation: str = "ReLU",
    ):
        super(TemporalConvolutionEncoder, self).__init__()
        layers = []
        for dilation in dilations:
            layers.append(
                CausalConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    activation=activation,
                    dilation=dilation,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # [N,T,C_in] -> [N,C_in,T] -> [N,T,C_out]
        x = x.permute(0, 2, 1).contiguous()
        x = self.tcn(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class TransEncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        hidden_size,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(TransEncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=conv_hidden_size, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class TransEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TransEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TransDecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        hidden_size,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(TransDecoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size, out_channels=conv_hidden_size, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TransDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(TransDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, hidden_size, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_heads)
        d_values = d_values or (hidden_size // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.value_projection = nn.Linear(hidden_size, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hidden_size)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask:
    """
    TriangularCausalMask
    """

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if not self.output_attention:  # flash attention not supported
            q = queries.permute(0, 2, 1, 3)  # [B, H, L, E]
            k = keys.permute(0, 2, 1, 3)
            v = values.permute(0, 2, 1, 3)

            scale = self.scale or 1.0 / math.sqrt(E)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=(
                    attn_mask.mask[:, 0] if (self.mask_flag and attn_mask) else None
                ),
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=scale,
            )
            V = attn_output.permute(0, 2, 1, 3).contiguous()
            return (V, None) if self.output_attention else (V, None)
        else:
            scale = self.scale or 1.0 / math.sqrt(E)
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

            return (
                (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)
            )


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, hidden_size):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=hidden_size,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        return self.embed(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model, dtype=torch.float32, requires_grad=False)
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(
        self, c_in, exog_input_size, hidden_size, pos_embedding=True, dropout=0.1
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, hidden_size=hidden_size)

        if pos_embedding:
            self.position_embedding = PositionalEmbedding(hidden_size=hidden_size)
        else:
            self.position_embedding = None

        if exog_input_size > 0:
            self.temporal_embedding = TimeFeatureEmbedding(
                input_size=exog_input_size, hidden_size=hidden_size
            )
        else:
            self.temporal_embedding = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):

        # Convolution
        x = self.value_embedding(x)

        # Add positional (relative withing window) embedding with sines and cosines
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        # Add temporal (absolute in time series) embedding with linear layer
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding(x_mark)

        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    DataEmbedding_inverted
    """

    def __init__(self, c_in, hidden_size, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate hidden_size]
        return self.dropout(x)


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.MovingAvg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.MovingAvg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):
    """RevIN (Reversible-Instance-Normalization)"""

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        """
        Args:
            num_features (int): The number of features or channels
            eps (float): A value added for numerical stability
            affine (bool): If True, RevIN has learnable affine parameters
            substract_last (bool): If True, the substraction is based on the last value
                               instead of the mean in normalization
            non_norm (bool): If True, no normalization performed.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class RevINMultivariate(nn.Module):
    """
    ReversibleInstanceNorm1d for Multivariate models
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones((1, 1, self.num_features)))
        self.affine_bias = nn.Parameter(torch.zeros((1, 1, self.num_features)))

    def _normalize(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=1, keepdim=True).detach()
        self.batch_std = torch.sqrt(
            torch.var(x, axis=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        # Reverse the normalization
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight

        x = x * self.batch_std
        x = x + self.batch_mean

        return x
