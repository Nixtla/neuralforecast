


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
    """Multi-Layer Perceptron for time series forecasting.  

    A feedforward neural network with configurable depth and width. The network  
    consists of an input layer, multiple hidden layers with activation functions  
    and dropout, and an output layer. All hidden layers have the same dimensionality.  

    Args:  
        in_features (int): Dimension of input features.  
        out_features (int): Dimension of output features.  
        activation (str): Activation function name. Must be one of the supported  
            activations in ACTIVATIONS list (e.g., 'ReLU', 'Tanh', 'GELU', 'ELU').  
        hidden_size (int): Number of units in each hidden layer. All hidden layers  
            share the same dimensionality.  
        num_layers (int): Total number of layers including input and output layers.  
            Must be at least 2. For example, num_layers=3 creates: input layer,  
            one hidden layer, and output layer.  
        dropout (float): Dropout probability applied after each hidden layer's  
            activation. Should be in range [0.0, 1.0]. Not applied to output layer.

    Returns:
        (torch.Tensor): Transformed output tensor of shape [..., out_features].

    Notes:  
        - The activation function is applied after each hidden layer's linear  
          transformation, but not after the final output layer.  
        - Dropout is applied after activation in hidden layers for regularization.  
        - This MLP is used as a decoder component in various forecasting models  
          including RNN, LSTM, GRU, DilatedRNN, TCN, and xLSTM.  
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
    """Temporal trimming layer for 1D sequences.  

    Removes the rightmost `horizon` timesteps from a 3D tensor. This is commonly  
    used to trim padding added by convolution operations, ensuring the output  
    sequence has the desired length.  

    The operation trims the temporal dimension: [N, C, T] -> [N, C, T-horizon]  

    Args:  
        horizon (int): Number of timesteps to remove from the end of the  
            temporal dimension.  
    
    Returns:
        (torch.Tensor): Trimmed tensor of shape [N, C, T-horizon].

    Notes:  
        - Commonly used in `CausalConv1d` to remove padding after convolution.
    """

    def __init__(self, horizon):
        super(Chomp1d, self).__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, :, : -self.horizon].contiguous()


class CausalConv1d(nn.Module):
    r"""Causal Convolution 1d

    Receives `x` input of dim [N,C_in,T], and computes a causal convolution
    in the time dimension. Skipping the H steps of the forecast horizon, through
    its dilation.
    Consider a batch of one element, the dilated convolution operation on the
    $t$ time step is defined:

    ```math
    \mathrm{Conv1D}(\mathbf{x},\mathbf{w})(t) = (\mathbf{x}_{[*d]} \mathbf{w})(t) = \sum^{K}_{k=1} w_{k} \mathbf{x}_{t-dk}
    ```

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
        (torch.Tensor): Torch tensor of dim [N,C_out,T] activation(conv1d(inputs, kernel) + bias).
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
    """Temporal Convolution Encoder for sequence modeling.

    Applies a deep stack of exponentially dilated causal convolutions to capture
    long-range temporal dependencies. The encoder receives input in [N, T, C_in]
    format, permutes it to [N, C_in, T] for convolution operations, and applies
    multiple causal convolutional layers with exponentially increasing dilations.
    This architecture enables efficient modeling of exponentially large receptive
    fields, creating weighted averages over long-term historical patterns.

    Args:
        in_channels (int): Number of input channels (features) in the time series.
        out_channels (int): Number of output channels after convolution operations.
            All layers in the stack output this dimensionality.
        kernel_size (int): Size of the convolving kernel for each causal convolution
            layer. Determines the local temporal window.
        dilations (list of int): List of dilation factors for each convolution layer.
            Controls the temporal spacing between kernel points. For example,
            [1, 2, 4, 8] creates 4 layers with exponentially increasing receptive fields.
        activation (str, optional): Activation function name. Must be one of the
            supported activations in ACTIVATIONS list (e.g., 'ReLU', 'Tanh', 'GELU').

    Returns:
        (torch.Tensor): Encoded output tensor of shape [N, T, C_out] after applying
            all causal convolution layers and permuting back to time-first format.

    Notes:
        - Input is automatically permuted from [N, T, C_in] to [N, C_in, T] before
          convolution and permuted back to [N, T, C_out] after processing.
        - Each layer uses CausalConv1d with padding to ensure causality (no future
          information leakage).
        - Exponentially increasing dilations allow the receptive field to grow
          exponentially with network depth, enabling efficient long-range modeling.
        - Used as an encoder component in TCN (Temporal Convolutional Network) models.
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
    """Transformer Encoder Layer.  

    A single layer of the transformer encoder that applies self-attention followed by   
    a position-wise feed-forward network with residual connections and layer normalization.   
    Dropout is applied after the self-attention output and twice in the feed-forward network   
    (after each convolution) before the residual connections for regularization.  

    Args:  
        attention (AttentionLayer): Self-attention mechanism to apply.  
        hidden_size (int): Dimension of the model's hidden representations.  
        conv_hidden_size (int, optional): Dimension of the feed-forward network's hidden layer.  
            Defaults to 4 * hidden_size if not specified.  
        dropout (float, optional): Dropout probability applied after attention and feed-forward layers.  
        activation (str, optional): Activation function to use in the feed-forward network.  
            Either "relu" or "gelu".  

    Returns:
        (torch.Tensor): Output tensor of shape [batch, seq_len, hidden_size] after
          applying self-attention and feed-forward transformations.
        (torch.Tensor or None): Attention weights of shape [batch, n_heads, seq_len, seq_len]
          if output_attention is True in the attention layer, otherwise None.

    Notes:
        The layer applies two main operations in sequence:
        1. Self-attention on the input with dropout, residual connection, and normalization
        2. Position-wise feed-forward network using 1D convolutions with dropout applied twice
           (after the first convolution with activation, and after the second convolution),
           residual connection, and normalization

        This layer is used as a building block in transformer-based models like Informer,
        VanillaTransformer, iTransformer, and SOFTS.
    """

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
    """Transformer Encoder.  

    A stack of transformer encoder layers that processes input sequences through   
    multiple self-attention and feed-forward layers. Optionally includes convolutional   
    layers between attention layers for distillation and a final normalization layer.  

    Args:  
        attn_layers (list of TransEncoderLayer): List of transformer encoder layers to stack.  
        conv_layers (list of nn.Module, optional): List of convolutional layers applied   
            between attention layers. Must have length len(attn_layers) - 1 if provided.  
            Used for distillation in models like Informer.
        norm_layer (nn.Module, optional): Normalization layer applied to the final output.
            Typically nn.LayerNorm.

    Returns:
        (torch.Tensor): Encoded output tensor of shape [batch, seq_len, hidden_size]
              after passing through all encoder layers and optional normalization.
        (list[torch.Tensor]]): List of attention weights from each encoder layer,
              each of shape [batch, n_heads, seq_len, seq_len] (or None if not computed).

    Notes:
        When conv_layers is provided, the encoder alternates between attention layers
        and convolutional layers, with the final attention layer applied without a
        subsequent convolution. This architecture is used in the Informer model.
    """

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
    """Transformer Decoder Layer.  

    A single layer of the transformer decoder that applies masked self-attention,   
    cross-attention with encoder outputs, and a position-wise feed-forward network   
    with residual connections and layer normalization. Dropout is applied after each   
    sub-layer (self-attention, cross-attention, and twice in the feed-forward network)   
    before the residual connection for regularization.  

    Args:  
        self_attention (AttentionLayer): Masked self-attention mechanism for the decoder.  
        cross_attention (AttentionLayer): Cross-attention mechanism to attend to encoder outputs.  
        hidden_size (int): Dimension of the model's hidden representations.  
        conv_hidden_size (int, optional): Dimension of the feed-forward network's hidden layer.  
            Defaults to 4 * hidden_size if not specified.  
        dropout (float, optional): Dropout probability applied after attention and feed-forward layers.  
        activation (str, optional): Activation function to use in the feed-forward network.
            Either "relu" or "gelu".

    Returns:
        (torch.Tensor): Output tensor of shape [batch, target_seq_len, hidden_size] after
            applying masked self-attention, cross-attention, and feed-forward transformations.

    Notes:
        The layer applies three main operations in sequence:
        1. Masked self-attention on the decoder input with dropout, residual connection, and normalization
        2. Cross-attention between decoder and encoder outputs with dropout, residual connection, and normalization
        3. Position-wise feed-forward network using 1D convolutions with dropout applied twice (after each convolution),
           residual connection, and normalization
    """

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
    """Transformer decoder module for sequence-to-sequence forecasting.  

    Stacks multiple TransDecoderLayer modules to process decoder inputs with   
    self-attention and cross-attention mechanisms. Optionally applies layer   
    normalization and a final projection layer to produce output predictions.  
      
    Args:  
        layers (list): List of TransDecoderLayer instances to stack sequentially.  
        norm_layer (nn.Module, optional): Layer normalization module applied after   
            all decoder layers.
        projection (nn.Module, optional): Final projection layer (typically nn.Linear)
            to map hidden representations to output dimension.

    Returns:
        (torch.Tensor): Decoded output tensor. If projection is provided, returns tensor
            of shape [batch, target_seq_len, output_dim]. Otherwise, returns tensor of
            shape [batch, target_seq_len, hidden_size].

    Notes:
        - The forward method requires both decoder input (x) and encoder output (cross).
        - Masks are optional and used for attention masking in self-attention (x_mask)
          and cross-attention (cross_mask).
        - Each layer performs self-attention on decoder input, cross-attention with
          encoder output, and feedforward transformation.
    """

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
    """Multi-head attention layer wrapper.  

    This layer wraps an attention mechanism and handles the linear projections  
    for queries, keys, and values in multi-head attention. It projects inputs  
    to multiple heads, applies the inner attention mechanism, and projects back  
    to the original hidden dimension.  

    Args:  
        attention (nn.Module): Inner attention mechanism (e.g., FullAttention,   
            ProbAttention) that computes attention scores and outputs.  
        hidden_size (int): Dimension of the model's hidden states.  
        n_heads (int): Number of attention heads.  
        d_keys (int, optional): Dimension of keys per head. If `None` defaults to   
            hidden_size // n_heads.  
        d_values (int, optional): Dimension of values per head. If `None` defaults to
            hidden_size // n_heads.

    Returns:
        (torch.Tensor): Output tensor of shape [batch, seq_len, hidden_size] after
          applying multi-head attention.
        (torch.Tensor) or None: Attention weights of shape [batch, n_heads, seq_len, seq_len]
          if output_attention is True in the inner attention mechanism, otherwise None.

    Notes:
        - The forward method accepts queries, keys, values, and optional masks.
        - Additional parameters tau and delta are passed through to the inner
          attention mechanism for specialized attention variants.
    """

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
    """Triangular causal mask for autoregressive attention.  

    Creates an upper triangular boolean mask that prevents attention mechanisms  
    from attending to future positions in the sequence. This ensures causality  
    in autoregressive models where predictions at time t should only depend on  
    positions before t.  

    The mask is created using torch.triu with diagonal=1, resulting in a mask  
    where positions (i, j) are True when j > i, effectively masking out future  
    positions during attention computation.  

    Args:  
        B (int): Batch size.  
        L (int): Sequence length.  
        device (str, optional): Device to place the mask tensor on.   

    Attributes:  
        _mask (torch.Tensor): Boolean mask tensor of shape [B, 1, L, L] where  
            True values indicate positions to mask (future positions).  

    Notes:  
        - The mask shape [B, 1, L, L] is designed for multi-head attention where  
          the second dimension broadcasts across attention heads.  
        - True values in the mask indicate positions that should be masked out  
          (set to -inf before softmax in attention).  
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
    """Full attention mechanism with scaled dot-product attention.  

    Implements standard multi-head attention using scaled dot-product attention.  
    Supports both efficient computation via PyTorch's scaled_dot_product_attention  
    and explicit attention computation when attention weights are needed. Optional  
    causal masking prevents attention to future positions in autoregressive models.  

    Args:  
        mask_flag (bool, optional): If True, applies causal masking to prevent  
            attention to future positions.
        factor (int, optional): Attention factor parameter (unused in FullAttention,  
            kept for API compatibility with ProbAttention).
        scale (float, optional): Custom scaling factor for attention scores. If None,  
            uses 1/sqrt(d_k) where d_k is the key dimension.  
        attention_dropout (float, optional): Dropout rate applied to attention  
            weights.
        output_attention (bool, optional): If True, returns attention weights along
            with output. If False, uses efficient flash attention.

    Returns:
        (torch.Tensor): Attention output of shape [batch, seq_len, n_heads, head_dim].
        (torch.Tensor or None): Attention weights of shape [batch, n_heads, seq_len, seq_len]
              if output_attention is True, otherwise None.

    Notes:
        - When output_attention=False, uses PyTorch's optimized scaled_dot_product_attention
          for better performance (flash attention).
        - When output_attention=True, computes attention explicitly using einsum operations.
        - If mask_flag=True and no attn_mask is provided, automatically creates a
          TriangularCausalMask for autoregressive attention.
        - The tau and delta parameters are accepted for API compatibility but unused.
    """

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
    """Sinusoidal positional embedding for transformer models.  

    Generates fixed sinusoidal positional encodings using sine and cosine functions  
    at different frequencies. These encodings provide position information to   
    transformer models, allowing them to understand the relative or absolute position  
    of tokens in a sequence. The encodings are precomputed and stored as a buffer,  
    making them non-trainable.  

    The positional encoding for position pos and dimension i is computed as:  
        PE(pos, 2i) = sin(pos / 10000^(2i/hidden_size))  
        PE(pos, 2i+1) = cos(pos / 10000^(2i/hidden_size))  

    Args:  
        hidden_size (int): Dimension of the model's hidden states. Must be even  
            for proper sine/cosine pairing.  
        max_len (int, optional): Maximum sequence length to precompute encodings for.

    Returns:
        (torch.Tensor): Positional encodings of shape [1, seq_len, hidden_size] where
            seq_len is the length of the input sequence.

    Notes:
        - The positional encodings are fixed (not learned) and stored as a buffer.
        - The forward method returns encodings for the input sequence length only.
        - Different frequencies allow the model to attend to relative positions.
        - The encoding dimension must match the model's hidden_size.
    """

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
    """1D convolutional token embedding for time series.  

    Transforms input time series into embeddings using a 1D convolution with  
    circular padding. This approach captures local temporal patterns and projects  
    the input channels to the desired hidden dimension. The circular padding mode  
    helps maintain temporal continuity at sequence boundaries.  

    The convolution uses kernel size 3 with Kaiming normal initialization for  
    stable training dynamics.  

    Args:  
        c_in (int): Number of input channels (variables in multivariate series).
        hidden_size (int): Dimension of the output embeddings.

    Returns:
        (torch.Tensor): Token embeddings of shape [batch, seq_len, hidden_size].

    Notes:
        - Uses circular padding mode to handle sequence boundaries smoothly.
        - Weights are initialized with Kaiming normal initialization.
        - The convolution kernel size is fixed at 3.
        - Padding size is version-dependent (1 for PyTorch >= 1.5.0, else 2).
    """

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
    """Linear embedding for temporal/calendar features.  

    Transforms time-based features (e.g., hour, day, month) into embeddings using  
    a single linear projection without bias. This embedding is typically used to  
    incorporate calendar information into transformer models, providing absolute  
    temporal context that complements positional encodings.  

    Args:  
        input_size (int): Number of input temporal features (e.g., 5 for month,  
            day, weekday, hour, minute).  
        hidden_size (int): Dimension of the output embeddings, matching the
            model's hidden dimension.

    Returns:
        (torch.Tensor): Time feature embeddings of shape [batch, seq_len, hidden_size].

    Notes:
        - Uses a bias-free linear layer for simple feature projection.
        - Typically combined with TokenEmbedding and PositionalEmbedding.
        - Input features are usually calendar-based (month, day, hour, etc.).
        - The embedding is learned during training, unlike fixed positional encodings.
    """

    def __init__(self, input_size, hidden_size):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        return self.embed(x)


class FixedEmbedding(nn.Module):  
    """Fixed sinusoidal embedding for categorical temporal features.  

    Creates non-trainable embeddings using sine and cosine functions at different  
    frequencies. Unlike PositionalEmbedding which encodes continuous positions,  
    FixedEmbedding is designed for discrete categorical inputs (e.g., hour of day,  
    day of month, month of year). The embeddings are precomputed and frozen,  
    making them non-learnable parameters.  

    The embedding for category c and dimension i is computed as:  
        Emb(c, 2i) = sin(c / 10000^(2i/d_model))  
        Emb(c, 2i+1) = cos(c / 10000^(2i/d_model))  

    Args:  
        c_in (int): Number of categories (e.g., 24 for hours, 32 for days).
        d_model (int): Dimension of the embedding vectors.

    Returns:
        (torch.Tensor): Fixed embeddings of shape [batch, seq_len, d_model], detached
            from the computation graph.

    Notes:
        - Embeddings are frozen and cannot be trained.
        - The forward method returns detached tensors to prevent gradient flow.
        - Used as an alternative to nn.Embedding for temporal features.
        - Provides consistent representations across different time periods.
    """

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
    """Temporal embedding module for encoding calendar-based time features.  

    Creates learnable or fixed embeddings for temporal features including month,   
    day, weekday, hour, and optionally minute. These embeddings are summed to   
    produce a combined temporal representation.  

    Args:  
        d_model (int): Dimension of the embedding vectors.  
        embed_type (str): Type of embedding to use. Options are "fixed" for   
            FixedEmbedding (sinusoidal) or "learned" for nn.Embedding (learnable).  
        freq (str): Frequency of the time series data. If "t", includes minute
            embeddings.

    Returns:
        (torch.Tensor): Combined temporal embeddings of shape [batch, seq_len, d_model],
            representing the sum of all temporal component embeddings.

    Notes:
        - Input tensor x should have shape [batch_size, seq_len, num_features] where
          features are ordered as [month, day, weekday, hour, minute].
        - Month embeddings use size 13 (0-12), day uses 32 (0-31), weekday uses 7 (0-6),
          hour uses 24 (0-23), and minute uses 4 (0-3).
        - The embeddings are summed element-wise to produce the final output.
    """

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
    """Data embedding module combining value, positional, and temporal embeddings.  

    Transforms time series data into high-dimensional embeddings by combining:  
    - Value embeddings: Convolutional encoding of the time series values  
    - Positional embeddings: Sinusoidal encodings for relative position within window  
    - Temporal embeddings: Linear projection of absolute calendar features (optional)  

    Args:  
        c_in (int): Number of input channels (variates) in the time series.  
        exog_input_size (int): Number of exogenous/temporal features. If 0, temporal   
            embeddings are disabled.  
        hidden_size (int): Dimension of the embedding vectors.  
        pos_embedding (bool): Whether to include positional embeddings.
        dropout (float): Dropout rate applied to the final embeddings.

    Returns:
        (torch.Tensor): Combined embeddings of shape [batch, seq_len, hidden_size] after
            applying dropout to the sum of value, positional, and temporal embeddings.

    Notes:
        - Value embeddings use `TokenEmbedding` with 1D convolution (kernel_size=3).
        - Positional embeddings use sinusoidal functions (sine for even dims, cosine for odd).
        - Temporal embeddings use a linear layer to project calendar features.
        - All three embeddings are summed element-wise before dropout is applied.
        - If `x_mark` is None, only value and positional embeddings are used.
    """

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

        # Add positional (relative within window) embedding with sines and cosines
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        # Add temporal (absolute in time series) embedding with linear layer
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding(x_mark)

        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):  
    """Inverted data embedding module for variate-as-token transformer architectures.  

    Transforms time series data by treating each variate (channel) as a token rather   
    than each time step. The input is permuted from [Batch, Time, Variate] to   
    [Batch, Variate, Time], then a linear layer projects the time dimension to the   
    hidden dimension. Optionally concatenates temporal covariates along the variate   
    dimension.  

    Args:  
        c_in (int): Number of input time steps (sequence length).  
        hidden_size (int): Dimension of the embedding vectors.
        dropout (float): Dropout rate applied to the embeddings.

    Returns:
        (torch.Tensor): Inverted embeddings of shape [batch, n_variates, hidden_size] or
            [batch, n_variates + n_temporal_features, hidden_size] if x_mark is provided.

    Notes:
        - Input x has shape [Batch, Time, Variate] and is permuted to [Batch, Variate, Time].
        - If x_mark is provided, it's concatenated along the variate dimension after permutation.
        - The linear layer projects from c_in (time steps) to hidden_size dimensions.
        - This architecture is used in inverted transformers like iTransformer and TimeXer.
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
    """Moving average block to highlight the trend of time series.  

    Applies a moving average filter using 1D average pooling to smooth time series   
    data and extract trend components. The input is padded on both ends by repeating   
    the first and last values to maintain the original sequence length.  

    Args:  
        kernel_size (int): Size of the moving average window.
        stride (int): Stride for the average pooling operation.

    Returns:
        (torch.Tensor): Smoothed time series of shape [batch, seq_len, channels],
            representing the trend component after applying moving average.

    Notes:
        - Input x has shape [Batch, Time, Channels].
        - Padding is applied by repeating the first value (kernel_size-1)//2 times at
          the beginning and the last value (kernel_size-1)//2 times at the end.
        - The output maintains the same shape as the input after padding and pooling.
        - Commonly used with stride=1 for trend extraction in decomposition models.
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
    """Series decomposition block for trend-residual decomposition.

    Decomposes time series into trend and residual components using moving average
    filtering. The trend is extracted via a moving average filter, and the residual
    is computed as the difference between the input and the trend.

    Args:
        kernel_size (int): Size of the moving average window for trend extraction.

    Returns:
        (torch.Tensor): Residual component of shape [batch, seq_len, channels],
          computed as the input minus the trend.
        (torch.Tensor): Trend component of shape [batch, seq_len, channels],
          extracted using the moving average filter.

    Notes:
        - The kernel_size is passed to MovingAvg with stride=1.
        - The residual component is computed as input minus trend.
        - The trend component is the smoothed series from the moving average.
        - Commonly used in decomposition-based forecasting models like DLinear and Autoformer.
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.MovingAvg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.MovingAvg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):  
    """Reversible Instance Normalization for time series forecasting.  

    Normalizes time series data by removing the mean (or last value) and scaling by   
    standard deviation. The normalization can be reversed after model predictions to   
    restore the original scale. Optionally includes learnable affine parameters for   
    additional transformation flexibility.  

    Args:  
        num_features (int): The number of features or channels in the time series.  
        eps (float): A value added for numerical stability.
        affine (bool): If True, RevIN has learnable affine parameters (weight and bias).   
        subtract_last (bool): If True, subtracts the last value instead of the mean   
            in normalization.  
        non_norm (bool): If True, no normalization is performed (identity operation).

    Returns:
        (torch.Tensor): Normalized tensor (if mode="norm") or denormalized tensor
            (if mode="denorm") of the same shape as the input [batch, seq_len, num_features].

    Notes:
        - The forward method requires a mode parameter: "norm" for normalization or
          "denorm" for denormalization.
        - Statistics (mean/last and stdev) are computed during normalization and stored
          for use in denormalization.
        - If affine=True, learnable parameters are initialized as weight=1 and bias=0.
        - The subtract_last option is useful for non-stationary time series.
        - Used in models like PatchTST and TimeLLM for input preprocessing.
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
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
    """Reversible Instance Normalization for multivariate time series models.  

    Normalizes multivariate time series data using batch statistics computed across   
    the time dimension. The normalization can be reversed after model predictions to   
    restore the original scale. Optionally includes learnable affine parameters for   
    additional transformation flexibility.  

    Args:  
        num_features (int): The number of features or channels in the time series.  
        eps (float): A value added for numerical stability.
        affine (bool): If True, RevINMultivariate has learnable affine parameters   
            (weight and bias).
        subtract_last (bool): Not used in this implementation (kept for API compatibility).  
        non_norm (bool): Not used in this implementation (kept for API compatibility).

    Returns:
        (torch.Tensor): Normalized tensor (if mode="norm") or denormalized tensor
            (if mode="denorm") of the same shape as the input [batch, seq_len, num_features].

    Notes:
        - The forward method requires a mode parameter: "norm" for normalization or
          "denorm" for denormalization.
        - Batch statistics (mean and std) are computed across axis=1 (time dimension).
        - If affine=True, learnable parameters have shape [1, 1, num_features].
        - Used in multivariate models like TSMixer, TSMixerx, and RMoK.
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
