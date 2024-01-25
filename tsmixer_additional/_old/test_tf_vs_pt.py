# This file tests the reference implementation of TSMixer against a Pytorch implementation
#%% Import stuff
import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#%% Reference implementation
# This is the reference implementation for TSMixer
# We added naming to the Dense layers to make testing
# for equivalence to PyTorch easier. 
# Source: https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/tsmixer.py

class RevNorm(layers.Layer):
  """Reversible Instance Normalization."""

  def __init__(self, axis, eps=1e-5, affine=True):
    super().__init__()
    self.axis = axis
    self.eps = eps
    self.affine = affine

  def build(self, input_shape):
    if self.affine:
      self.affine_weight = self.add_weight(
          'affine_weight', shape=input_shape[-1], initializer='ones'
      )
      self.affine_bias = self.add_weight(
          'affine_bias', shape=input_shape[-1], initializer='zeros'
      )

  def call(self, x, mode, target_slice=None):
    if mode == 'norm':
      self._get_statistics(x)
      x = self._normalize(x)
    elif mode == 'denorm':
      x = self._denormalize(x, target_slice)
    else:
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    self.mean = tf.stop_gradient(
        tf.reduce_mean(x, axis=self.axis, keepdims=True)
    )
    self.stdev = tf.stop_gradient(
        tf.sqrt(
            tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
        )
    )

  def _normalize(self, x):
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    if self.affine:
      x = x - self.affine_bias
      x = x / self.affine_weight
    x = x * self.stdev
    x = x + self.mean
    return x

def res_block(inputs, norm_type, activation, dropout, ff_dim, i):
  """Residual block of TSMixer."""

  norm = (
      layers.LayerNormalization
      if norm_type == 'L'
      else layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(x.shape[-1], activation=activation, name='temporal_lin.'+str(i))(x)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  x = norm(axis=[-2, -1])(res)
  x = layers.Dense(ff_dim, activation=activation, name='feature_lin_1.'+str(i))(
      x
  )  # [Batch, Input Length, FF_Dim]
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1], name='feature_lin_2.'+str(i))(x)  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  return x + res

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  rev_norm = RevNorm(axis=-2)
  x = rev_norm(x, 'norm')
  for i in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim, i)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len, name='out')(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
  outputs = rev_norm(outputs, 'denorm', target_slice)

  return tf.keras.Model(inputs, outputs)

#%% Pytorch implementation of TSMixer
class MixingLayer(nn.Module):
    def __init__(self, n_series, input_size, dropout, ff_dim, norm_type):
        super().__init__()
        # Normalization layers
        self.norm_type = norm_type
        if self.norm_type == 'L':
            self.temporal_norm = nn.LayerNorm(normalized_shape=(input_size, n_series), eps=0.001)
            self.feature_norm = nn.LayerNorm(normalized_shape=(input_size, n_series), eps=0.001)
        else:
            self.temporal_norm = nn.BatchNorm1d(num_features=n_series * input_size, eps=0.001, momentum=0.01)
            self.feature_norm = nn.BatchNorm1d(num_features=n_series * input_size, eps=0.001, momentum=0.01)
        
        # Linear layers
        self.temporal_lin = nn.Linear(input_size, input_size)
        self.feature_lin_1 = nn.Linear(n_series, ff_dim)
        self.feature_lin_2 = nn.Linear(ff_dim, n_series)

        # Drop out layers
        self.temporal_drop = nn.Dropout(dropout)
        self.feature_drop_1 = nn.Dropout(dropout)
        self.feature_drop_2 = nn.Dropout(dropout)

    def forward(self, input):
        batch_size = input.shape[0]
        n_series = input.shape[1]
        input_size = input.shape[2]

        # Temporal MLP
        if self.norm_type == 'L':
            x = self.temporal_norm(input)
        else:
            x = input.reshape(batch_size, -1)
            x = self.temporal_norm(x)
            x = x.reshape(batch_size, n_series, input_size)
        x = x.permute(0, 2, 1) 
        x = F.relu(self.temporal_lin(x))
        x = x.permute(0, 2, 1)
        x = self.temporal_drop(x)
        res = x + input

        # Feature MLP
        if self.norm_type == 'L':
            x = self.feature_norm(res)
        else:
            x = res.reshape(batch_size, -1)
            x = self.feature_norm(x)
            x = x.reshape(batch_size, n_series, input_size)
        x = F.relu(self.feature_lin_1(x))
        x = self.feature_drop_1(x)
        x = self.feature_lin_2(x)
        x = self.feature_drop_2(x)

        return x + res
    
class ReversibleInstanceNorm1d(nn.Module):
    def __init__(self, n_series, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_series))
        self.bias = nn.Parameter(torch.zeros(n_series))
        self.eps = eps

    def forward(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=1, keepdim=True).detach()
        self.batch_std = torch.sqrt(torch.var(x, axis=1, keepdim=True, unbiased=False) + self.eps).detach()
        
        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std
        x = x * self.weight
        x = x + self.bias
        
        return x

    def reverse(self, x):
        # Reverse the normalization
        x = x - self.bias
        x = x / self.weight        
        x = x * self.batch_std
        x = x + self.batch_mean       

        return x

class TSMixer(nn.Module):
    def __init__(self, n_series, input_size, dropout, ff_dim, norm_type, n_layers, h):
        super().__init__()
        # Instance Normalization
        # self.norm = nn.InstanceNorm1d(n_series, affine=True, eps=1e-5, track_running_stats=True)
        self.norm = ReversibleInstanceNorm1d(n_series, eps=1e-5)

        # Mixing layers
        mixing_layers = [MixingLayer(n_series=n_series, 
                                     input_size=input_size, 
                                     dropout=dropout, 
                                     ff_dim=ff_dim, 
                                     norm_type=norm_type) 
                                     for _ in range(n_layers)]
        self.mixing_layers = nn.Sequential(*mixing_layers)

        # Linear output
        self.out = nn.Linear(in_features=input_size, 
                             out_features=h)

    def forward(self, windows_batch):
        # print(windows_batch.shape)
        x = self.norm(windows_batch)
        x = self.mixing_layers(x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x.permute(0, 2, 1)

        # Reverse the Instance Normalization
        x = self.norm.reverse(x)

        return x


#%% Compare implementations
# Generate random tensor
g1 = tf.random.Generator.from_seed(123456789)
batch = 2
input_length = 24
n_series = 2
ff_dim = 32
pred_length = 12
norm_type = 'B'
n_layers = 2
dropout = 0.0   # Set to zero beccause we can't make this identical in TF/PT
inputs_tf = g1.normal(shape=[batch, input_length, n_series])
inputs_np = np.array(inputs_tf)
inputs_pt = torch.from_numpy(inputs_np)

# Tensorflow
model_tf = build_model(input_shape = [input_length, n_series],  
                    pred_len=pred_length, 
                    norm_type=norm_type,
                    activation='relu',
                    n_block=n_layers,
                    dropout=dropout,
                    ff_dim=ff_dim,
                    target_slice=None)

output_tf = model_tf(inputs_tf, training=True)

#%% Pytorch
model_pt = TSMixer(n_series=n_series, 
                input_size=input_length,
                dropout=dropout,
                ff_dim=ff_dim,
                norm_type=norm_type,
                n_layers=n_layers,
                h=pred_length)

# We have to set the weights in the layers equal to assure output equivalence
# The batch-norm and layer-norm weights are identically initialized
# by the frameworks so we don't have to include these below
model_tf_layer_names = [layer.name for layer in model_tf.layers]

for i in range(n_layers):
    # Temporal dense layer
    temporal_lin_weights = model_tf.get_layer('temporal_lin.'+str(i)).get_weights()
    model_pt.mixing_layers[i].temporal_lin.weight.data = torch.from_numpy(temporal_lin_weights[0].T)
    model_pt.mixing_layers[i].temporal_lin.bias.data = torch.from_numpy(temporal_lin_weights[1])

    # Feature dense layers
    feature_lin_1_weights = model_tf.get_layer('feature_lin_1.'+str(i)).get_weights()
    feature_lin_2_weights = model_tf.get_layer('feature_lin_2.'+str(i)).get_weights()
    model_pt.mixing_layers[i].feature_lin_1.weight.data = torch.from_numpy(feature_lin_1_weights[0].T)
    model_pt.mixing_layers[i].feature_lin_1.bias.data = torch.from_numpy(feature_lin_1_weights[1])
    model_pt.mixing_layers[i].feature_lin_2.weight.data = torch.from_numpy(feature_lin_2_weights[0].T)
    model_pt.mixing_layers[i].feature_lin_2.bias.data = torch.from_numpy(feature_lin_2_weights[1])

# Output dense layer
out_weights = model_tf.get_layer('out').get_weights()
model_pt.out.weight.data = torch.from_numpy(out_weights[0].T)
model_pt.out.bias.data = torch.from_numpy(out_weights[1])

# PyTorch model output
output_pt = model_pt(inputs_pt)
#%% Assert equivalence between Tensorflow and PyTorch implementations
assert np.allclose(output_pt.detach().numpy(), output_tf, atol=1e-4)