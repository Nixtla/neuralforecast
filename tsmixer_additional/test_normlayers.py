# This file tests normalization layers of tensorflow/keras against PyTorch
#%% Import stuff
import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import numpy as np
#%% Generate random tensor
g1 = tf.random.Generator.from_seed(123456789)
batch = 32
input_length = 160
channel = 32
inputs_tf = g1.normal(shape=[batch, input_length, channel])
#%% LayerNormalization
# Tensorflow
inputs_np = np.array(inputs_tf)
norm = layers.LayerNormalization
norm_tf = norm(axis=[-2, -1])
x_tf = norm_tf(inputs_tf)
x_np = np.array(x_tf)
# PyTorch
inputs_pt = torch.from_numpy(inputs_np)
norm_pt = nn.LayerNorm(normalized_shape=(input_length, channel), eps=0.001)
x_pt = norm_pt(inputs_pt)
# Assert equivalence
assert np.allclose(x_np, x_pt.detach().numpy(), atol=1e-7)
#%% BatchNormalization
# Tensorflow
inputs_np = np.array(inputs_tf)
norm = layers.BatchNormalization
norm_tf = norm(axis=[-2, -1])
x_tf = norm_tf(inputs_tf, training=True)
x_np = np.array(x_tf)
# Pytorch
inputs_pt = torch.from_numpy(inputs_np)
norm_pt = nn.BatchNorm1d(channel * input_length, eps=0.001, momentum=0.01)
x_pt = norm_pt(inputs_pt.view(batch, -1)).view(batch, input_length, channel)
# Assert equivalence
assert np.allclose(x_np, x_pt.detach().numpy(), atol=1e-7)