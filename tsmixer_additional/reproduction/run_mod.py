# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train and evaluate models for time series forecasting."""
#%%
import sys
import argparse
import glob
import logging
import os
import time

from data_loader import TSFDataLoader
import models
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.argv = ['']
def parse_args():
  """Parse the arguments for experiment configuration."""

  parser = argparse.ArgumentParser(
      description='TSMixer for Time Series Forecasting'
  )

  # basic config
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument(
      '--model',
      type=str,
      default='tsmixer_rev_in',
      help='model name, options: [tsmixer, tsmixer_rev_in]',
  )

  # data loader
  parser.add_argument(
      '--data',
      type=str,
      default='ETTm2',
      choices=[
          'electricity',
          'exchange_rate',
          'national_illness',
          'traffic',
          'weather',
          'ETTm1',
          'ETTm2',
          'ETTh1',
          'ETTh2',
      ],
      help='data name',
  )
  parser.add_argument(
      '--feature_type',
      type=str,
      default='M',
      choices=['S', 'M', 'MS'],
      help=(
          'forecasting task, options:[M, S, MS]; M:multivariate predict'
          ' multivariate, S:univariate predict univariate, MS:multivariate'
          ' predict univariate'
      ),
  )
  parser.add_argument(
      '--target', type=str, default='OT', help='target feature in S or MS task'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoints/',
      help='location of model checkpoints',
  )
  parser.add_argument(
      '--delete_checkpoint',
      action='store_true',
      help='delete checkpoints after the experiment',
  )

  # forecasting task
  parser.add_argument(
      '--seq_len', type=int, default=512, help='input sequence length'
  )
  parser.add_argument(
      '--pred_len', type=int, default=96, help='prediction sequence length'
  )

  # model hyperparameter
  parser.add_argument(
      '--n_block',
      type=int,
      default=2,
      help='number of block for deep architecture',
  )
  parser.add_argument(
      '--ff_dim',
      type=int,
      default=64,
      help='fully-connected feature dimension',
  )
  parser.add_argument(
      '--dropout', type=float, default=0.9, help='dropout rate'
  )
  parser.add_argument(
      '--norm_type',
      type=str,
      default='B',
      choices=['L', 'B'],
      help='LayerNorm or BatchNorm',
  )
  parser.add_argument(
      '--activation',
      type=str,
      default='relu',
      choices=['relu', 'gelu'],
      help='Activation function',
  )
  parser.add_argument(
      '--kernel_size', type=int, default=4, help='kernel size for CNN'
  )
  parser.add_argument(
      '--temporal_dim', type=int, default=16, help='temporal feature dimension'
  )
  parser.add_argument(
      '--hidden_dim', type=int, default=64, help='hidden feature dimension'
  )

  # optimization
  parser.add_argument(
      '--num_workers', type=int, default=10, help='data loader num workers'
  )
  parser.add_argument(
      '--train_epochs', type=int, default=10, help='train epochs'
  )
  parser.add_argument(
      '--batch_size', type=int, default=32, help='batch size of input data'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='optimizer learning rate',
  )
  parser.add_argument(
      '--patience', type=int, default=5, help='number of epochs to early stop'
  )

  # save results
  parser.add_argument(
      '--result_path', default='result.csv', help='path to save result'
  )

  args = parser.parse_args()

  tf.keras.utils.set_random_seed(args.seed)

  return args
#%% Tensorflow model

args = parse_args()
if 'tsmixer' in args.model:
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
elif args.model == 'full_linear':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}'
elif args.model == 'cnn':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_ks{args.kernel_size}'
else:
    raise ValueError(f'Unknown model type: {args.model}')

# load datasets
data_loader = TSFDataLoader(
    args.data,
    args.batch_size,
    args.seq_len,
    args.pred_len,
    args.feature_type,
    args.target,
)
train_data = data_loader.get_train()
val_data = data_loader.get_val()
test_data = data_loader.get_test()

# train model
if 'tsmixer' or 'tsmixer_rev_in' in args.model:
    build_model = getattr(models, args.model).build_model
    model_tf = build_model(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
    )
elif args.model == 'full_linear':
    model_tf = models.full_linear.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
    )
elif args.model == 'cnn':
    model_tf = models.cnn.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
        kernel_size=args.kernel_size,
    )
else:
    raise ValueError(f'Model not supported: {args.model}')
#%% Assign same model layer weights
def set_layer_weights_pt(model_tf, model_pt, n_layers):
    # We have to set the weights in the layers equal to assure output equivalence
    # The batch-norm and layer-norm weights are identically initialized
    # by the frameworks so we don't have to include these below
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

    return model_pt

#%% Training Loop
# Tensorflow
loss = tf.keras.losses.MeanSquaredError()
# loss = tf.keras.losses.MeanAbsoluteError()
optimizer_tf = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
train_metric_tf = tf.keras.metrics.MeanAbsoluteError()
val_metric_tf = tf.keras.metrics.MeanAbsoluteError()
test_metric_tf = tf.keras.metrics.MeanAbsoluteError()

# Pytorch
from models.tsmixer_pt import TSMixer_pt
device = torch.device('cuda')
model_pt = TSMixer_pt(n_series=data_loader.n_feature,
                      input_size=args.seq_len,
                      dropout=args.dropout,
                      ff_dim=args.ff_dim,
                      n_layers=args.n_block,
                      h=args.pred_len)
model_pt = set_layer_weights_pt(model_tf, model_pt, n_layers=args.n_block)
model_pt.to(device)
loss_pt = nn.MSELoss()
# loss_pt = nn.L1Loss()
optimizer_pt = torch.optim.Adam(model_pt.parameters(), lr = args.learning_rate)
f_train_metric_pt = nn.L1Loss()
f_val_metric_pt = nn.L1Loss()
f_test_metric_pt = nn.L1Loss()

# num_epochs = args.train_epochs
num_epochs = 2
n_batches_train = len(list(train_data)) - 1
n_batches_val = len(list(val_data)) - 1
n_batches_test = len(list(test_data)) - 1

@tf.function
def train_step_tf(x, y):
    with tf.GradientTape() as tape:
        yhat_batch = model_tf(x, training=True)
        loss_batch = loss(y_true=y, y_pred=yhat_batch)
    grads = tape.gradient(loss_batch, model_tf.trainable_variables)
    optimizer_tf.apply_gradients(zip(grads, model_tf.trainable_variables))

    # Track progress
    train_metric_tf.update_state(y, yhat_batch)

    return loss_batch

@tf.function
def test_step_tf(x, y, metric_tf):
    yhat_batch_val = model_tf(x, training=False)
    metric_tf.update_state(y, yhat_batch_val)

# PyTorch
def train_step_pt(x, y, train_metric_pt, device):
    optimizer_pt.zero_grad()
    x_pt = torch.from_numpy(x.numpy()).to(device)
    y_pt = torch.from_numpy(y.numpy()).to(device)
    yhat_batch = model_pt(x_pt)
    loss_batch = loss_pt(yhat_batch, y_pt)
    loss_batch.backward()
    optimizer_pt.step()

    # Track progress
    with torch.no_grad():
        train_metric_pt += f_train_metric_pt(yhat_batch, y_pt)
    
    return train_metric_pt

def test_step_pt(x, y, f_metric_pt, val_metric_pt, device):
    x_pt = torch.from_numpy(x.numpy()).to(device)
    y_pt = torch.from_numpy(y.numpy()).to(device)
    with torch.no_grad():
        yhat_batch = model_pt(x_pt)
        val_metric_pt += f_metric_pt(yhat_batch, y_pt)

    return val_metric_pt

# Training loop
for epoch in range(num_epochs):
    model_pt.train(True)
    train_metric_pt = 0
    val_metric_pt = 0

    # Training loop - using batches of 32
    for batch, (x_batch, y_batch) in enumerate(train_data):
        if x_batch.shape[0] == 1:
            pass
        else:
            loss_batch = train_step_tf(x_batch, y_batch)
            train_metric_pt = train_step_pt(x_batch, y_batch, train_metric_pt, device)

    train_acc_tf = train_metric_tf.result()
    print(f"Epoch {epoch+1}: Training acc (tf)  : {train_acc_tf:.4f}")
    print(f"Epoch {epoch+1}: Training acc (pt)  : {train_metric_pt / n_batches_train:.4f}")


    # Reset training metrics at the end of each epoch
    train_metric_tf.reset_states()
    model_pt.eval()

    # Run validation loop
    for x_batch_val, y_batch_val in val_data:
        if x_batch_val.shape[0] == 1:
            pass
        else:
            test_step_tf(x_batch_val, y_batch_val, val_metric_tf)
            val_metric_pt = test_step_pt(x_batch_val, y_batch_val, f_val_metric_pt, val_metric_pt, device)

    val_acc = val_metric_tf.result()
    val_metric_tf.reset_states()        

    print(f"Epoch {epoch+1}: Validation acc (tf): {val_acc:.4f}")
    print(f"Epoch {epoch+1}: Validation acc (pt): {val_metric_pt / n_batches_val:.4f}")

# Run test loop
test_metric_pt = 0
for x_batch_test, y_batch_test in test_data:
    if x_batch_test.shape[0] == 1:
        pass
    else:
        test_step_tf(x_batch_test, y_batch_test, test_metric_tf)
        test_metric_pt = test_step_pt(x_batch_test, y_batch_test, f_test_metric_pt, test_metric_pt, device)

test_acc = test_metric_tf.result()

print(f"Test acc (tf): {test_acc:.4f}")
print(f"Test acc (pt): {test_metric_pt / n_batches_test:.4f}")

