
import torch
import torch.nn as nn
import torch.nn.functional as F
#%% Pytorch implementation of TSMixer
class MixingLayer(nn.Module):
    def __init__(self, n_series, input_size, dropout, ff_dim):
        super().__init__()
        # Normalization layers
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
        x = input.reshape(batch_size, -1)
        x = self.temporal_norm(x)
        x = x.reshape(batch_size, n_series, input_size)
        x = x.permute(0, 2, 1) 
        x = F.relu(self.temporal_lin(x))
        x = x.permute(0, 2, 1)
        x = self.temporal_drop(x)
        res = x + input

        # Feature MLP
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

class TSMixer_pt(nn.Module):
    def __init__(self, n_series, input_size, dropout, ff_dim, n_layers, h):
        super(TSMixer_pt, self).__init__()
        # Instance Normalization
        self.norm = ReversibleInstanceNorm1d(n_series, eps=1e-5)

        # Mixing layers
        mixing_layers = [MixingLayer(n_series=n_series, 
                                     input_size=input_size, 
                                     dropout=dropout, 
                                     ff_dim=ff_dim) 
                                     for _ in range(n_layers)]
        self.mixing_layers = nn.Sequential(*mixing_layers)

        # Linear output
        self.out = nn.Linear(in_features=input_size, 
                             out_features=h)

    def forward(self, windows_batch):
        x = self.norm(windows_batch)
        x = self.mixing_layers(x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x.permute(0, 2, 1)

        # Reverse the Instance Normalization
        x = self.norm.reverse(x)

        return x
