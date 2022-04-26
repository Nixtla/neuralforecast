# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data__hiertsdataset.ipynb (unless otherwise specified).

__all__ = ['HierTimeSeriesDataset']

# Cell
import numpy as np
from torch.utils.data import Dataset

# Cell
class HierTimeSeriesDataset(Dataset):
    def __init__(self,
                 # Bottom data
                 Y, X, S,
                 xcols,
                 xcols_hist,
                 xcols_futr,
                 xcols_sample_mask,
                 xcols_available_mask,
                 dates,
                 # Aggregated data
                 Y_agg, X_agg, S_agg,
                 xcols_agg,
                 xcols_hist_agg,
                 xcols_futr_agg,
                 # Generator parameters
                 T0, T, H,
                 lastwindow_mask=False,
                ):

        # Assert that raw data has correct series lens
        series_len = [len(Y_agg), len(Y), len(X_agg), len(X), len(S_agg), len(S)]
        assert len(np.unique(series_len))==1, f'Check your series length {series_len}'

        # Assert that raw data has correct time lens
        assert Y.shape[-1]>=T0+T and X.shape[-1]>=T0+T+H+1,\
            f'Times (Y.shape={Y.shape[-1]},T0+T={T0+T}), (X.shape={X.shape[-1]}, T0+T+H+1={T0+T+H+1})'

        # Bottom data
        self.Y = Y
        self.X = X
        self.S = S

        self.xcols      = xcols
        self.xcols_hist = xcols_hist
        self.xcols_futr = xcols_futr
        self.xcols_sample_mask = xcols_sample_mask
        self.xcols_available_mask = xcols_available_mask

        # Aggregated data
        self.Y_agg = Y_agg
        self.X_agg = X_agg
        self.S_agg = S_agg

        self.xcols_agg = xcols_agg
        self.xcols_hist_agg = xcols_hist_agg
        self.xcols_futr_agg = xcols_futr_agg

        # Feature indexes
        self.hist_agg_col    = list(xcols_agg.get_indexer(xcols_hist_agg))
        self.futr_agg_col    = list(xcols_agg.get_indexer(xcols_futr_agg))

        self.hist_col        = list(xcols.get_indexer(xcols_hist))
        self.futr_col        = list(xcols.get_indexer(xcols_futr))
        self.sample_mask_col    = xcols.get_loc(xcols_sample_mask)
        self.available_mask_col = xcols.get_loc(xcols_available_mask)

        self.dates = dates[T0+1:T0+T+H+1]

        # Copy sample mask to avoid overwriting it
        self.sample_mask = self.X[:,:,self.sample_mask_col,:].copy()
        if lastwindow_mask:
            # Create dummy to identify observations of
            # Y's last H steps (X's last H steps are forecast)
            lastwindow_mask = np.zeros(self.X[:,:,self.sample_mask_col,:].shape)
            lastwindow_mask[:,:,T0+T-H:T0+T] = 1
            self.sample_mask = lastwindow_mask

        # Batch parameters
        self.T0   = T0
        self.T    = T
        self.H    = H

    def __len__(self):
        return len(self.Y_agg)

    def __getitem__(self, idx):
        # Parse time indexes
        T0 = self.T0
        T  = self.T
        H  = self.H

        # [G,N,C,T]
        s = self.S[idx,:,:]
        x = self.X[idx,:,:,T0:T0+T][:,self.hist_col,:]
        f = self.X[idx,:,:,T0+1:T0+T+H+1][:,self.futr_col,:]
        y = self.Y[idx,:,T0+1:T0+T]

        sample_mask    = self.sample_mask[idx,:,T0+1:T0+T]
        available_mask = self.X[idx,:,:,T0+1:T0+T][:,self.available_mask_col,:]

        # [G,C,T] Shared features across N
        s_agg    = self.S_agg[idx,:]
        x_agg    = self.X_agg[idx,:,T0:T0+T,][self.hist_agg_col,:]
        f_agg    = self.X_agg[idx,:,T0+1:T0+T+H+1][self.futr_agg_col,:]
        y_agg    = self.Y_agg[idx,:,T0+1:T0+T]

        batch = dict(# Bottom data
                     Y=y, S=s, X=x, F=f,
                     sample_mask=sample_mask,
                     available_mask=available_mask,
                     # Aggregated data
                     Y_agg=y_agg, S_agg=s_agg, X_agg=x_agg, F_agg=f_agg)

        #print('\n')
        #print('================== BATCH ==============')
        #for key in batch.keys():
        #    print(f'{key}.shape:\t{batch[key].shape}')
        #print('\n')
        return batch