from nixtla.data.tsdataset import TimeSeriesDataset
from nixtla.data.datasets.m4 import M4Info, M4

group = M4Info['Yearly']
Y_df, _, S_df = M4.load(directory='data', group=group.name)

train_ts_dataset = TimeSeriesDataset(Y_df=Y_df, S_df=S_df,
                                     ds_in_test=group.horizon,
                                     mode='full',
                                     window_sampling_limit=20_000, # To limit backprop time
                                     input_size=4,
                                     output_size=group.horizon,
                                     idx_to_sample_freq=1,
                                     len_sample_chunks=group.horizon * 3,
                                     complete_inputs=True,
                                     skip_nonsamplable=True)
