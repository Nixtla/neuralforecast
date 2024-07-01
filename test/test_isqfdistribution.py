#%% Test IQLoss for all types of architectures
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, BiTCN
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
import matplotlib.pyplot as plt
import pandas as pd

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test
max_steps = 1000

fcst = NeuralForecast(
    models=[  
            NHITS(h=12,
                input_size=24,
                loss=DistributionLoss(distribution="ISQF", level=[10, 20, 30, 40, 50, 60, 70, 80, 90], num_pieces=1),
                learning_rate=1e-4,
                max_steps=max_steps,
                scaler_type='robust',
                futr_exog_list=['y_[lag12]'],
                hist_exog_list=None,
                stat_exog_list=['airline1'],
                early_stop_patience_steps=3,
                ),                                                   
            BiTCN(h=12,
                input_size=24,
                loss=DistributionLoss(distribution="ISQF", level=[10, 20, 30, 40, 50, 60, 70, 80, 90], num_pieces=1),
                dropout=0.1,
                max_steps=max_steps,
                scaler_type='standard',
                futr_exog_list=['y_[lag12]'],
                hist_exog_list=None,
                stat_exog_list=['airline1'],
                early_stop_patience_steps=3,
                ),           
    ],
    freq='M'
)
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
#%%
forecasts = fcst.predict(futr_df=Y_test_df)
#%%
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

# model = 'BiTCN'
model = 'NHITS'
level = 90
plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df[f'{model}-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:], 
                 y1=plot_df[f'{model}-lo-{level}'][-12:].values,
                 y2=plot_df[f'{model}-hi-{level}'][-12:].values,
                 alpha=0.4, label=f'level {level}')
plt.legend()
plt.grid()