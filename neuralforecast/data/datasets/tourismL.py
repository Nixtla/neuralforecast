# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data_datasets__tourismL.ipynb (unless otherwise specified).

__all__ = ['nonzero_indexes_by_row', 'TourismL']

# Cell
import os
import gc
import copy
import zipfile
from pathlib import Path
from collections import defaultdict
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder

from dataclasses import dataclass

from .utils import (
    CodeTimer, download_file,
    one_hot_encoding
)

from ...losses.numpy import (
    mqloss, rmse
)

# Cell
def nonzero_indexes_by_row(M):
    return [np.nonzero(M[row,:])[0] for row in range(len(M))]

# Cell
@dataclass
class TourismL:

    url = 'https://robjhyndman.com/publications/mint/'
    source_url = 'https://robjhyndman.com/data/TourismData_v3.csv'
    H = 12

    @staticmethod
    def get_normalized_crps(y, y_hat, q_to_pred):
        norm  = np.sum(y)
        loss  = mqloss(y=y, y_hat=y_hat, quantiles=q_to_pred)
        loss  = 2 * loss * np.sum(np.ones(y.shape)) / norm
        return loss

    @staticmethod
    def get_normalized_rmse(y, y_hat):
        norm = np.sum(y)
        loss = rmse(y=y, y_hat=y_hat)
        loss = loss * np.sum(np.ones(y.shape)) / norm
        return loss

    @staticmethod
    def get_hierarchical_crps(data, Y, Y_hat, q_to_pred, model_name='current'):
        hier_idxs   = data['hier_idxs']
        hier_levels = data['hier_levels']

        dpmn_gbu = [0.1260,0.0411,0.0624,0.1122,0.1571,0.0747,0.1100,0.1901,0.2600]
        dpmn_nbu = [0.1578,0.1130,0.1189,0.1466,0.1759,0.1315,0.1416,0.1908,0.2428]
        hiere2e  = [0.1520,0.0810,0.1030,0.1361,0.1752,0.1027,0.1403,0.2050,0.2727]
        permbu_mint = [np.nan] * 9
        arima_erm = [0.1689,0.0725,0.1071,0.1541,0.2052,0.1095,0.1628,0.2435,0.3076]
        arima_mint_shr = [0.1609,0.0440,0.0816,0.1433,0.2036,0.0830,0.1479,0.2437,0.3406]
        glm_poisson = [0.1762,0.0854,0.1153,0.1691,0.2165,0.0954,0.1682,0.2458,0.3134]

        crps_list = []
        for i, idxs in enumerate(hier_idxs):
            # Get the series specific to the hierarchical level
            y     = Y[idxs, :]
            y_hat = Y_hat[idxs, :, :]

            crps  = TourismL.get_normalized_crps(y, y_hat, q_to_pred)
            crps_list.append(crps)

        crps_df = pd.DataFrame({'Level': hier_levels,
                                'DPMN-GBU': dpmn_gbu,
                                'DPMN-NBU': dpmn_nbu,
                                'HierE2E': hiere2e,
                                'PERMBU-MinT': permbu_mint,
                                'ARIMA-ERM': arima_erm,
                                'ARIMA-MinT-shr': arima_mint_shr,
                                'GLM-Poisson': glm_poisson})
        crps_df[model_name] = crps_list

        return crps_df

    @staticmethod
    def get_hierarchical_rmse(data, Y, Y_hat, model_name='current'):
        # Parse data
        hier_idxs = data['hier_idxs']
        hier_levels = data['hier_levels']

        mqcnn_nbu= [np.nan] * 9
        dpmn_gbu = [np.nan] * 9
        dpmn_nbu = [np.nan] * 9
        hiere2e  = [np.nan] * 9
        arima_mint_shr = [np.nan] * 9
        glm_poisson = [0.5230,0.1052,0.1660,0.2726,0.4046,0.1408,0.2824,0.4718,0.7297]

        measure_list = []
        for i, idxs in enumerate(hier_idxs):
            y     = Y[idxs, :]
            y_hat = Y_hat[idxs, :]

            measure = TourismL.get_normalized_rmse(y, y_hat)
            measure_list.append(measure)

        eval_df = pd.DataFrame({'Level': hier_levels,
                                'MQCNN-NBU': mqcnn_nbu,
                                'DPMN-GBU': dpmn_gbu,
                                'DPMN-NBU': dpmn_nbu,
                                'HierE2E': hiere2e,
                                'ARIMA-MinT-shr': arima_mint_shr,
                                'GLM-Poisson': glm_poisson})
        eval_df[model_name] = measure_list

        return eval_df

    @staticmethod
    def load(directory: str):
        """
        Downloads and loads TourismL data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.

        Returns
        -------
        df: pd.DataFrame
            Target time series with columns ['unique_id', 'ds', 'y'].
        """

        TourismL.download(directory)

        file = f'{directory}/TourismData_v3.csv'
        if not os.path.exists(file):
            raise ValueError(
                "TourismL zip file not found in {}!".format(file) +
                " Please manually download from {}".format(TourismL.url))

        # Fixing original dataset dates
        df = pd.read_csv(file)
        df['ds'] = np.arange("1998-01-01","2017-01-01",dtype='datetime64[M]')
        df = df.drop(labels=['Year', 'Month'], axis=1)

        # Add aditional H steps for predictions
        # declare skipping ds from column names
        extra    = np.empty((12, len(df.columns)-1,))
        extra[:] = np.nan
        extra_df = pd.DataFrame.from_records(extra, columns=df.columns[1:])
        extra_df['ds'] = np.arange("2017-01-01", "2018-01-01",dtype='datetime64[M]')

        raw_df = pd.concat([df, extra_df], axis=0).reset_index(drop=True)

        return raw_df

    @staticmethod
    def download(directory: str) -> None:
        """
        Download M3 Dataset.

        Parameters
        ----------
        directory: str
            Directory path to download dataset.
        """
        path = f'{directory}'
        if not os.path.exists(path):
            download_file(path, TourismL.source_url)

    @staticmethod
    def preprocess_data(directory: str, verbose=False) -> None:
        # Read raw data and create ids for later use
        raw_df = TourismL.load(directory)
        dates = raw_df.ds
        raw_df.drop(['ds'], axis=1, inplace=True)
        unique_ids  = np.array(list(raw_df.columns))
        region_ids  = np.array([u_id[:3] for u_id in unique_ids]) #AAAHol->AAA
        purpose_ids = np.array([u_id[3:] for u_id in unique_ids]) #AAAHol->Hol

        with CodeTimer('Create H.  constraints', verbose):
            # Create static features df,
            # 1 country, 7 States, 27 Zones, 76 Regions
            H_df = pd.DataFrame({'unique_id': unique_ids})
            H_df['country'] = 'Australia'                 # 1
            H_df['state']   = H_df['unique_id'].str[:1]   # 7
            H_df['zone']    = H_df['unique_id'].str[:2]   # 27
            H_df['region']  = H_df['unique_id'].str[:3]   # 76

            # GeographyXpurpose fixed effects
            # 4 countryXpurpose, 28 StatesXpurpose,
            # 108 ZonesXpurpose, 304 RegionsXpurpose (unique_id)
            H_df['purpose']       = H_df['unique_id'].str[-3:]      # 4
            H_df['state_purpose'] = H_df['state'] + H_df['purpose'] # 28
            H_df['zone_purpose']  = H_df['zone']  + H_df['purpose'] # 108

            # Hierarchical aggregation matrix
            Hencoded = one_hot_encoding(df=H_df, index_col='unique_id')
            Hsum     = Hencoded.values[:, 1:].T # Eliminate unique_id index
            H        = np.concatenate((Hsum, np.eye(len(unique_ids))), axis=0)

        with CodeTimer('Create static_agg     ', verbose):
            # Final wrangling + save
            # Create aggregate(region) level dummy variables
            H_df_agg   = H_df[['region', 'country', 'state', 'zone']]
            H_df_agg   = H_df_agg.drop_duplicates()
            static_agg = one_hot_encoding(df=H_df_agg, index_col='region')

        with CodeTimer('Create temporal_agg   ', verbose):
            #----------------------- Y Variables -----------------------#

            # Hierarchical Y for Y_agg features
            Y_bottom = raw_df.values.T
            Y_hier    = H @ Y_bottom      # [n_agg+n_bottom, n_bottom] x [n_bottom, T]

            # [Total, State, Zone, Region, TotalP, StateP, ZoneP] [0, 1, 2, 3, 4, 5, 6]
            # [Total, State, Zone, Region] <--> [0, 1, 2, 3]
            # [TotalP, StateP, ZoneP, RegionP] <--> [4, 5, 6]
            Y_agg_idxs = nonzero_indexes_by_row(Hsum.T)
            Y_agg  = np.stack([Y_hier[idx, :].T for idx in Y_agg_idxs ])
            Y_agg  = Y_agg[:,:,[0, 1, 2, 3]]

            # Keep total geographic visits, erase redundant purpose
            n_groups = len(H_df.region.unique())
            n_purposes = len(H_df.purpose.unique())
            n_time = len(dates)
            n_agg = Y_agg.shape[-1]
            Y_agg    = Y_agg.reshape((n_groups,n_purposes,n_time,n_agg))
            Y_agg    = Y_agg[:,0,:,:] # Skip redundant purpose dimension

            #----------------------- X variables -----------------------#
            # Calendar variables December distance
            calendar_df = pd.DataFrame()
            calendar_df['ds'] = dates
            calendar_df['distance_month'] = calendar_df.ds.dt.month
            # December/January? spike match and normalization of month distance
            calendar_df['distance_month'] = (calendar_df['distance_month'] - 3) % 12
            calendar_df['distance_month'] = (calendar_df.distance_month) / 11.0 - 0.5
            distance_month = calendar_df.distance_month.values
            distance_month = np.tile(distance_month, (76,1))
            X_agg = np.tile(calendar_df.values, (76,1,1))

            # Final wrangling + save
            temporal_agg = np.concatenate([X_agg, Y_agg], axis=2)
            temporal_agg = temporal_agg.reshape(-1, temporal_agg.shape[-1])
            temporal_agg = pd.DataFrame.from_records(temporal_agg,
                                     columns=['ds', 'distance_month',
                                              'y_[total]', 'y_[state]',
                                              'y_[zone]', 'y_[region]'])
            temporal_agg['region'] = np.tile(static_agg.region.values[:,None], (240,1))

        with CodeTimer('Create static_bottom  ', verbose):
            # This version intends to help the model learn in this low sample task
            # Providing it with precomputed level/spread will model to "level" predictions
            # Some ideas to try:
            # 1) seasonal levels
            # 2) robust vs non robust levels and spreads
            # 3) like moving average, compute this metrics filtering the available data.

            # Create S_bottom matrix (previous static features)
            # S_bottom   = H[:n_agg, :].T

            # Avoid leakage into statistics and collapse time dimension,
            # using median (it is robust to outliers)
            Y_available = Y_bottom[:,:-12-12] # skip test and validation

            # [n_regions*n_purpose, n_time] --> [n_regions*n_purpose]
            level  = np.median(Y_available, axis=1)
            spread = np.median(np.abs(Y_available - level[:,None]), axis=1)

            # Seasonal levels
            Y_december = Y_available.reshape(304,18,12)[:,:,0]
            Y_january  = Y_available.reshape(304,18,12)[:,:,-1]
            december_level = np.median(Y_december/(level[:,None]+1), axis=1)
            january_level  = np.median(Y_january/(level[:,None]+1), axis=1)

            # Final wrangling + save
            static_bottom = np.concatenate([level[:,None], spread[:,None],
                                           december_level[:,None],
                                           january_level[:,None]], axis=1)
            static_bottom = pd.DataFrame.from_records(static_bottom,
                                        columns=['level', 'spread',
                                                 'level_[december]',
                                                 'level_[january]'])
            static_bottom['unique_id'] = unique_ids
            static_bottom['region']    = region_ids
            static_bottom['purpose']   = purpose_ids

        with CodeTimer('Create temporal_bottom', verbose):
            # To help the model learn seasonalities(12 months)
            # we compute lag 12 for 304 regions x purposes
            # Elements that roll beyond the last position are re-introduced at the first.
            Y_lags = np.roll(Y_bottom, shift=12, axis=1)
            Y_lags[:,:12] = Y_lags[:,12:24] # Clean raw_df NAs from first 12 entries

            # December/January dummy variables
            encoder = OneHotEncoder()
            month_cols = [f'month_[{str(x+1)}]' for x in range(12)]
            month_df = calendar_df['distance_month'].values.reshape(-1,1)
            month_df = encoder.fit_transform(month_df).toarray()
            month_df = pd.DataFrame(month_df, columns=month_cols)
            month_df['ds'] = calendar_df['ds']

            # Available and sample masks
            month_df['available_mask'] = 1
            month_df['sample_mask']    = 1

            month_df = month_df[['ds', 'month_[11]', 'month_[12]',
                                 'available_mask', 'sample_mask']]       # Filter, Dec/Jan
            month_bottom = np.tile(month_df.values[None,:,:], (304,1,1)) # Dummies for each serie

            # Final wrangling + save
            temporal_bottom = np.concatenate([month_bottom, Y_lags[:,:,None]], axis=2)
            temporal_bottom = np.concatenate([temporal_bottom, Y_bottom[:,:,None]], axis=2)
            temporal_bottom = temporal_bottom.reshape(-1, temporal_bottom.shape[-1])
            temporal_bottom = pd.DataFrame.from_records(temporal_bottom,
                                        columns=list(month_df.columns) + \
                                                ['y_[lag12]', 'y',])
            temporal_bottom['unique_id'] = np.repeat(unique_ids, 240)
            temporal_bottom['region']    = np.repeat(region_ids, 240)
            temporal_bottom['purpose']   = np.repeat(purpose_ids,240)

        # Checking dtypes correctness
        if verbose:
            print('1. static_agg.dtypes \n', static_agg.dtypes)
            print('2. temporal_agg.dtypes \n', temporal_agg.dtypes)
            print('3. static_bottom.dtypes \n', static_bottom.dtypes)
            print('4. temporal_bottom.dtypes \n', temporal_bottom.dtypes)

        # Save feathers for fast access
        H_df.to_csv(f'{directory}/H_df.csv', index=False)
        static_agg.to_csv(f'{directory}/static_agg.csv', index=False)
        temporal_agg.to_csv(f'{directory}/temporal_agg.csv', index=False)
        static_bottom.to_csv(f'{directory}/static_bottom.csv', index=False)
        temporal_bottom.to_csv(f'{directory}/temporal_bottom.csv', index=False)

    @staticmethod
    def load_process(directory=str, verbose=False) -> dict:

        with CodeTimer('Reading data           ', verbose):
            static_agg    = pd.read_csv(f'{directory}/static_agg.csv')
            temporal_agg    = pd.read_csv(f'{directory}/temporal_agg.csv')

            static_bottom = pd.read_csv(f'{directory}/static_bottom.csv')
            temporal_bottom = pd.read_csv(f'{directory}/temporal_bottom.csv')

            # Extract datasets dimensions for later use
            dates       = pd.to_datetime(temporal_agg.ds.unique())
            unique_ids  = static_bottom.unique_id.values
            region_ids  = static_bottom.region.values
            purpose_ids = static_bottom.purpose.values
            n_time      = len(dates)                             # 228
            n_group     = len(static_bottom.region.unique())     # 76
            n_series    = len(static_bottom.purpose.unique())    # 4

        #-------------------------------------- S/X/Y_agg    --------------------------------------#
        with CodeTimer('Process temporal_agg   ', verbose):
            # Drop observation indexes and obtain column indexes
            temporal_agg.drop(['region', 'ds'], axis=1, inplace=True)
            xcols_agg       = temporal_agg.columns
            xcols_hist_agg  = ['y_[total]', 'y_[state]', 'y_[zone]', 'y_[region]']
            xcols_futr_agg  = ['distance_month']

            X_agg  = temporal_agg.values
            Y_agg  = temporal_agg[['y_[total]', 'y_[state]',
                                   'y_[zone]', 'y_[region]']].values

            X_agg  = X_agg.reshape((n_group,n_time,temporal_agg.shape[1]))
            Y_agg  = Y_agg.reshape((n_group,n_time,4)) #features: total,state,zone,region

        with CodeTimer('Process static_agg     ', verbose):
            # Drop observation indexes and obtain column indexes
            S_agg = static_agg.drop(['region'], axis=1)
            scols_agg = S_agg.columns

            S_agg = S_agg.values

        del temporal_agg, static_agg
        gc.collect()

        #--------------------------------------- S/X/Y_bottom ---------------------------------------#
        with CodeTimer('Process temporal_bottom', verbose):
            # Drop observation indexes and obtain column indexes
            temporal_bottom.drop(['unique_id', 'region', 'purpose', 'ds'], axis=1, inplace=True)
            xcols      = temporal_bottom.columns
            xcols_hist = ['y']
            xcols_futr = ['month_[11]', 'month_[12]', 'y_[lag12]']

            X  = temporal_bottom.values
            Y  = temporal_bottom['y'].values

            # temporal_bottom assumes to be balanced ie len(Y) = n_group*n_series*n_time
            n_features = temporal_bottom.shape[1]
            X  = X.reshape((n_group,n_series,n_time,n_features))
            Y  = Y.reshape((n_group,n_series,n_time))

            #Reshape to match with the Mixture's inputs
            #n_groups, n_series, n_time, n_features -> n_group, n_time, n_series, n_features
            X = np.transpose(X, (0, 2, 1, 3))
            Y = np.transpose(Y, (0, 2, 1))

        with CodeTimer('Process static_bottom  ', verbose):
            # Drop observation indexes and obtain column indexes
            static_bottom.drop(['unique_id', 'region', 'purpose'], axis=1, inplace=True)
            scols = static_bottom.columns

            S = static_bottom.values
            S = S.reshape(n_group,n_series,static_bottom.shape[1])

        with CodeTimer('Hier constraints       ', verbose):
            # Read hierarchical constraints dataframe
            # Create hierarchical aggregation numpy
            H_df = pd.read_csv(f'{directory}/H_df.csv')
            Hencoded = one_hot_encoding(df=H_df, index_col='unique_id')
            Hsum = Hencoded.values[:, 1:].T # Eliminate stores index
            H = np.concatenate((Hsum, np.eye(len(unique_ids))), axis=0)
            hier_linked_idxs = nonzero_indexes_by_row(H.T)

            # Hierarchical dataset and evaluation utility
            Y_flat  = temporal_bottom['y'].values
            Y_flat  = Y_flat.reshape((n_group*n_series,n_time))

            Y_hier      = H @ Y_flat
            hier_labels = list(Hencoded.columns[1:]) + list(H_df.unique_id)
            hier_labels = np.array(hier_labels) # Numpy for easy list indexing
            hier_idxs   = [range(555),
                           range(0, 1), range(1, 1+7), range(8, 8+27), range(35, 111),
                           range(111, 111+4), range(115, 115+28),
                           range(143, 143+108), range(251, 251+304)] # Hardcoded hier_idxs
            hier_levels = ['Overall',
                           '1 (geo.)', '2 (geo.)', '3 (geo.)', '4 (geo.)',
                           '5 (prp.)', '6 (prp.)', '7 (prp.)', '8 (prp.)']

        del temporal_bottom, static_bottom
        gc.collect()

        #---------------------------------- Logs and Output ----------------------------------#
        with CodeTimer('Final processing       ', verbose):
            # NaiveSeasonal compatibility X_futr indexing
            # Consider T0=0, L=8, H=3 and Y=[0,1,2,3,4,5,6,7|,8,9,10,11]
            # T0=0 --> [0,1,2,3,4,5,6,7|,8,9,10]
            # T0=1 --> [1,2,3,4,5,6,7,8|9,10,11]
            # T0=1 uses T0+1 X_futr=[2,3,4,5,6,7,8,9|10,11,12] ???
            # Seasonal Naive12 switch to [-1] for Naive1
            X = np.concatenate((X, X[:,[-12],:]), axis=1)
            X_agg = np.concatenate((X_agg, X_agg[:,[-12],:]), axis=1)

            # Reshape NumPy arrays for Pytorch inputs
            # [0,1,2,3][G,T,N,C] -> [0,2,3,1][G,N,C,T]
            S_agg = np.float32(S_agg)
            Y_agg = np.float32(np.transpose(Y_agg, (0, 2, 1)))
            X_agg = np.float32(np.transpose(X_agg, (0, 2, 1)))

            S = np.float32(S)
            Y = np.float32(np.transpose(Y, (0, 2, 1)))
            X = np.float32(np.transpose(X, (0, 2, 3, 1)))

            # Skip NAs from Y data [G,N,T]
            Y = Y[:,:,:-12]
            Y_agg = Y_agg[:,:,:-12]
            Y_hier = np.float32(Y_hier[:,:-12])

            # Assert that the variables are contained in the column indexes
            assert all(c in list(xcols_agg) for c in xcols_hist_agg)
            assert all(c in list(xcols_agg) for c in xcols_futr_agg)
            assert all(c in list(xcols) for c in xcols_hist)
            assert all(c in list(xcols) for c in xcols_futr)

            # Hierarchical data for baseline models
            month_11 = np.repeat(X[[0],0,0,:228], 555, axis=0) #-12-1 iff 228
            month_12 = np.repeat(X[[0],0,1,:228], 555, axis=0)

            Y_lags = np.roll(Y_hier, shift=12, axis=1)
            Y_lags[:,:12] = Y_lags[:,12:24] # Clean raw_df NAs from first 12 entries

            X_hier = np.concatenate([Y_hier[:,:,None],  month_11[:,:,None],
                                      month_12[:,:,None], Y_lags[:,:,None]], axis=2)
            X_hier = X_hier.reshape(-1,4)
            xcols_hier = ['y','month_11', 'month_12', 'y_lag12']

            X_hier_df   = pd.DataFrame.from_records(X_hier, columns=xcols_hier)
            X_hier_df['unique_id'] = np.repeat(range(555), 228)
            X_hier_df['ds'] = np.tile(dates[:228], 555)

            del Y_lags, X_hier
            gc.collect()

            data = {# Bottom data
                    'S': S, 'X': X, 'Y': Y,
                    'scols': scols,
                    'xcols': xcols,
                    'xcols_hist': xcols_hist,
                    'xcols_futr': xcols_futr,
                    'xcols_sample_mask': 'sample_mask',
                    'xcols_available_mask': 'available_mask',
                    # Aggregate data
                    'S_agg': S_agg, 'X_agg': X_agg, 'Y_agg': Y_agg,
                    'scols_agg': scols_agg,
                    'xcols_agg': xcols_agg,
                    'xcols_hist_agg': xcols_hist_agg,
                    'xcols_futr_agg': xcols_futr_agg,
                    # Hierarchical data for evaluation
                    'Y_hier': Y_hier,
                    'hier_idxs': hier_idxs,
                    'hier_levels': hier_levels,
                    'hier_labels': hier_labels,
                    'hier_linked_idxs': hier_linked_idxs,
                    'X_hier_df': X_hier_df,
                    # Shared data
                    'H': H,
                    'dates': dates,
                    'unique_ids': unique_ids,
                    'region_ids': region_ids,
                    'pupose_ids': purpose_ids}

        if verbose:
            print('\n')
            h = 34
            print(f'Total days {len(dates)}, train {n_time-h*3} validation {h}, test {h}')
            print(f'Whole dates: \t\t [{min(dates)}, {max(dates[:-h])}]')
            print(f'Validation dates: \t '+\
                  f'[{min(dates[n_time-h*3:n_time-h*2])}, {max(dates[n_time-h*3:n_time-h*2])}]')
            print(f'Test dates: \t\t '+\
                  f'[{min(dates[n_time-h*2:n_time-h*1])}, {max(dates[n_time-h*2:n_time-h*1])}]')
            print(f'Forecast dates: \t '+\
                  f'[{min(dates[n_time-h*1:n_time-h*0])}, {max(dates[n_time-h*1:n_time-h*0])}]')
            print('\n')
            print(' '*35 +'BOTTOM')
            print('S.shape (n_regions,n_purpose,n_features):        \t' + str(data['S'].shape))
            print('X.shape (n_regions,n_purpose,n_features,n_time+H+1):    ' + str(data['X'].shape))
            print('Y.shape (n_regions,n_purpose,n_time):            \t' + str(data['Y'].shape))
            #print(f"scols ({len(scols)}) {data['scols']}")
            #print(f"xcols ({len(xcols)}) {data['xcols']}")
            #print(f"xcols_hist ({len(xcols_hist)}) {data['xcols_hist']}")
            #print(f"xcols_futr ({len(xcols_futr)}) {data['xcols_futr']}")
            print(' '*35 +'AGGREGATE')
            print('S_agg.shape (n_regions,n_features):           \t\t' + str(data['S_agg'].shape))
            print('X_agg.shape (n_regions,n_features,n_time+H+1):  \t' + str(data['X_agg'].shape))
            print('Y_agg.shape (n_regions,n_features,n_time):    \t\t' + str(data['Y_agg'].shape))
            #print(f"scols_agg ({len(scols_agg)}) {data['scols_agg'][:4]}")
            #print(f"xcols_agg ({len(xcols_agg)}) {data['xcols_agg']}")
            #print(f"xcols_hist_agg ({len(xcols_hist_agg)}) {data['xcols_hist_agg']}")
            #print(f"xcols_futr_agg ({len(xcols_futr_agg)}) {data['xcols_futr_agg']}")
            print(' '*35 +'HIERARCHICAL')
            print('Y_hier.shape (n_regions,n_time):   \t\t\t' + str(data['Y_hier'].shape))
            #print('hier_idxs', hier_idxs)
            print('overall.shape '+6*'\t',  data['Y_hier'][hier_idxs[0],:].shape)
            print('country.shape '+6*'\t',  data['Y_hier'][hier_idxs[1],:].shape)
            print('states.shape  '+6*'\t',  data['Y_hier'][hier_idxs[2],:].shape)
            print('zones.shape   '+6*'\t',  data['Y_hier'][hier_idxs[3],:].shape)
            print('regions.shape '+6*'\t',  data['Y_hier'][hier_idxs[4],:].shape)
            print('country_purpose.shape '+5*'\t',  data['Y_hier'][hier_idxs[5],:].shape)
            print('states_purpose.shape  '+5*'\t',  data['Y_hier'][hier_idxs[6],:].shape)
            print('zones_purpose.shape   '+5*'\t',  data['Y_hier'][hier_idxs[7],:].shape)
            print('regions_purpose.shape '+5*'\t',  data['Y_hier'][hier_idxs[8],:].shape)

        return data