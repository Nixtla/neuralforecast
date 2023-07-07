import os
import time
import argparse
import pandas as pd
import numpy as np
import gc

from neuralforecast.core import NeuralForecast
from config_timenet import MODEL_LIST, load_model


def main(args):
    frequency = args.frequency
    model_name = args.model + '_' + frequency.lower()

    model_type = args.model.split('_')[0]
    # make sure folder exists, then check if the file exists in the folder
    model_dir = f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/'
    os.makedirs(model_dir, exist_ok=True)
    file_exists = os.path.isfile(
        f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/{model_type}_0.ckpt')

    if (not file_exists):

        # Frequency for nf
        if frequency == 'Monthly':
            freq = 'M'
        elif frequency == 'Weekly':
            freq = 'W'
        elif frequency == 'Daily':
            freq = 'D'

        model = load_model(model_name)
        nf = NeuralForecast(models=[model], freq=freq)

        # TOTAL STEPS
        if frequency in ['Monthly', 'Weekly', 'Daily']:
            total_steps = 20000

        individual_steps = model.max_steps

        partitions_df = pd.read_csv('partitions_df.csv')
        partitions_df = partitions_df[partitions_df['frequency'] == frequency]
        print('PARTITIONS: ', partitions_df)

        for i in range(total_steps//individual_steps):
            start = time.time()
            print('ITERATION: ', i , 'out of ', total_steps//individual_steps)

            df_list = []
            # Add WIKI
            if frequency in ['Monthly', 'Weekly', 'Daily']:
                print("Loading WIKI")

                # Limit number of partitions to control RAM usage
                if frequency == 'Monthly':
                    num_partitions = 10
                elif frequency == 'Weekly':
                    num_partitions = 5
                elif frequency == 'Daily':
                    num_partitions = 1

                wiki_urls = partitions_df[partitions_df['dataset'] == 'Wikipedia']['url'].values
                wiki_urls = np.random.choice(wiki_urls, size=num_partitions, replace=False) # Ensure no duplicates

                for wiki_url in wiki_urls:
                    df_list.append(pd.read_parquet(wiki_url))

            # OTHER
            other_urls = partitions_df[partitions_df['dataset'] != 'Wikipedia']['url'].values
            if frequency == 'Monthly':
                other_urls = np.random.choice(other_urls, size=5, replace=False) # Ensure no duplicates
            elif frequency == 'Daily':
                other_urls = np.random.choice(other_urls, size=1, replace=False) # Ensure no duplicates
            
            print('Other urls: ', other_urls)

            # Load
            print("Loading OTHER")
            for url in other_urls:
                df_list.append(pd.read_parquet(url))
                              
            # Process data
            Y_df = pd.concat(df_list, axis=0).reset_index(drop=True)
            Y_df = Y_df.drop(Y_df.groupby(['unique_id']).tail(model.h).index, axis=0).reset_index(drop=True)

            if frequency == 'Daily':
                print('Subsampling to 3000 unique ids')
                unique_ids = Y_df['unique_id'].unique()
                if len(unique_ids) > 3000:
                    unique_ids = np.random.choice(unique_ids, size=3000, replace=False)
                    Y_df = Y_df[Y_df['unique_id'].isin(unique_ids)].reset_index(drop=True)
            
            nf.fit(df=Y_df, val_size=model.h, use_init_models=False)
            end = time.time()            
            print('Time: ', end-start)

        del Y_df
        del df_list
        gc.collect()

        # Save model
        print('Saving model')
        nf.save(path=f'./results/stored_models/{args.source_dataset}/{model_name}/{args.experiment_id}/',
            overwrite=False, save_dataset=False)
    else:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="script arguments")
    parser.add_argument('--source_dataset', type=str, help='dataset to train models on')
    parser.add_argument('--model', type=str, help='auto model to use')
    parser.add_argument('--experiment_id', type=str, help='identify experiment')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    frequency_list = ['Monthly', 'Weekly', 'Daily'] #['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']

    for frequency in frequency_list:
        print('Frequency: ', frequency)
        args.frequency = frequency
        main(args)


# CUDA_VISIBLE_DEVICES=0 python run_timenet.py --model "nhits_30_1024" --source_dataset "timenet" --experiment_id "20230626"

# 4 with 17
# 3 with 10
# 2 with 5