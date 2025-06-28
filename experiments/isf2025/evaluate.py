import os
import glob
import s3fs


import polars as pl
from dotenv import load_dotenv
from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from pathlib import Path

load_dotenv()
storage_options = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "aws_region": "us-east-1",
}

s3 = s3fs.S3FileSystem(anon=False)
s3_path = f"s3://timenet/isf2025"
pattern = f"{s3_path}/**/**/**/eval_h*.parquet"
paths = s3.glob(pattern)
for path in paths:
    dataset = path.split("/")[-3]
    model_name = path.split("/")[-2]
    loss_name = path.split("/")[-4]

    df = pl.read_parquet(f"s3://{path}", storage_options=storage_options)
    df = df.with_columns(pl.lit(model_name).alias("model"))
    df = df.with_columns(pl.lit(loss_name).alias("loss_name"))
    df = df.rename({f"{model_name}": "values"})

    horizon = df["horizon"].unique().item()

    # Check if the file already exists
    filename = f"eval/{loss_name}_{dataset}_{horizon}_{model_name}.parquet"
    if Path(filename).exists():
        print(f"File already exists: {filename}")
        continue
    else:
        print(f"Writing file: {filename}")
        df.write_parquet(
        f"{filename}",
    )
#%% Plot 1: overall results
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
df = pl.scan_parquet("eval/*.parquet")
metrics = ["scaled_crps"]
new_loss_name = "method"
final_metric_name = "CRPS"
df = df.filter(pl.col("metric").is_in(metrics))
df = df.rename({"loss_name": new_loss_name})
df_base = df.filter(pl.col(new_loss_name) == "normal")
df = df.join(df_base, on=["dataset", "horizon", "model", "metric"], suffix="_base")
df = df.with_columns(
    (pl.col("values") / pl.col("values_base")).alias(final_metric_name),
)
df_stats = df.group_by([new_loss_name])\
            .agg(
                pl.col(final_metric_name).mean().alias(f"{final_metric_name}_ratio"),
            ).collect()

fig, ax = plt.subplots(figsize=(20, 10))
sns.set_theme(context={'font.size': 16})
sns.set_style("dark")
sns.boxplot(
    data=df.collect().to_pandas(),
    x=new_loss_name,
    y=final_metric_name,
    showfliers=False,
    order=["normal", "studentt", "gmm", "hubermqloss", "iqloss", "iqf", "isqf", "conformal"],
    ax=ax,
    )
plt.xticks(rotation=30)
#%% Plot 2: results by dataset
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
df = pl.scan_parquet("eval/*.parquet")
metrics = ["scaled_crps"]
new_loss_name = "method"
final_metric_name = "CRPS"
df = df.filter(pl.col("metric").is_in(metrics))
df = df.rename({"loss_name": new_loss_name})
df_base = df.filter(pl.col(new_loss_name) == "normal")
df = df.join(df_base, on=["dataset", "horizon", "model", "metric"], suffix="_base")
df = df.with_columns(
    (pl.col("values") / pl.col("values_base")).alias(final_metric_name),
)
df_stats = df.group_by([new_loss_name])\
            .agg(
                pl.col(final_metric_name).mean().alias(f"{final_metric_name}_ratio"),
            ).collect()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
sns.set_theme(context={'font.size': 16})
sns.set_style("dark")
# Iterate over datasets
datasets = df.collect()["dataset"].unique().to_list()
for i, dataset in enumerate(datasets):
    ax = axs[i // 3, i % 3]
    df_dataset = df.filter(pl.col("dataset") == dataset)
    g = sns.boxplot(
        data=df_dataset.collect().to_pandas(),
        x=new_loss_name,
        y=final_metric_name,
        showfliers=False,
        order=["normal", "studentt", "gmm", "hubermqloss", "iqloss", "iqf", "isqf", "conformal"],
        ax=ax,
    )
    ax.set_title(dataset)
    if i < 6:
        g.set(xticklabels="")
        g.set(xlabel="")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_ylim(0, 3)
#%% Plot 3: results by model
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
df = pl.scan_parquet("eval/*.parquet")
metrics = ["scaled_crps"]
new_loss_name = "method"
final_metric_name = "CRPS"
df = df.filter(pl.col("metric").is_in(metrics))
df = df.rename({"loss_name": new_loss_name})
df_base = df.filter(pl.col(new_loss_name) == "normal")
df = df.join(df_base, on=["dataset", "horizon", "model", "metric"], suffix="_base")
df = df.with_columns(
    (pl.col("values") / pl.col("values_base")).alias(final_metric_name),
)
df_stats = df.group_by([new_loss_name])\
            .agg(
                pl.col(final_metric_name).mean().alias(f"{final_metric_name}_ratio"),
            ).collect()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
sns.set_theme(context={'font.size': 16})
sns.set_style("dark")
# Iterate over datasets
models =['BiTCN',
 'DLinear',
 'DeepAR',
 'LSTM',
 'MLP',
 'MLPMultivariate',
 'NBEATS',
 'NHITS',
 'NLinear']
# models = [
#  'PatchTST',
#  'TFT',
#  'TSMixer',
#  'TiDE',
#  'TimeMixer',
#  'TimesNet',
#  'VanillaTransformer',
#  'iTransformer']
for i, model in enumerate(models):
    ax = axs[i // 3, i % 3]
    df_dataset = df.filter(pl.col("model") == model)
    g = sns.boxplot(
        data=df_dataset.collect().to_pandas(),
        x=new_loss_name,
        y=final_metric_name,
        showfliers=False,
        order=["normal", "studentt", "gmm", "hubermqloss", "iqloss", "iqf", "isqf", "conformal"],
        ax=ax,
    )
    ax.set_title(model)
    if i < 6:
        g.set(xticklabels="")
        g.set(xlabel="")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_ylim(0, 3)
# axs[2, 2].axis('off')  # Hide the empty subplot
#%%
import seaborn as sns
import matplotlib.pyplot as plt
df = pl.scan_parquet("eval/*.parquet")
metrics = ["scaled_crps"]
new_loss_name = "method"
final_metric_name = "CRPS"
models = ["PatchTST"]
datasets = ["M5"]

df = df.filter(pl.col("metric").is_in(metrics))\
       .filter(pl.col("model").is_in(models))\
         .filter(pl.col("dataset").is_in(datasets))
df = df.rename({"loss_name": new_loss_name})
df_base = df.filter(pl.col(new_loss_name) == "normal")
df = df.join(df_base, on=["dataset", "horizon", "model", "metric"], suffix="_base")
df = df.with_columns(
    (pl.col("values") / pl.col("values_base")).alias(final_metric_name),
)
# df = df.group_by(["loss_name"])\
#         .agg(
#             pl.col("values_ratio").mean().alias("mean_ratio"),
#             pl.col("values_ratio").std().alias("std_ratio"),
#         )

sns.boxplot(
    data=df.collect().to_pandas(),
    x=new_loss_name,
    y=final_metric_name,
    showfliers=False,
    order=["normal", "studentt", "gmm", "hubermqloss", "iqloss", "iqf", "isqf", "conformal"],
    )
plt.xticks(rotation=30)

df.group_by([new_loss_name]).agg(pl.col(final_metric_name).median()).collect()
#%% Result per method in table
import numpy as np
df = pl.scan_parquet("eval/*.parquet")
metrics = ["coverage_level10", "coverage_level20", "coverage_level30", "coverage_level40", "coverage_level50", "coverage_level60", "coverage_level70", "coverage_level80", "coverage_level90"]
new_loss_name = "method"
new_metric_name = "coverage"
datasets = ["ETTm1"]

df = df.filter(pl.col("metric").is_in(metrics))\
       .filter(pl.col("dataset").is_in(datasets))
df = df.rename({"loss_name": new_loss_name, "values": new_metric_name}).collect()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
sns.set_theme(context={'font.size': 16})
sns.set_style("dark")
methods = ["normal", "studentt", "gmm", "hubermqloss", "iqloss", "iqf", "conformal"]
for i, method in enumerate(methods):
    ax = axs[i // 3, i % 3]
    df_dataset = df.filter(pl.col(new_loss_name) == method)
    g = sns.boxplot(
        data=df_dataset,
        x="metric",
        y=new_metric_name,
        showfliers=False,
        order=metrics,
        ax=ax,
    )
    ax.set_title(method, pad=-20)
    sns.lineplot(
        x=np.arange(0, 10, 1),
        y=np.arange(0, 1, 0.1),
        color='black',
        linestyle='--',
        ax=ax,
    )
    # if i < 6:
    #     g.set(xticklabels="")
    # else:
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    g.set(xlabel="")
    ax.set_ylim(0, 1)
axs[2, 1].axis('off')  # Hide the empty subplot
axs[2, 2].axis('off')  # Hide the empty subplot
