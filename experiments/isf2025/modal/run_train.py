import modal
import time

# from experiment_conformal import cross_validation
# from experiment_normal import cross_validation
# from experiment_studentt import cross_validation
# from experiment_hubermqloss import cross_validation
# from experiment_iqloss import cross_validation
# from experiments.isf2025.modal.experiment_iqf import cross_validation
from experiment_isqf import cross_validation
# from experiment_gmm import cross_validation

from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from utilsforecast.losses import mae, mse, smape, mqloss, scaled_mqloss, scaled_crps, mase, coverage

APP = modal.App("isf_2025")  # type:ignore
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")  # type: ignore
    .apt_install("git")
    .run_commands("pip install git+https://github.com/Nixtla/neuralforecast.git@f52f2ef534660016d7081f818b3e7b867421122d")  # type:ignore
    .pip_install("awscli", "s3fs", "polars", "utilsforecast", "datasetsforecast")
    .add_local_python_source("experiment_conformal")  # type:ignore
    .add_local_python_source("experiment_normal")  # type:ignore
    .add_local_python_source("experiment_studentt")  # type:ignore
    .add_local_python_source("experiment_hubermqloss")  # type:ignore
    .add_local_python_source("experiment_iqloss")  # type:ignore
    .add_local_python_source("experiment_iqf")  # type:ignore
    .add_local_python_source("experiment_isqf")  # type:ignore
    .add_local_python_source("experiment_gmm")  # type:ignore

)

S3_VOLUMES = {
    "/isf2025": modal.CloudBucketMount(  # type:ignore
        bucket_name="timenet", 
        key_prefix="isf2025/",
        secret=modal.Secret.from_dotenv(), 
    ),
}

@APP.function(
    image=IMAGE,
    secrets=[modal.Secret.from_dotenv()],
    volumes=S3_VOLUMES,
    gpu="H100",
    timeout=int(16 * 3600),
    cpu=4,
    memory=8 * 1024,
)
def evaluate_models(dataset, horizon) -> None:
    # Input_size and metrics to evaluate.
    metrics = [
        mse,
        mae,
        mqloss,
        scaled_mqloss,
        scaled_crps,
        mase,
        coverage,
    ]
    cross_validation(dataset, horizon, metrics, seed=1234567)

@APP.local_entrypoint()
def main(
):
    start = time.time()
    datasets = [
              'ETTh1',
              'ETTh2',
              'ETTm1',
              'ETTm2',
              'ECL',
              'TrafficL',
              'Weather',
              'ILI',
              'M5',
                ]

    experiments = []
    for dataset in datasets:
        if dataset == 'M5':
            horizons = [28]
        else:
            horizons = LongHorizonInfo[dataset].horizons
        for horizon in horizons:
            experiments.append((dataset, horizon))
    
    list(evaluate_models.starmap(experiments, return_exceptions=True))

    end = time.time()
    print(f"Training took {end - start:.2f} seconds")