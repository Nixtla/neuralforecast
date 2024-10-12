import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

import mlforecast
import mlforecast.flavor
from mlforecast import MLForecast


FLAVOR_NAME = "mlforecast"
_MODEL_DATA_SUBPATH = "mlforecast-model"


def get_default_pip_requirements():
    """Create list of default pip requirements for MLflow Models.

    Returns
    -------
    list of default pip requirements for MLflow Models produced by this flavor.
    Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
    that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("mlforecast")]


def get_default_conda_env():
    """Return default Conda environment for MLflow Models.

    Returns
    -------
    The default Conda environment for MLflow Models produced by calls to
    :func:`save_model()` and :func:`log_model()`
    """
    return _mlflow_conda_env(additional_conda_deps=get_default_pip_requirements())


def save_model(
    model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """Save an ``MLForecast`` model to a local path

    Parameters
    ----------
    model : MLForecast
        Fitted ``MLForecast`` model object.
    path : str
        Local path where the model is to be saved.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    mlflow_model: mlflow.models.Model, optional (default=None)
        mlflow.models.Model configuration to which to add the python_function flavor.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["mlforecast", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_path = os.path.join(path, _MODEL_DATA_SUBPATH)
    model.save(model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlforecast.flavor",
        model_path=_MODEL_DATA_SUBPATH,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=_MODEL_DATA_SUBPATH,
        mlforecast_version=mlforecast.__version__,
        serialization_format="cloudpickle",
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def log_model(
    model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log an ``MLForecast`` model as an MLflow artifact for the current run.

    Parameters
    ----------
    model : MLForecast
        Fitted ``MLForecast`` model object.
    artifact_path : str
        Run-relative artifact path to save the model to.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    registered_model_name : str, optional (default=None)
        If given, create a model version under ``registered_model_name``, also creating
        a registered model if one with the given name does not exist.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    await_registration_for : int, optional (default=None)
        Number of seconds to wait for the model version to finish being created and is
        in ``READY`` status. By default, the function waits for five minutes. Specify 0
        or None to skip waiting.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["mlforecast", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    kwargs:
        Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns
    -------
    A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
    metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlforecast.flavor,
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    Load an ``MLForecast`` model from a local file or a run.

    Parameters
    ----------
    model_uri : str
        The location, in URI format, of the MLflow model. For example:

                    - ``/Users/me/path/to/local/model``
                    - ``relative/path/to/local/model``
                    - ``s3://my_bucket/path/to/model``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                    - ``mlflow-artifacts:/path/to/model``

        For more information about supported URI schemes, see
        `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
        artifact-locations>`_.
    dst_path : str, optional (default=None)
        The local filesystem path to which to download the model artifact.This
        directory must already exist. If unspecified, a local output path will
        be created.

    Returns
    -------
    An ``MLForecast`` model instance.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_file_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    return MLForecast.load(model_file_path)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Parameters
    ----------
    path : str
        Local filesystem path to the MLflow Model with the ``mlforecast`` flavor.

    """
    pyfunc_flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=pyfunc.FLAVOR_NAME)
    path = os.path.join(path, pyfunc_flavor_conf["model_path"])
    return _MLForecastModelWrapper(MLForecast.load(path))


class _MLForecastModelWrapper:
    def __init__(self, model: MLForecast):
        self.model = model

    def predict(
        self,
        config_df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,  # noqa
    ) -> pd.DataFrame:
        n_rows = config_df.shape[0]

        if n_rows > 1:
            raise MlflowException(
                f"The provided prediction DataFrame contains {n_rows} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        attrs = config_df.iloc[0].to_dict()
        h = attrs.get("h")
        if h is None:
            raise MlflowException(
                "The `h` parameter is required to make forecasts.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        ts = self.model.ts
        col_types = {
            ts.id_col: ts.uids.dtype,
            ts.time_col: ts.last_dates.dtype,
        }
        level = attrs.get("level")
        new_df = attrs.get("new_df")
        if new_df is not None:
            if level is not None:
                raise MlflowException(
                    "Prediction intervals are not supported in transfer learning. "
                    "Please provide either `level` or `new_df`, but not both.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            new_df = pd.DataFrame(new_df).astype(col_types)
        X_df = attrs.get("X_df")
        if X_df is not None:
            X_df = pd.DataFrame(X_df).astype(col_types)
        ids = attrs.get("ids")
        return self.model.predict(h=h, new_df=new_df, level=level, X_df=X_df, ids=ids)
