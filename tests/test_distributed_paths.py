from unittest.mock import MagicMock, patch

import pytest

from neuralforecast.core import (
    _as_distributed_file_uri,
    _dbfs_path_for_pandas,
    _fsspec_entry_path,
    _fsspec_listdir,
    _list_distributed_parquet_files,
)


def test_fsspec_entry_path_from_string():
    assert _fsspec_entry_path("dbfs:/tmp/part-0.parquet") == "dbfs:/tmp/part-0.parquet"


def test_fsspec_entry_path_from_detail_dict():
    assert (
        _fsspec_entry_path(
            {"name": "/tmp/part-0.parquet", "type": "file", "size": 123}
        )
        == "/tmp/part-0.parquet"
    )


def test_fsspec_entry_path_missing_name_raises():
    with pytest.raises(ValueError, match="Cannot determine path"):
        _fsspec_entry_path({"type": "file", "size": 1})


def test_fsspec_listdir_normalizes_detail_dicts():
    fs = MagicMock()
    fs.ls.return_value = [
        {"name": "/tmp/part-0.parquet", "type": "file", "size": 10},
        {"name": "/tmp/_SUCCESS", "type": "file", "size": 0},
    ]
    assert _fsspec_listdir(fs, "dbfs:/tmp") == [
        "/tmp/part-0.parquet",
        "/tmp/_SUCCESS",
    ]
    fs.ls.assert_called_once_with("dbfs:/tmp", detail=False)


def test_fsspec_listdir_falls_back_when_detail_unsupported():
    fs = MagicMock()
    fs.ls.side_effect = [
        TypeError("detail"),
        ["s3://bucket/part-0.parquet", "s3://bucket/_SUCCESS"],
    ]
    assert _fsspec_listdir(fs, "s3://bucket") == [
        "s3://bucket/part-0.parquet",
        "s3://bucket/_SUCCESS",
    ]


def test_dbfs_path_for_pandas_uses_fuse_when_available():
    with patch("os.path.isdir", return_value=True):
        assert (
            _dbfs_path_for_pandas("dbfs:/FileStore/partitions/part-0.parquet")
            == "/dbfs/FileStore/partitions/part-0.parquet"
        )
        assert (
            _dbfs_path_for_pandas("dbfs:///FileStore/partitions/part-0.parquet")
            == "/dbfs/FileStore/partitions/part-0.parquet"
        )


def test_dbfs_path_for_pandas_keeps_uri_without_fuse():
    with patch("os.path.isdir", return_value=False):
        assert (
            _dbfs_path_for_pandas("dbfs:/FileStore/partitions/part-0.parquet")
            == "dbfs:/FileStore/partitions/part-0.parquet"
        )


def test_as_distributed_file_uri_for_dbfs_and_s3():
    with patch("os.path.isdir", return_value=False):
        assert (
            _as_distributed_file_uri("dbfs", "/FileStore/part-0.parquet")
            == "dbfs:/FileStore/part-0.parquet"
        )
        assert (
            _as_distributed_file_uri("s3", "bucket/part-0.parquet")
            == "s3://bucket/part-0.parquet"
        )


def test_list_distributed_parquet_files_handles_dbfs_detail_dicts():
    fs = MagicMock()
    fs.protocol = "dbfs"
    fs.ls.return_value = [
        {
            "name": "/FileStore/distributed/partitions/part-000.parquet",
            "type": "file",
            "size": 421,
        },
        {
            "name": "/FileStore/distributed/partitions/_SUCCESS",
            "type": "file",
            "size": 0,
        },
        {
            "name": "/FileStore/distributed/partitions/part-001.parquet",
            "type": "file",
            "size": 224,
        },
    ]

    with patch("os.path.isdir", return_value=True):
        files = _list_distributed_parquet_files(
            fs, "dbfs:/FileStore/distributed/partitions"
        )

    assert files == [
        "/dbfs/FileStore/distributed/partitions/part-000.parquet",
        "/dbfs/FileStore/distributed/partitions/part-001.parquet",
    ]


def test_list_distributed_parquet_files_keeps_string_paths():
    fs = MagicMock()
    fs.protocol = ("s3", "s3a")
    fs.ls.return_value = [
        "bucket/partitions/part-000.parquet",
        "bucket/partitions/_committed_123",
    ]

    files = _list_distributed_parquet_files(fs, "s3://bucket/partitions")
    assert files == ["s3://bucket/partitions/part-000.parquet"]
