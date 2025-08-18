


__all__ = []


try:
    from pyspark.sql import DataFrame as SparkDataFrame
except ImportError:

    class SparkDataFrame: ...
