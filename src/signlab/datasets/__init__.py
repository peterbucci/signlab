"""Validated, deterministic storage adapters for SignLab datasets."""

from signlab.datasets.bundle import (
    DatasetBundleError,
    DatasetBundleValidationResult,
    WrittenDatasetBundle,
    validate_dataset_bundle,
    write_dataset_bundle,
)
from signlab.datasets.parquet import (
    DATASET_PARQUET_SCHEMAS,
    DATASET_PARQUET_SCHEMAS_BY_TABLE,
    DatasetParquetError,
    ParquetTableResult,
    build_dataset_table,
    parquet_schema_snapshot,
    read_dataset_table,
    read_parquet_table,
    resolve_workspace_locator,
    semantic_table_sha256,
    write_dataset_table,
    write_parquet_table,
)
from signlab.datasets.validation import (
    DatasetValidationError,
    DatasetValidationResult,
    validate_dataset_manifest_tables,
)

__all__ = [
    "DATASET_PARQUET_SCHEMAS",
    "DATASET_PARQUET_SCHEMAS_BY_TABLE",
    "DatasetBundleError",
    "DatasetBundleValidationResult",
    "DatasetParquetError",
    "DatasetValidationError",
    "DatasetValidationResult",
    "ParquetTableResult",
    "WrittenDatasetBundle",
    "build_dataset_table",
    "parquet_schema_snapshot",
    "read_dataset_table",
    "read_parquet_table",
    "resolve_workspace_locator",
    "semantic_table_sha256",
    "validate_dataset_bundle",
    "validate_dataset_manifest_tables",
    "write_dataset_bundle",
    "write_dataset_table",
    "write_parquet_table",
]
