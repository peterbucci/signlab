"""Cross-platform regression tests for shared contract primitives."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from signlab.contracts.core import UtcTimestamp

_UTC_TIMESTAMP_ADAPTER = TypeAdapter(UtcTimestamp)


@pytest.mark.parametrize(
    "timestamp",
    [
        "0001-01-01T00:00:00Z",
        "0999-12-31T23:59:59Z",
    ],
)
def test_utc_timestamp_accepts_zero_padded_early_years(timestamp: str) -> None:
    assert _UTC_TIMESTAMP_ADAPTER.validate_python(timestamp, strict=True) == timestamp


@pytest.mark.parametrize(
    "timestamp",
    [
        "1-01-01T00:00:00Z",
        "0001-02-29T00:00:00Z",
    ],
)
def test_utc_timestamp_rejects_noncanonical_or_invalid_early_dates(timestamp: str) -> None:
    with pytest.raises(ValidationError):
        _UTC_TIMESTAMP_ADAPTER.validate_python(timestamp, strict=True)
