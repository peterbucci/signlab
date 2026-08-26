"""Strict I-JSON parsing and RFC 8785 canonical identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Final, NoReturn

import rfc8785
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

_SAFE_INTEGER_MAX: Final = (2**53) - 1
_SAFE_INTEGER_MIN: Final = -_SAFE_INTEGER_MAX
_DIGEST_PREFIX: Final = b"signlab.contract/"

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type JsonDocument = str | bytes | bytearray | Mapping[str, object]
type CanonicalValue = BaseModel | Mapping[str, object]


class CanonicalizationError(ValueError):
    """Raised when input cannot safely participate in a portable identity."""


def _reject_nonfinite_constant(_value: str) -> NoReturn:
    raise CanonicalizationError("JSON numbers must be finite")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError("JSON objects must not contain duplicate member names")
        result[key] = value
    return result


def _validate_unicode_scalar_text(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise CanonicalizationError(
            "JSON strings must contain only valid Unicode scalar values"
        ) from error
    return value


def _normalize_json_value(value: object, active_containers: set[int]) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _validate_unicode_scalar_text(value)
    if type(value) is int:
        if value < _SAFE_INTEGER_MIN or value > _SAFE_INTEGER_MAX:
            raise CanonicalizationError(
                "JSON integers must be within the interoperable safe-integer range"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise CanonicalizationError("JSON values must not contain reference cycles")
        active_containers.add(identity)
        try:
            normalized: JsonObject = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise CanonicalizationError("JSON object member names must be strings")
                normalized[_validate_unicode_scalar_text(key)] = _normalize_json_value(
                    nested, active_containers
                )
            return normalized
        finally:
            active_containers.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise CanonicalizationError("JSON values must not contain reference cycles")
        active_containers.add(identity)
        try:
            return [_normalize_json_value(item, active_containers) for item in value]
        finally:
            active_containers.remove(identity)
    raise CanonicalizationError("canonical input contains a non-JSON value")


def parse_json_object(document: JsonDocument) -> JsonObject:
    """Parse one duplicate-safe UTF-8 I-JSON object into plain JSON values."""

    parsed: object
    if isinstance(document, Mapping):
        parsed = document
    elif isinstance(document, (bytes, bytearray)):
        try:
            decoded = bytes(document).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise CanonicalizationError("JSON bytes must use valid UTF-8") from error
        try:
            parsed = json.loads(
                decoded,
                object_pairs_hook=_object_from_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except CanonicalizationError:
            raise
        except (json.JSONDecodeError, RecursionError) as error:
            raise CanonicalizationError("invalid JSON document") from error
    elif isinstance(document, str):
        try:
            parsed = json.loads(
                document,
                object_pairs_hook=_object_from_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except CanonicalizationError:
            raise
        except (json.JSONDecodeError, RecursionError) as error:
            raise CanonicalizationError("invalid JSON document") from error
    else:
        raise CanonicalizationError("JSON input must be UTF-8 JSON or a mapping")

    if not isinstance(parsed, Mapping):
        raise CanonicalizationError("the top-level JSON value must be an object")
    try:
        normalized = _normalize_json_value(parsed, set())
    except RecursionError as error:
        raise CanonicalizationError("JSON document nesting is too deep") from error
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by the Mapping check.
        raise CanonicalizationError("the top-level JSON value must be an object")
    return normalized


def canonical_json_bytes(value: CanonicalValue) -> bytes:
    """Return validated RFC 8785 JSON bytes for a model or JSON object."""

    payload: Mapping[str, object]
    if isinstance(value, BaseModel):
        try:
            payload = value.model_dump(mode="json", round_trip=True)
        except (PydanticSerializationError, TypeError, ValueError) as error:
            raise CanonicalizationError("model cannot be represented as JSON") from error
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise CanonicalizationError("canonical input must be a model or mapping")

    normalized = parse_json_object(payload)
    try:
        return rfc8785.dumps(normalized)
    except rfc8785.CanonicalizationError as error:
        raise CanonicalizationError("RFC 8785 canonicalization failed") from error


def canonical_sha256(value: CanonicalValue, *, domain: str) -> str:
    """Return a domain-separated SHA-256 identity over canonical JSON bytes."""

    if (
        not isinstance(domain, str)
        or not domain
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in domain)
    ):
        raise CanonicalizationError("canonical digest domain must be non-empty printable ASCII")
    digest_input = _DIGEST_PREFIX + domain.encode("ascii") + b"\0" + canonical_json_bytes(value)
    return f"sha256:{hashlib.sha256(digest_input).hexdigest()}"


__all__ = [
    "CanonicalizationError",
    "canonical_json_bytes",
    "canonical_sha256",
    "parse_json_object",
]
