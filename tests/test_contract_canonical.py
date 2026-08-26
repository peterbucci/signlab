from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
import rfc8785
from pydantic import BaseModel, ConfigDict

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.taxonomy import BUILTIN_TAXONOMY_DIGEST, load_builtin_taxonomy

SAFE_INTEGER_MAX = (2**53) - 1
SAFE_INTEGER_MIN = -SAFE_INTEGER_MAX


class ExampleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    z: int
    a: str


class OpaqueValue:
    pass


class NonJsonModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    value: OpaqueValue


@pytest.mark.golden
def test_rfc8785_sorts_object_names_by_utf16_code_units() -> None:
    # RFC 8785 section 3.2.3's sorting vector.
    payload = {
        "\u20ac": "Euro Sign",
        "\r": "Carriage Return",
        "\ufb33": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "\u00f6": "Latin Small Letter O With Diaeresis",
    }

    expected = (
        '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
        '"\u00f6":"Latin Small Letter O With Diaeresis","\u20ac":"Euro Sign",'
        '"\U0001f600":"Emoji: Grinning Face",'
        '"\ufb33":"Hebrew Letter Dalet With Dagesh"}'
    ).encode()
    assert canonical_json_bytes(payload) == expected


@pytest.mark.golden
def test_rfc8785_uses_ecmascript_number_spelling_and_normalizes_negative_zero() -> None:
    # The first five values are RFC 8785 section 3.2.2.3's number vector.
    payload = {
        "numbers": [
            333333333.33333329,
            1e30,
            4.50,
            2e-3,
            1e-27,
            -0.0,
            1e-6,
            1e-7,
            1e20,
            1e21,
        ]
    }

    assert canonical_json_bytes(payload) == (
        b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0,'
        b"0.000001,1e-7,100000000000000000000,1e+21]}"
    )


def test_canonicalization_orders_nested_objects_and_is_idempotent() -> None:
    payload = {
        "z": {"\u03b2": 2, "a": [{"z": -0.0, "a": 0}]},
        "a": 1,
    }

    first = canonical_json_bytes(payload)
    assert first == '{"a":1,"z":{"a":[{"a":0,"z":0}],"\u03b2":2}}'.encode()
    assert canonical_json_bytes(parse_json_object(first)) == first
    assert canonical_json_bytes(parse_json_object(first.decode())) == first


def test_canonicalization_accepts_validated_models() -> None:
    assert canonical_json_bytes(ExampleModel(z=2, a="value")) == b'{"a":"value","z":2}'


@pytest.mark.parametrize(
    "document",
    [
        '{"enabled":true,"nested":{"value":null}}',
        b'{"enabled":true,"nested":{"value":null}}',
        bytearray(b'{"enabled":true,"nested":{"value":null}}'),
        {"enabled": True, "nested": {"value": None}},
    ],
)
def test_parse_json_object_accepts_each_supported_input_type(
    document: str | bytes | bytearray | Mapping[str, object],
) -> None:
    assert parse_json_object(document) == {"enabled": True, "nested": {"value": None}}


@pytest.mark.parametrize(
    "document",
    [
        '{"secret":1,"secret":2}',
        b'{"outer":{"secret":1,"secret":2}}',
        '{"a":1,"\\u0061":2}',
    ],
)
def test_raw_json_rejects_duplicate_names_without_echoing_them(document: str | bytes) -> None:
    with pytest.raises(CanonicalizationError, match="duplicate member names") as caught:
        parse_json_object(document)

    assert "secret" not in str(caught.value)


def test_raw_bytes_must_be_utf8() -> None:
    with pytest.raises(CanonicalizationError, match="valid UTF-8"):
        parse_json_object(b'{"value":"\xff"}')


@pytest.mark.parametrize("document", ["{private", b"{private"])
def test_invalid_json_syntax_is_sanitized(document: str | bytes) -> None:
    with pytest.raises(CanonicalizationError, match="invalid JSON document") as caught:
        parse_json_object(document)

    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "document",
    [
        '{"value":"\\ud800"}',
        {"value": "\udfff"},
        {"\ud800": "value"},
    ],
)
def test_lone_surrogates_are_rejected(
    document: str | Mapping[str, object],
) -> None:
    with pytest.raises(CanonicalizationError, match="Unicode scalar"):
        parse_json_object(document)


@pytest.mark.parametrize("document", ["[]", "null", "1", '"value"'])
def test_parse_json_requires_a_top_level_object(document: str) -> None:
    with pytest.raises(CanonicalizationError, match="top-level JSON value must be an object"):
        parse_json_object(document)


def test_parse_json_rejects_unsupported_document_types() -> None:
    with pytest.raises(CanonicalizationError, match="UTF-8 JSON or a mapping"):
        parse_json_object(Path("not-json"))  # type: ignore[arg-type]


def test_safe_integer_bounds_are_enforced_for_raw_and_mapping_inputs() -> None:
    assert parse_json_object(f'{{"maximum":{SAFE_INTEGER_MAX},"minimum":{SAFE_INTEGER_MIN}}}') == {
        "maximum": SAFE_INTEGER_MAX,
        "minimum": SAFE_INTEGER_MIN,
    }
    assert canonical_json_bytes({"maximum": SAFE_INTEGER_MAX, "minimum": SAFE_INTEGER_MIN})

    for unsafe in (SAFE_INTEGER_MAX + 1, SAFE_INTEGER_MIN - 1):
        with pytest.raises(CanonicalizationError, match="safe-integer range"):
            parse_json_object(f'{{"value":{unsafe}}}')
        with pytest.raises(CanonicalizationError, match="safe-integer range"):
            canonical_json_bytes({"value": unsafe})


@pytest.mark.parametrize("document", ['{"value":NaN}', '{"value":Infinity}', '{"value":1e400}'])
def test_raw_json_rejects_nonfinite_numbers(document: str) -> None:
    with pytest.raises(CanonicalizationError, match="numbers must be finite"):
        parse_json_object(document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_mapping_canonicalization_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(CanonicalizationError, match="numbers must be finite"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {1: "value"}},
        {"value": ("tuple",)},
        {"value": {"set"}},
        {"value": b"bytes"},
        {"value": Path("not-json")},
    ],
)
def test_canonicalization_rejects_non_json_mappings(payload: Mapping[str, object]) -> None:
    with pytest.raises(CanonicalizationError, match=r"strings|non-JSON"):
        canonical_json_bytes(payload)


def test_reference_cycles_fail_closed() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    with pytest.raises(CanonicalizationError, match="reference cycles"):
        canonical_json_bytes(payload)


def test_list_reference_cycles_fail_closed() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(CanonicalizationError, match="reference cycles"):
        canonical_json_bytes({"recursive": recursive})


def test_excessive_mapping_nesting_is_sanitized() -> None:
    payload: dict[str, object] = {}
    cursor = payload
    for _ in range(1_100):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested

    with pytest.raises(CanonicalizationError, match="nesting is too deep"):
        parse_json_object(payload)


def test_canonicalization_rejects_non_object_inputs_at_runtime() -> None:
    with pytest.raises(CanonicalizationError, match="model or mapping"):
        canonical_json_bytes(["not", "an", "object"])  # type: ignore[arg-type]


def test_model_serialization_errors_are_sanitized() -> None:
    with pytest.raises(CanonicalizationError, match="cannot be represented as JSON") as caught:
        canonical_json_bytes(NonJsonModel(value=OpaqueValue()))

    assert "OpaqueValue" not in str(caught.value)


def test_dependency_errors_are_translated_and_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_canonicalization(_value: object) -> bytes:
        raise rfc8785.CanonicalizationError("sensitive dependency detail")

    monkeypatch.setattr(rfc8785, "dumps", fail_canonicalization)
    with pytest.raises(CanonicalizationError, match="RFC 8785 canonicalization failed") as caught:
        canonical_json_bytes({"value": 1})

    assert "sensitive" not in str(caught.value)


def test_canonical_sha256_is_exact_and_domain_separated() -> None:
    payload = {"z": 2, "a": 1}
    canonical = b'{"a":1,"z":2}'
    expected = hashlib.sha256(b"signlab.contract/example/1\0" + canonical).hexdigest()

    assert canonical_sha256(payload, domain="example/1") == f"sha256:{expected}"
    assert canonical_sha256(payload, domain="example/1") != canonical_sha256(
        payload, domain="other/1"
    )


@pytest.mark.parametrize("domain", ["", "contains space", "non-ascii-\u00e9", "nul\0byte"])
def test_canonical_sha256_rejects_unsafe_domains(domain: str) -> None:
    with pytest.raises(CanonicalizationError, match="non-empty printable ASCII"):
        canonical_sha256({"value": 1}, domain=domain)


@pytest.mark.golden
def test_rfc8785_preserves_the_published_builtin_taxonomy_digest() -> None:
    digest = hashlib.sha256(canonical_json_bytes(load_builtin_taxonomy())).hexdigest()
    assert f"sha256:{digest}" == BUILTIN_TAXONOMY_DIGEST
