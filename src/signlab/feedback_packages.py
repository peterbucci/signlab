"""Strict import boundary for consented browser feedback packages."""

# ruff: noqa: E501, E731 -- compact, auditable mirrors of the browser rule tables.

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn, cast

_PACKAGE = "signlab-feedback-package/1"
_RECORD_FORMAT = "signlab-feedback-record/1"
_EXPORT_CONSENT = "signlab-feedback-export-consent/1"
_RECEIPT = "signlab-feedback-quarantine-receipt/1"
_LABELS = ("hello", "no", "please", "thank_you", "yes", "other", "inactive")
_HANDS = ("hand_0", "hand_1")
# fmt: off
_ANCHORS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
# fmt: on
_MAX_SAFE_INTEGER = (2**53) - 1


class FeedbackPackageError(ValueError):
    """Sanitized rejection safe to surface at the command boundary."""

    def __init__(self, code: str = "invalid") -> None:
        self.code = code
        super().__init__(f"feedback package {code}")


@dataclass(frozen=True, slots=True)
class FeedbackPackageImport:
    """Aggregate result of one newly published quarantine package."""

    package_sha256: str
    record_count: int
    destination: Path


type Predicate = Callable[[object], bool]
type Rule = Predicate | dict[str, Rule] | tuple[Rule, ...] | list[Rule]


def _fail() -> NoReturn:
    raise FeedbackPackageError()


def _require(condition: object) -> None:
    if not condition:
        _fail()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            _fail()
        result[key] = value
    return result


def _parse(document: bytes | str) -> object:
    try:
        source = document.decode("utf-8", errors="strict") if isinstance(document, bytes) else document  # fmt: skip
        return json.loads(
            source,
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: _fail(),
        )
    except FeedbackPackageError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise FeedbackPackageError() from error


_text: Predicate = lambda value: isinstance(value, str)


def _finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(cast(float, value))
    except OverflowError:
        return False


_nonnegative: Predicate = lambda value: _finite(value) and cast(float, value) >= 0
_whole: Predicate = lambda value: type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER
_probability: Predicate = lambda value: _nonnegative(value) and cast(float, value) <= 1


def _iso(value: object) -> bool:
    try:
        return isinstance(value, str) and len(value) == 24 and datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").isoformat(timespec="milliseconds") + "Z" == value  # fmt: skip
    except ValueError:
        return False


def _one(*values: str) -> Predicate:
    return lambda value: type(value) is str and value in values


def _nullable(rule: Rule) -> Predicate:
    return lambda value: value is None or _matches(value, rule)


def _matches(value: object, rule: Rule) -> bool:
    if callable(rule):
        return rule(value)
    if isinstance(rule, dict):
        if type(value) is not dict:
            return False
        checked = cast(dict[str, object], value)
        return checked.keys() == rule.keys() and all(
            _matches(checked[key], child) for key, child in rule.items()
        )
    if isinstance(rule, tuple):
        return (
            isinstance(value, list)
            and len(value) == len(rule)
            and all(_matches(item, child) for item, child in zip(value, rule, strict=True))
        )
    return isinstance(value, list) and all(_matches(item, rule[0]) for item in value)


_bool: Predicate = lambda value: type(value) is bool
# fmt: off
_POINT: Rule = {"x": _finite, "y": _finite, "z": _finite, "visibility": _nullable(_probability), "presence": _nullable(_probability)}

def _hand(slot: str) -> Rule:
    return {"slotId": _one(slot), "present": _bool, "detectorIndex": _nullable(_whole), "trackingId": _nullable(_one(*_HANDS)), "handedness": _nullable(_one("left", "right")), "handednessConfidence": _nullable(_probability), "imageLandmarks": _nullable([_POINT]), "worldLandmarks": _nullable([_POINT])}

def _anchor(name: str) -> Rule:
    return {"name": _one(name), "present": _bool, "imagePoint": _nullable(_POINT), "worldPoint": _nullable(_POINT)}

_FRAME: Rule = {"relativeTimestampUs": _whole, "valid": _bool, "hands": tuple(_hand(slot) for slot in _HANDS), "bodyAnchors": tuple(_anchor(name) for name in _ANCHORS)}

def _decision(value: object) -> bool:
    rules: tuple[Rule, ...] = ({"kind": _one("abstain")}, {"kind": _one("other"), "label": _one("other"), "confidence": _probability}, {"kind": _one("target"), "label": _one(*_LABELS[:5]), "confidence": _probability})
    return any(_matches(value, rule) for rule in rules)

_RECORD: dict[str, Rule] = {
    "format": _one(_RECORD_FORMAT), "id": _text, "savedAt": _iso,
    "bundle": {"id": _text, "version": _text},
    "prediction": {"decision": _decision, "reason": _one("accepted_target", "accepted_other", "below_threshold")},
    "scores": [{"label": _one(*_LABELS[:6]), "confidence": _probability}],
    "timings": {"preprocessingMs": _nonnegative, "inferenceMs": _nonnegative, "decisionMs": _nonnegative, "totalMs": _nonnegative},
    "event": {"firstFrameIndex": _whole, "lastFrameIndex": _whole, "firstTimestampUs": _whole, "lastTimestampUs": _whole, "terminationReason": _one("settled", "signal_gap", "max_duration", "stream_end"), "configSha256": _text, "durationUs": _nonnegative},
    "conditions": {"sourceMirrorState": _one("mirrored", "not_mirrored"), "previewMirrored": _bool},
    "correction": _one(*_LABELS),
    "consent": {"mode": _one("per_event"), "scope": _one("local_feedback_only"), "grantedAt": _iso, "landmarksIncluded": _bool},
}


def _record(value: object) -> dict[str, object]:
    _require(type(value) is dict)
    checked = cast(dict[str, object], value)
    has_landmarks = "landmarks" in checked
    _require(_matches(checked, {**_RECORD, **({"landmarks": [_FRAME]} if has_landmarks else {})}))
    consent, scores, event = cast(dict[str, object], checked["consent"]), cast(list[dict[str, object]], checked["scores"]), cast(dict[str, object], checked["event"])
    _require(consent["landmarksIncluded"] is has_landmarks and consent["grantedAt"] == checked["savedAt"] and len(scores) == 6 and len({score["label"] for score in scores}) == 6)
    _require(cast(int, event["lastFrameIndex"]) >= cast(int, event["firstFrameIndex"]) and event["durationUs"] == cast(int, event["lastTimestampUs"]) - cast(int, event["firstTimestampUs"]))
    return checked
# fmt: on


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _receipt(package_bytes: bytes) -> dict[str, object]:
    package = _parse(package_bytes)
    _require(type(package) is dict)
    package = cast(dict[str, object], package)
    _require(package.keys() == {"format", "manifest", "payloadJson", "payloadSha256"})
    _require(package["format"] == _PACKAGE and _text(package["payloadJson"]))
    try:
        payload_bytes = cast(str, package["payloadJson"]).encode("utf-8")
    except UnicodeEncodeError as error:
        raise FeedbackPackageError() from error
    _require(package["payloadSha256"] == _digest(payload_bytes))
    payload = _parse(payload_bytes)
    _require(isinstance(payload, list))
    records = [_record(value) for value in cast(list[object], payload)]
    ids = [cast(str, record["id"]) for record in records]
    _require(ids == sorted(ids) and len(ids) == len(set(ids)))

    manifest = package["manifest"]
    _require(type(manifest) is dict)
    manifest = cast(dict[str, object], manifest)
    manifest_keys = {"recordFormat", "recordCount", "bundleVersions", "fields", "localConsentScope", "landmarkRecordCount", "landmarksIncluded", "exportConsent"}  # fmt: skip
    _require(manifest.keys() == manifest_keys)
    pairs = {
        (
            cast(dict[str, object], record["bundle"])["id"],
            cast(dict[str, object], record["bundle"])["version"],
        )
        for record in records
    }
    bundles = [{"id": identity, "version": version} for identity, version in sorted(pairs)]
    fields = sorted({field for record in records for field in record})
    landmark_count = sum("landmarks" in record for record in records)
    _require(manifest["recordFormat"] == _RECORD_FORMAT)
    _require(_whole(manifest["recordCount"]) and manifest["recordCount"] == len(records))
    _require(manifest["bundleVersions"] == bundles and manifest["fields"] == fields)
    _require(manifest["localConsentScope"] == "local_feedback_only")
    _require(_whole(manifest["landmarkRecordCount"]))
    _require(manifest["landmarkRecordCount"] == landmark_count)
    _require(type(manifest["landmarksIncluded"]) is bool)
    _require(cast(bool, manifest["landmarksIncluded"]) or landmark_count == 0)

    consent = manifest["exportConsent"]
    _require(type(consent) is dict)
    consent = cast(dict[str, object], consent)
    consent_keys = {"statementVersion", "granted", "grantedAt", "scope", "trainingUse"}
    _require(consent.keys() == consent_keys)
    _require(consent["statementVersion"] == _EXPORT_CONSENT and consent["granted"] is True)
    _require(_iso(consent["grantedAt"]) and consent["scope"] == "manual_research_review")
    _require(consent["trainingUse"] == "requires_review_and_approval")
    # fmt: off
    return {
        "format": _RECEIPT, "packageSha256": _digest(package_bytes),
        "payloadSha256": package["payloadSha256"], "recordFormat": _RECORD_FORMAT,
        "recordCount": len(records), "bundleVersions": bundles,
        "landmarkRecordCount": landmark_count, "landmarksIncluded": manifest["landmarksIncluded"],
        "exportConsent": {"statementVersion": consent["statementVersion"], "scope": consent["scope"], "grantedAt": consent["grantedAt"]},
        "trainable": False,
        "remainingGates": ["human_review", "cross_package_deduplication", "split_decision", "new_dvc_version"],
    }
    # fmt: on


def import_feedback_package(
    package: str | Path,
    *,
    quarantine_root: str | Path = Path("data/private/feedback-quarantine"),
) -> FeedbackPackageImport:
    """Validate exact bytes, then publish one immutable quarantine directory."""

    try:
        package_bytes = Path(package).read_bytes()
    except OSError as error:
        raise FeedbackPackageError() from error
    receipt = _receipt(package_bytes)
    root = Path(quarantine_root)
    digest = cast(str, receipt["packageSha256"])
    destination = root / digest.replace(":", "-")
    try:
        root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FeedbackPackageError("duplicate")
        staging = Path(tempfile.mkdtemp(prefix=".feedback-package-", dir=root))
        try:
            (staging / "package.signlab-feedback.json").write_bytes(package_bytes)
            serialized = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            (staging / "receipt.json").write_bytes(serialized)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    except FeedbackPackageError:
        raise
    except OSError as error:
        if destination.exists():
            raise FeedbackPackageError("duplicate") from error
        raise FeedbackPackageError("storage_failed") from error
    return FeedbackPackageImport(
        package_sha256=digest,
        record_count=cast(int, receipt["recordCount"]),
        destination=destination,
    )


__all__ = ["FeedbackPackageError", "FeedbackPackageImport", "import_feedback_package"]
