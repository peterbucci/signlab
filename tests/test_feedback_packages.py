"""Browser-package quarantine tests kept independent of private participant data."""

# ruff: noqa: E501 -- compact literal mirrors of the browser feedback contract.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.feedback_packages import FeedbackPackageError, import_feedback_package

_NOW = "2026-08-31T12:34:56.789Z"
_LABELS = ("hello", "no", "please", "thank_you", "yes", "other")


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


# fmt: off
def _frame() -> dict[str, Any]:
    point = {"x": 0.1, "y": 0.2, "z": -0.1, "visibility": 0.9, "presence": 0.8}
    hands = [
        {"slotId": "hand_0", "present": True, "detectorIndex": 0, "trackingId": "hand_0", "handedness": "left", "handednessConfidence": 0.9, "imageLandmarks": [point], "worldLandmarks": None},
        {"slotId": "hand_1", "present": False, "detectorIndex": None, "trackingId": None, "handedness": None, "handednessConfidence": None, "imageLandmarks": None, "worldLandmarks": None},
    ]
    anchors = [{"name": name, "present": False, "imagePoint": None, "worldPoint": None} for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")]
    return {"relativeTimestampUs": 0, "valid": True, "hands": hands, "bodyAnchors": anchors}


def _record(identity: str, *, landmarks: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format": "signlab-feedback-record/1", "id": identity, "savedAt": _NOW,
        "bundle": {"id": "candidate-five", "version": "1.0.0"},
        "prediction": {"decision": {"kind": "target", "label": "hello", "confidence": 0.9}, "reason": "accepted_target"},
        "scores": [{"label": label, "confidence": 0.5} for label in _LABELS],
        "timings": {"preprocessingMs": 1, "inferenceMs": 2, "decisionMs": 0, "totalMs": 3},
        "event": {"firstFrameIndex": 0, "lastFrameIndex": 1, "firstTimestampUs": 0, "lastTimestampUs": 10, "terminationReason": "settled", "configSha256": "sha256:fixture", "durationUs": 10},
        "conditions": {"sourceMirrorState": "not_mirrored", "previewMirrored": True},
        "correction": "hello",
        "consent": {"mode": "per_event", "scope": "local_feedback_only", "grantedAt": _NOW, "landmarksIncluded": landmarks},
    }
    if landmarks:
        record["landmarks"] = [_frame()]
    return record


def _package(records: list[dict[str, Any]], *, landmarks_included: bool | None = None) -> bytes:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    pairs = sorted({(record["bundle"]["id"], record["bundle"]["version"]) for record in records})
    landmark_count = sum("landmarks" in record for record in records)
    consent = {"statementVersion": "signlab-feedback-export-consent/1", "granted": True, "grantedAt": _NOW, "scope": "manual_research_review", "trainingUse": "requires_review_and_approval"}
    manifest = {"recordFormat": "signlab-feedback-record/1", "recordCount": len(records), "bundleVersions": [{"id": identity, "version": version} for identity, version in pairs], "fields": sorted({field for record in records for field in record}), "localConsentScope": "local_feedback_only", "landmarkRecordCount": landmark_count, "landmarksIncluded": landmark_count > 0 if landmarks_included is None else landmarks_included, "exportConsent": consent}
    package = {"format": "signlab-feedback-package/1", "manifest": manifest, "payloadJson": payload, "payloadSha256": _digest(payload.encode())}
    return (json.dumps(package, ensure_ascii=False, indent=2) + "\n").encode()
# fmt: on


def _encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


# fmt: off
def test_valid_package_is_stored_exactly_with_sanitized_nontrainable_receipt(
    tmp_path: Path,
) -> None:
    raw = (Path(__file__).parent / "fixtures/public/feedback/browser-feedback-package-v1.json").read_bytes()
    source, root = tmp_path / "private-name.json", tmp_path / "quarantine"
    source.write_bytes(raw)

    result = import_feedback_package(source, quarantine_root=root)
    receipt = json.loads((result.destination / "receipt.json").read_bytes())

    assert result.package_sha256 == _digest(raw)
    assert result.record_count == 2
    assert result.destination == root / _digest(raw).replace(":", "-")
    assert (result.destination / "package.signlab-feedback.json").read_bytes() == raw
    assert set(receipt) == {"format", "packageSha256", "payloadSha256", "recordFormat", "recordCount", "bundleVersions", "landmarkRecordCount", "landmarksIncluded", "exportConsent", "trainable", "remainingGates"}
    assert receipt["format"] == "signlab-feedback-quarantine-receipt/1"
    assert receipt["recordCount"] == 2
    assert receipt["landmarkRecordCount"] == 1
    assert receipt["trainable"] is False
    assert receipt["remainingGates"] == ["human_review", "cross_package_deduplication", "split_decision", "new_dvc_version"]
    assert receipt["exportConsent"] == {"statementVersion": "signlab-feedback-export-consent/1", "scope": "manual_research_review", "grantedAt": "2026-08-31T13:00:00.000Z"}
    evidence = json.dumps(receipt)
    assert "record-a" not in evidence
    assert str(tmp_path) not in evidence
    assert list(root.iterdir()) == [result.destination]


@pytest.mark.parametrize(
    "case",
    ["package_key", "manifest_key", "package_version", "record_version", "privacy_record", "privacy_landmark", "local_scope", "record_landmark", "record_consent_time", "scores", "event_order", "event_duration", "duplicate_id", "unsorted", "consent_key", "consent_false", "consent_missing", "consent_version", "consent_scope", "consent_time", "training_use", "checksum", "record_count", "bundles", "fields", "landmark_count", "landmark_excluded", "duplicate_key", "surrogate"],
)
def test_invalid_packages_are_rejected_before_quarantine(case: str, tmp_path: Path) -> None:
    records = [_record("record-a"), _record("record-b")]
    if case == "record_version":
        records[0]["format"] = "signlab-feedback-record/2"
    if case == "privacy_record":
        records[0]["rawVideo"] = "private.mp4"
    if case == "privacy_landmark":
        records[0] = _record("record-a", landmarks=True)
        records[0]["landmarks"][0]["hands"][0]["imageLandmarks"][0]["deviceFingerprint"] = "private"
    if case == "local_scope":
        records[0]["consent"]["scope"] = "training"
    if case == "record_landmark":
        records[0]["landmarks"] = []
    if case == "record_consent_time":
        records[0]["consent"]["grantedAt"] = "2026-08-31T12:34:57.789Z"
    if case == "scores":
        records[0]["scores"] = [records[0]["scores"][0]] * 6
    if case == "event_order":
        records[0]["event"]["lastFrameIndex"] = -1
    if case == "event_duration":
        records[0]["event"]["durationUs"] = 11
    if case == "duplicate_id":
        records[1]["id"] = "record-a"
    if case == "unsorted":
        records.reverse()
    raw = _package(records, landmarks_included=False if case == "landmark_excluded" else None)
    if case == "landmark_excluded":
        raw = _package([_record("record-a", landmarks=True)], landmarks_included=False)
    package = json.loads(raw)
    manifest = package["manifest"]
    consent = manifest["exportConsent"]
    if case == "package_key":
        package["sourcePath"] = "private"
    if case == "manifest_key":
        manifest["recordIds"] = ["private"]
    if case == "package_version":
        package["format"] = "signlab-feedback-package/2"
    if case == "consent_key":
        consent["reviewerName"] = "private"
    if case == "consent_false":
        consent["granted"] = False
    if case == "consent_missing":
        del consent["granted"]
    if case == "consent_version":
        consent["statementVersion"] = "signlab-feedback-export-consent/2"
    if case == "consent_scope":
        consent["scope"] = "training"
    if case == "consent_time":
        consent["grantedAt"] = "2026-8-31T12:34:56.7890Z"
    if case == "training_use":
        consent["trainingUse"] = "approved"
    if case == "checksum":
        package["payloadSha256"] = "sha256:" + "0" * 64
    if case == "record_count":
        manifest["recordCount"] += 1
    if case == "bundles":
        manifest["bundleVersions"] = []
    if case == "fields":
        manifest["fields"] = []
    if case == "landmark_count":
        manifest["landmarkRecordCount"] += 1
    if case == "duplicate_key":
        raw = b'{"format":"one","format":"two"}'
    elif case == "surrogate":
        raw = b'{"format":"signlab-feedback-package/1","manifest":{},"payloadJson":"\\ud800","payloadSha256":"x"}'
    else:
        raw = _encode(package)
    source = tmp_path / "private-package.json"
    source.write_bytes(raw)

    with pytest.raises(FeedbackPackageError, match="feedback package invalid"):
        import_feedback_package(source, quarantine_root=tmp_path / "quarantine")
    assert not (tmp_path / "quarantine").exists()


def test_duplicate_package_is_never_replaced(tmp_path: Path) -> None:
    source = tmp_path / "package.json"
    source.write_bytes(_package([_record("record-a")]))
    first = import_feedback_package(source, quarantine_root=tmp_path / "quarantine")

    with pytest.raises(FeedbackPackageError, match="duplicate"):
        import_feedback_package(source, quarantine_root=tmp_path / "quarantine")
    assert (first.destination / "package.signlab-feedback.json").read_bytes() == source.read_bytes()


def test_cli_reports_only_sanitized_aggregate_success_and_failure(tmp_path: Path) -> None:
    source, root = tmp_path / "participant-private.json", tmp_path / "quarantine"
    source.write_bytes((Path(__file__).parent / "fixtures/public/feedback/browser-feedback-package-v1.json").read_bytes())
    runner = CliRunner(env={"NO_COLOR": "1"})

    success = runner.invoke(cli.app, ["data", "import-feedback-package", str(source), "--quarantine-root", str(root)])
    assert success.exit_code == 0
    assert success.output.splitlines() == ["Feedback package quarantined: 2 records.", f"Package SHA-256: {_digest(source.read_bytes())}", "Trainable: no; manual review gates remain."]
    missing = tmp_path / "secret-missing.json"
    failure = runner.invoke(cli.app, ["data", "import-feedback-package", str(missing), "--quarantine-root", str(root)])
    assert failure.exit_code == 1
    assert failure.output.strip() == "Feedback package import failed."
    assert source.name not in success.output
    assert missing.name not in failure.output
    assert str(tmp_path) not in success.output + failure.output
# fmt: on
