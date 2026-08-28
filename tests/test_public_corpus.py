from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from feature_fixtures import make_feature_fixture, make_hand_row
from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.external_dataset import ExternalMediaRecordV1, ExternalTargetLabel
from signlab.contracts.extraction import LandmarkFramesTableV1
from signlab.datasets import public_corpus
from signlab.extraction.batch import ExtractionBatchError
from signlab.extraction.runtime import VerifiedModelAssets


def _media(
    index: int,
    payload: bytes,
    *,
    target: ExternalTargetLabel = "hello",
) -> ExternalMediaRecordV1:
    digit = str(index)
    fingerprint_digit = "0123456789abcdef"[index + 1]
    media_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    digest = media_sha256.removeprefix("sha256:")
    return ExternalMediaRecordV1(
        schema_version="external-media-record/1",
        sample_id=f"sample_{digit * 32}",
        recording_id=f"recording_{digit * 32}",
        participant_id=f"participant_{'a' * 32}",
        archive_id="popsign_game_train_hello",
        source_member_fingerprint=f"sha256:{fingerprint_digit * 64}",
        category="game",
        source_split="train",
        source_label=target,
        target_label_id=target,
        media_type="video/mp4",
        sha256=media_sha256,
        size_bytes=len(payload),
        locator=WorkspaceRelativeLocatorV1(
            kind="workspace_relative",
            path=f"media/sha256/{digest[:2]}/{digest}.mp4",
        ),
        eligible_for_extraction=True,
    )


def _with_recording_id(table: LandmarkFramesTableV1, recording_id: str) -> LandmarkFramesTableV1:
    payload = table.model_dump(mode="json", round_trip=True)
    for row in payload["rows"]:
        row["source_recording_id"] = recording_id
    return LandmarkFramesTableV1.model_validate_json(
        canonical_json_bytes(payload),
        strict=True,
    )


def test_public_corpus_tries_next_candidate_and_repeats_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_payloads = tuple(f"video-{index}".encode("ascii") for index in range(1, 10))
    candidates = tuple(
        _media(index, payload, target="hello" if index <= 3 else "yes")
        for index, payload in enumerate(media_payloads, start=1)
    )
    manifest = SimpleNamespace(
        content_sha256=f"sha256:{'d' * 64}",
        media=candidates,
        archives=(
            SimpleNamespace(
                archive_id="popsign_game_train_hello",
                split="train",
                source_label="hello",
                sha256=f"sha256:{'e' * 64}",
                size_bytes=2048,
                member_count=2,
            ),
        ),
    )
    timestamps = (0, 33_333, 66_667)
    rejected_fixture = make_feature_fixture(
        timestamps,
        hand_rows=tuple(
            make_hand_row(timestamp_us=timestamp, first_present=False) for timestamp in timestamps
        ),
    )
    accepted_fixture = make_feature_fixture(timestamps)
    tables = {
        candidates[0].recording_id: _with_recording_id(
            rejected_fixture.table, candidates[0].recording_id
        ),
        candidates[1].recording_id: _with_recording_id(
            accepted_fixture.table, candidates[1].recording_id
        ),
    }

    monkeypatch.setattr(
        public_corpus,
        "validate_external_dataset_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(archive_byte_integrity="verified"),
    )
    monkeypatch.setattr(
        public_corpus,
        "validate_external_dataset_manifest",
        lambda _document: manifest,
    )
    monkeypatch.setattr(
        public_corpus,
        "load_popsign_source",
        lambda: SimpleNamespace(
            license=SimpleNamespace(
                attribution_text="PopSign fixture attribution.",
                license_id="CC-BY-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
        ),
    )
    monkeypatch.setattr(
        public_corpus,
        "verify_model_assets",
        lambda _root: VerifiedModelAssets(
            hand_model_bytes=b"hand",
            pose_model_bytes=b"pose",
            hand_model_sha256=f"sha256:{'a' * 64}",
            pose_model_sha256=f"sha256:{'b' * 64}",
        ),
    )
    monkeypatch.setattr(
        public_corpus,
        "extract_media_landmarks",
        lambda recording_id, *_args, **_kwargs: (
            tables[recording_id]
            if recording_id in tables
            else (_ for _ in ()).throw(ExtractionBatchError("execution.failed"))
        ),
    )

    manifest_path = tmp_path / "external-dataset-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    external_root = tmp_path / "external"
    external_root.mkdir()
    for media, payload in zip(candidates, media_payloads, strict=True):
        media_path = external_root.joinpath(*media.locator.path.split("/"))
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(payload)
    output = tmp_path / "corpus"
    progress: list[tuple[int, int, bool]] = []

    first = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=output,
        progress=lambda index, total, accepted: progress.append((index, total, accepted)),
    )
    first_corpus = (output / "public-corpus.json").read_bytes()
    rerun_output = tmp_path / "corpus-rerun"
    second = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=rerun_output,
    )

    assert first.corpus_sha256 == second.corpus_sha256
    assert first_corpus == (rerun_output / "public-corpus.json").read_bytes()
    assert first.selected_count == 1
    assert first.group_count == 2
    assert first.summary["exclusions"] == {
        "extraction.failed": 5,
        "quality.reject": 1,
        "selection.attempt_limit": 1,
        "selection.not_needed_after_accepted": 1,
    }
    assert progress == [(1, 2, True), (2, 2, False)]
    corpus = json.loads(first_corpus)
    assert corpus["selected"][0]["sample_id"] == candidates[1].sample_id
    assert corpus["selected"][0]["feature_sequence_sha256"].startswith("sha256:")
    assert (output / "public-corpus-summary.md").is_file()


def test_public_corpus_rejects_changed_media_and_nonempty_output(tmp_path: Path) -> None:
    media = _media(1, b"expected")
    changed = tmp_path / "changed.mp4"
    changed.write_bytes(b"different")
    with pytest.raises(public_corpus.PublicCorpusError):
        public_corpus._verify_media_bytes(changed, media)

    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "partial.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(public_corpus.PublicCorpusError):
        public_corpus._fresh_output_root(output)


def test_public_corpus_report_write_removes_temporary_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(
        "signlab.datasets.public_corpus.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic publication failure")),
    )

    with pytest.raises(public_corpus.PublicCorpusError):
        public_corpus._write_bytes(destination, b"report")

    assert not destination.exists()
    assert not tuple(tmp_path.iterdir())


def test_single_frame_duration_is_positive() -> None:
    row = make_feature_fixture().table.rows[0]

    assert public_corpus._duration_us((row,)) == 1
