from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from feature_fixtures import make_feature_fixture, make_hand_row
from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.external_dataset import (
    ExternalMediaRecordV1,
    ExternalSplit,
    ExternalTargetLabel,
)
from signlab.contracts.extraction import LandmarkFramesTableV1
from signlab.contracts.features import PortableFeatureSequenceV1, validate_portable_feature_sequence
from signlab.datasets import public_corpus
from signlab.extraction.batch import ExtractionBatchError
from signlab.extraction.runtime import VerifiedModelAssets


def _media(
    index: int,
    payload: bytes,
    *,
    target: ExternalTargetLabel = "hello",
    source_split: ExternalSplit = "train",
    signer_index: int | None = None,
) -> ExternalMediaRecordV1:
    identity = f"{index:032x}"
    signer_identity = "a" * 32 if signer_index is None else f"{signer_index:032x}"
    media_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    digest = media_sha256.removeprefix("sha256:")
    return ExternalMediaRecordV1(
        schema_version="external-media-record/1",
        sample_id=f"sample_{identity}",
        recording_id=f"recording_{identity}",
        participant_id=f"participant_{signer_identity}",
        archive_id=f"popsign_game_{source_split}_{target}",
        source_member_fingerprint=(
            f"sha256:{hashlib.sha256(f'fingerprint:{index}'.encode()).hexdigest()}"
        ),
        category="game",
        source_split=source_split,
        source_label=target.replace("_", "-"),
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


def _patch_verified_inputs(
    monkeypatch: pytest.MonkeyPatch,
    manifest: SimpleNamespace,
) -> None:
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

    _patch_verified_inputs(monkeypatch, manifest)
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


def _trainable_media(
    candidate_count_per_group: int | None = None,
    candidate_count_by_split: dict[ExternalSplit, int] | None = None,
) -> tuple[tuple[ExternalMediaRecordV1, bytes], ...]:
    quotas: dict[ExternalSplit, int] = {"train": 10, "val": 3, "test": 3}
    targets: tuple[ExternalTargetLabel, ...] = (
        "hello",
        "no",
        "please",
        "thank_you",
        "yes",
    )
    signer_offsets: dict[ExternalSplit, int] = {"train": 0, "val": 100, "test": 200}
    result: list[tuple[ExternalMediaRecordV1, bytes]] = []
    index = 1
    for source_split, quota in quotas.items():
        for target in targets:
            group_candidate_count = (
                candidate_count_by_split[source_split]
                if candidate_count_by_split is not None
                else candidate_count_per_group or quota
            )
            for position in range(group_candidate_count):
                payload = f"{source_split}:{target}:{position}".encode()
                result.append(
                    (
                        _media(
                            index,
                            payload,
                            target=target,
                            source_split=source_split,
                            signer_index=signer_offsets[source_split] + position + 1,
                        ),
                        payload,
                    )
                )
                index += 1
    return tuple(result)


def _write_trainable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_and_payloads: tuple[tuple[ExternalMediaRecordV1, bytes], ...],
) -> tuple[Path, Path]:
    manifest = SimpleNamespace(
        content_sha256=f"sha256:{'d' * 64}",
        media=tuple(media for media, _payload in media_and_payloads),
        archives=(),
    )
    _patch_verified_inputs(monkeypatch, manifest)
    manifest_path = tmp_path / "external-dataset-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    external_root = tmp_path / "external"
    external_root.mkdir()
    for media, payload in media_and_payloads:
        media_path = external_root.joinpath(*media.locator.path.split("/"))
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(payload)
    return manifest_path, external_root


def _mean_feature_vector(sequence: PortableFeatureSequenceV1) -> tuple[float, ...]:
    totals = [0] * len(sequence.feature_names)
    counts = [0] * len(sequence.feature_names)
    for values, valid, padded in zip(
        sequence.values_q,
        sequence.valid_mask,
        sequence.padding_mask,
        strict=True,
    ):
        if not padded:
            for index, (value, is_valid) in enumerate(zip(values, valid, strict=True)):
                if is_valid:
                    totals[index] += value
                    counts[index] += 1
    return tuple(
        total / count if count else 0.0 for total, count in zip(totals, counts, strict=True)
    )


def _consume_trainable_manifest(output: Path) -> tuple[int, int]:
    corpus = json.loads((output / "public-corpus.json").read_bytes())
    unhashed = dict(corpus)
    claimed_sha256 = unhashed.pop("corpus_sha256")
    assert canonical_sha256(unhashed, domain="signlab-public-corpus/1") == claimed_sha256
    rows = corpus["selected"]
    assert len(rows) == 80
    group_counts: Counter[tuple[str, str]] = Counter()
    group_signers: dict[tuple[str, str], set[str]] = defaultdict(set)
    signer_splits: dict[str, str] = {}
    recording_splits: dict[str, str] = {}
    vectors: dict[tuple[str, str], list[tuple[float, ...]]] = defaultdict(list)
    resolved_output = output.resolve(strict=True)
    for row in rows:
        group = (row["source_split"], row["target_label_id"])
        assert row["source_signer_id"] not in group_signers[group]
        group_signers[group].add(row["source_signer_id"])
        group_counts[group] += 1
        prior_split = signer_splits.setdefault(row["source_signer_id"], row["source_split"])
        assert prior_split == row["source_split"]
        recording_split = recording_splits.setdefault(
            row["source_recording_id"], row["source_split"]
        )
        assert recording_split == row["source_split"]
        feature_path = (resolved_output / row["feature_path"]).resolve(strict=True)
        assert feature_path.is_relative_to(resolved_output)
        feature = validate_portable_feature_sequence(feature_path.read_bytes())
        assert feature.sequence_sha256 == row["feature_sequence_sha256"]
        assert feature.source_recording_id == row["source_recording_id"]
        assert feature.feature_plan_sha256 == corpus["feature_plan_sha256"]
        vectors[group].append(_mean_feature_vector(feature))

    targets = ("hello", "no", "please", "thank_you", "yes")
    for source_split, quota in (("train", 10), ("val", 3), ("test", 3)):
        assert all(group_counts[(source_split, target)] == quota for target in targets)
    centroids = {
        target: tuple(sum(values) / 10 for values in zip(*vectors[("train", target)], strict=True))
        for target in targets
    }
    prediction_count = 0
    for target in targets:
        for vector in vectors[("val", target)]:
            min(
                targets,
                key=lambda label: sum(
                    (value - center) ** 2
                    for value, center in zip(vector, centroids[label], strict=True)
                ),
            )
            prediction_count += 1
    return sum(len(vectors[("train", target)]) for target in targets), prediction_count


def test_trainable_public_corpus_is_deterministic_split_ready_and_smoke_trainable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_and_payloads = _trainable_media()
    manifest_path, external_root = _write_trainable_inputs(
        tmp_path,
        monkeypatch,
        media_and_payloads,
    )
    fixture = make_feature_fixture((0, 33_333, 66_667))
    monkeypatch.setattr(
        public_corpus,
        "extract_media_landmarks",
        lambda recording_id, *_args, **_kwargs: _with_recording_id(
            fixture.table,
            recording_id,
        ),
    )
    output = tmp_path / "trainable-corpus"
    result = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=output,
        trainable_smoke=True,
    )
    rerun_output = tmp_path / "trainable-corpus-rerun"
    rerun = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=rerun_output,
        trainable_smoke=True,
    )

    assert result.decision == "ready"
    assert result.selected_count == result.target_count == 80
    assert result.attempted_count == 80
    assert result.attempt_limit == 750
    assert result.exclusion_count == 0
    assert result.corpus_sha256 == rerun.corpus_sha256
    assert (output / "public-corpus.json").read_bytes() == (
        rerun_output / "public-corpus.json"
    ).read_bytes()
    assert result.summary["selected_by_split"] == {"test": 15, "train": 50, "val": 15}
    assert result.summary["selected_by_target"] == {
        "hello": 16,
        "no": 16,
        "please": 16,
        "thank_you": 16,
        "yes": 16,
    }
    assert result.summary["selected_signer_count"] == 16
    assert result.summary["unfilled_group_count"] == 0
    assert _consume_trainable_manifest(output) == (50, 15)


def test_trainable_public_corpus_stops_at_the_global_attempt_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = list(_trainable_media(candidate_count_per_group=2))
    first_media = original[0][0]
    duplicate_media = original[1][0].model_copy(
        update={"participant_id": first_media.participant_id}
    )
    original[1] = (duplicate_media, original[1][1])
    media_and_payloads = tuple(original)
    manifest_path, external_root = _write_trainable_inputs(
        tmp_path,
        monkeypatch,
        media_and_payloads,
    )
    monkeypatch.setattr(public_corpus, "_TRAINABLE_ATTEMPT_LIMIT", 16)
    fixture = make_feature_fixture((0, 33_333, 66_667))

    def extract(recording_id: str, *_args: object, **_kwargs: object) -> LandmarkFramesTableV1:
        if recording_id == first_media.recording_id:
            return _with_recording_id(fixture.table, recording_id)
        raise ExtractionBatchError("execution.failed")

    monkeypatch.setattr(
        public_corpus,
        "extract_media_landmarks",
        extract,
    )

    result = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=tmp_path / "insufficient-corpus",
        trainable_smoke=True,
    )

    assert result.decision == "insufficient"
    assert result.selected_count == 1
    assert result.attempted_count == result.attempt_limit == 16
    assert result.exclusion_count == len(media_and_payloads) - 1
    assert result.summary["unfilled_group_count"] == 15
    exclusions = result.summary["exclusions"]
    assert isinstance(exclusions, dict)
    assert exclusions["selection.signer_already_selected"] == 1
    group_results = result.summary["selection_groups"]
    assert isinstance(group_results, list)
    assert min(item["attempted_count"] for item in group_results) == 1
    assert max(item["attempted_count"] for item in group_results) == 2
    assert {item["terminal_reason"] for item in group_results} == {
        "attempt_limit_reached",
        "candidates_exhausted",
    }


def test_trainable_public_corpus_uses_literal_cap_and_fair_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_and_payloads = _trainable_media(
        candidate_count_by_split={"train": 95, "val": 29, "test": 29}
    )
    manifest_path, external_root = _write_trainable_inputs(
        tmp_path,
        monkeypatch,
        media_and_payloads,
    )
    extraction_calls = 0

    def fail_extraction(*_args: object, **_kwargs: object) -> LandmarkFramesTableV1:
        nonlocal extraction_calls
        extraction_calls += 1
        raise ExtractionBatchError("execution.failed")

    monkeypatch.setattr(public_corpus, "extract_media_landmarks", fail_extraction)
    result = public_corpus.build_public_corpus(
        manifest_path,
        external_root=external_root,
        model_root=tmp_path,
        output_root=tmp_path / "literal-cap-corpus",
        trainable_smoke=True,
    )

    assert result.decision == "insufficient"
    assert extraction_calls == result.attempted_count == result.attempt_limit == 750
    assert result.exclusion_count == len(media_and_payloads) == 765
    assert result.summary["exclusions"] == {
        "extraction.failed": 750,
        "selection.total_attempt_limit": 15,
    }
    group_results = result.summary["selection_groups"]
    assert isinstance(group_results, list)
    expected_by_split = {"train": 94, "val": 28, "test": 28}
    assert all(
        item["attempted_count"] == expected_by_split[str(item["source_split"])]
        for item in group_results
    )
    assert {item["terminal_reason"] for item in group_results} == {"attempt_limit_reached"}


def test_trainable_signer_interleaving_retries_only_after_every_signer() -> None:
    candidates = (
        _media(1, b"one", signer_index=1),
        _media(2, b"two", signer_index=1),
        _media(3, b"three", signer_index=2),
    )

    ordered = public_corpus._signer_interleaved(candidates)

    assert [item.participant_id for item in ordered] == [
        candidates[0].participant_id,
        candidates[2].participant_id,
        candidates[1].participant_id,
    ]


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
