from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import cast

import pytest

from signlab.contracts.external_dataset import (
    ExternalAcquisitionPlanV1,
    ExternalDatasetContractError,
    ExternalDatasetManifestV1,
    external_acquisition_plan_digest,
    external_dataset_manifest_digest,
    external_dataset_selection_digest,
    licensed_dataset_source_digest,
    validate_external_acquisition_plan,
    validate_external_dataset_manifest,
    validate_external_dataset_selection,
    validate_licensed_dataset_source,
)
from signlab.datasets.external_resources import (
    build_popsign_source,
    build_signlab_five_popsign_selection,
    external_resource_reference,
)

ZERO_DIGEST = "sha256:" + "0" * 64
MEDIA_DIGEST = "sha256:" + "a" * 64


def _plan_payload() -> dict[str, object]:
    source = build_popsign_source()
    selection = build_signlab_five_popsign_selection()
    archives: list[dict[str, object]] = []
    for split in selection.splits:
        for mapping in selection.mappings:
            source_label = mapping.source_label
            archives.append(
                {
                    "schema_version": "external-archive-plan/1",
                    "archive_id": f"popsign_v1_game_{split}_{source_label}",
                    "category": "game",
                    "split": split,
                    "source_label": source_label,
                    "download_url": source.download_url_template.format(
                        download_id=source.download_id,
                        category="game",
                        split=split,
                        source_label=source_label,
                    ),
                    "local_archive": {
                        "kind": "workspace_relative",
                        "path": f"archives/game/{split}/{source_label}.tar",
                    },
                    "archive_format": "tar",
                    "publisher_sha256": None,
                    "integrity_basis": "trust_on_first_use_then_sha256",
                }
            )
    payload: dict[str, object] = {
        "schema_version": "external-acquisition-plan/1",
        "plan_id": "popsign_v1_signlab_five",
        "version": "1.0.0",
        "source": external_resource_reference(source).model_dump(mode="json", round_trip=True),
        "selection": external_resource_reference(selection).model_dump(
            mode="json", round_trip=True
        ),
        "network_access": "forbidden",
        "preview_media": "forbidden",
        "required_license_acknowledgement": "CC-BY-4.0",
        "archives": archives,
        "plan_sha256": ZERO_DIGEST,
    }
    payload["plan_sha256"] = external_acquisition_plan_digest(payload)
    return payload


def _manifest_payload() -> dict[str, object]:
    source = build_popsign_source()
    selection = build_signlab_five_popsign_selection()
    plan = validate_external_acquisition_plan(_plan_payload())
    payload: dict[str, object] = {
        "schema_version": "external-dataset-manifest/1",
        "dataset_id": "popsign_v1_signlab_five",
        "version": "1.0.0",
        "source": external_resource_reference(source).model_dump(mode="json", round_trip=True),
        "selection": external_resource_reference(selection).model_dump(
            mode="json", round_trip=True
        ),
        "acquisition_plan_sha256": plan.plan_sha256,
        "taxonomy": selection.taxonomy.model_dump(mode="json", round_trip=True),
        "license_acknowledgement": {
            "schema_version": "external-license-acknowledgement/1",
            "license_id": "CC-BY-4.0",
            "accepted": True,
            "authorization_basis": "licensed_public_dataset",
            "signlab_participant_consent": "not_applicable",
        },
        "contains_identifiable_human_video": True,
        "source_metadata_retained": False,
        "claim_scope": "isolated_predefined_gesture_research_only",
        "archives": [
            {
                "schema_version": "external-archive-record/1",
                "archive_id": "popsign_v1_game_train_hello",
                "category": "game",
                "split": "train",
                "source_label": "hello",
                "local_archive": {
                    "kind": "workspace_relative",
                    "path": "archives/game/train/hello.tar",
                },
                "sha256": "sha256:" + "b" * 64,
                "size_bytes": 1024,
                "member_count": 1,
                "uncompressed_size_bytes": 5,
                "publisher_checksum_available": False,
                "integrity_basis": "local_sha256_after_download",
            }
        ],
        "media": [
            {
                "schema_version": "external-media-record/1",
                "sample_id": "sample_" + "1" * 32,
                "recording_id": "recording_" + "2" * 32,
                "participant_id": "participant_" + "3" * 32,
                "archive_id": "popsign_v1_game_train_hello",
                "source_member_fingerprint": "sha256:" + "4" * 64,
                "category": "game",
                "source_split": "train",
                "source_label": "hello",
                "target_label_id": "hello",
                "media_type": "video/mp4",
                "sha256": MEDIA_DIGEST,
                "size_bytes": 5,
                "locator": {
                    "kind": "workspace_relative",
                    "path": f"media/sha256/aa/{'a' * 64}.mp4",
                },
                "eligible_for_extraction": True,
            }
        ],
        "content_sha256": ZERO_DIGEST,
    }
    payload["content_sha256"] = external_dataset_manifest_digest(payload)
    return payload


def _redigest_manifest(payload: dict[str, object]) -> dict[str, object]:
    payload["content_sha256"] = external_dataset_manifest_digest(payload)
    return payload


def test_source_and_selection_round_trip_with_stable_semantic_identities() -> None:
    source = build_popsign_source()
    selection = build_signlab_five_popsign_selection()

    assert validate_licensed_dataset_source(source.model_dump_json()) == source
    assert validate_external_dataset_selection(selection.model_dump_json()) == selection
    assert licensed_dataset_source_digest(source) == licensed_dataset_source_digest(source)
    assert external_dataset_selection_digest(selection) == external_dataset_selection_digest(
        selection
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(total_videos="200686"), "invalid licensed dataset source"),
        (
            lambda value: value.update(dataset_url="https://user:secret@example.test/data"),
            "invalid licensed dataset source",
        ),
        (
            lambda value: value.update(categories=["non-game", "game"]),
            "invalid licensed dataset source",
        ),
        (
            lambda value: value.update(unexpected=True),
            "invalid licensed dataset source",
        ),
    ],
)
def test_source_contract_rejects_coercion_credentials_order_and_extra_fields(
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    payload = build_popsign_source().model_dump(mode="json", round_trip=True)
    mutation(payload)

    with pytest.raises(ExternalDatasetContractError, match=message):
        validate_licensed_dataset_source(payload)


def test_acquisition_plan_is_exactly_canonical_and_digest_bound() -> None:
    payload = _plan_payload()
    checked = validate_external_acquisition_plan(json.dumps(payload))

    assert isinstance(checked, ExternalAcquisitionPlanV1)
    assert len(checked.archives) == 15
    assert checked.plan_sha256 == external_acquisition_plan_digest(checked)
    assert tuple((item.split, item.source_label) for item in checked.archives) == tuple(
        (split, source_label)
        for split in ("train", "val", "test")
        for source_label in ("hello", "no", "please", "thankyou", "yes")
    )
    assert all(item.publisher_sha256 is None for item in checked.archives)
    assert all("preview" not in item.download_url for item in checked.archives)


@pytest.mark.parametrize("defect", ["digest", "order", "url", "locator", "extra"])
def test_acquisition_plan_fails_closed_on_structural_or_identity_defects(defect: str) -> None:
    payload = _plan_payload()
    archives = cast(list[dict[str, object]], payload["archives"])
    if defect == "digest":
        payload["plan_sha256"] = ZERO_DIGEST
    elif defect == "order":
        archives[0], archives[1] = archives[1], archives[0]
        payload["plan_sha256"] = external_acquisition_plan_digest(payload)
    elif defect == "url":
        archives[0]["download_url"] = "https://example.test/wrong.tar"
        payload["plan_sha256"] = external_acquisition_plan_digest(payload)
    elif defect == "locator":
        archives[0]["local_archive"] = {
            "kind": "workspace_relative",
            "path": "archives/wrong.tar",
        }
        payload["plan_sha256"] = external_acquisition_plan_digest(payload)
    else:
        payload["unexpected"] = True
        payload["plan_sha256"] = external_acquisition_plan_digest(payload)

    with pytest.raises(ExternalDatasetContractError, match="invalid external acquisition plan"):
        validate_external_acquisition_plan(payload)


def test_manifest_has_stable_identity_and_retains_no_upstream_filename_or_timestamp() -> None:
    payload = _manifest_payload()
    checked = validate_external_dataset_manifest(json.dumps(payload))

    assert isinstance(checked, ExternalDatasetManifestV1)
    assert checked.content_sha256 == external_dataset_manifest_digest(checked)
    serialized = checked.model_dump_json()
    assert "provider-signer" not in serialized
    assert "recording_start_time" not in serialized
    assert "--" not in serialized
    assert checked.license_acknowledgement.authorization_basis == "licensed_public_dataset"
    assert checked.license_acknowledgement.signlab_participant_consent == "not_applicable"


@pytest.mark.parametrize("defect", ["digest", "locator", "archive", "count", "filename"])
def test_manifest_rejects_byte_lineage_and_privacy_defects(defect: str) -> None:
    payload = _manifest_payload()
    media = cast(list[dict[str, object]], payload["media"])
    archives = cast(list[dict[str, object]], payload["archives"])
    if defect == "digest":
        payload["content_sha256"] = ZERO_DIGEST
    elif defect == "locator":
        media[0]["locator"] = {
            "kind": "workspace_relative",
            "path": "media/not-content-addressed.mp4",
        }
        _redigest_manifest(payload)
    elif defect == "archive":
        media[0]["archive_id"] = "unknown_archive"
        _redigest_manifest(payload)
    elif defect == "count":
        archives[0]["member_count"] = 2
        _redigest_manifest(payload)
    else:
        media[0]["upstream_filename"] = "participant--timestamp-.mp4"
        _redigest_manifest(payload)

    with pytest.raises(ExternalDatasetContractError, match="invalid external dataset manifest"):
        validate_external_dataset_manifest(payload)


def test_manifest_rejects_external_signer_leakage_across_source_splits() -> None:
    payload = _manifest_payload()
    archives = cast(list[dict[str, object]], payload["archives"])
    media = cast(list[dict[str, object]], payload["media"])
    second_archive = copy.deepcopy(archives[0])
    second_archive.update(
        archive_id="popsign_v1_game_val_hello",
        split="val",
        local_archive={
            "kind": "workspace_relative",
            "path": "archives/game/val/hello.tar",
        },
        sha256="sha256:" + "c" * 64,
    )
    archives.append(second_archive)
    second_media = copy.deepcopy(media[0])
    second_media.update(
        sample_id="sample_" + "5" * 32,
        recording_id="recording_" + "6" * 32,
        archive_id="popsign_v1_game_val_hello",
        source_member_fingerprint="sha256:" + "7" * 64,
        source_split="val",
        sha256="sha256:" + "d" * 64,
        locator={
            "kind": "workspace_relative",
            "path": f"media/sha256/dd/{'d' * 64}.mp4",
        },
    )
    media.append(second_media)
    media.sort(key=lambda item: cast(str, item["sample_id"]))
    _redigest_manifest(payload)

    with pytest.raises(ExternalDatasetContractError, match="invalid external dataset manifest"):
        validate_external_dataset_manifest(payload)


def test_external_validator_rejects_duplicate_json_members_without_echoing_values() -> None:
    document = '{"schema_version":"external-acquisition-plan/1","schema_version":"private"}'

    with pytest.raises(
        ExternalDatasetContractError, match="invalid external acquisition plan"
    ) as error:
        validate_external_acquisition_plan(document)
    assert "private" not in str(error.value)
