"""Dataset and legacy-evidence command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from signlab.commands._group import create_group
from signlab.legacy.exporter import LegacyExportError, export_legacy_evidence
from signlab.legacy.validator import validate_legacy_export

app = create_group(
    help_text=(
        "Capture, import, validate, version, and split consent- or license-authorized datasets."
    )
)


@app.command("import-feedback-package")
def import_feedback_package_command(
    package: Annotated[
        Path,
        typer.Argument(help="Browser-downloaded signlab-feedback-package/1 JSON file."),
    ],
    quarantine_root: Annotated[
        Path,
        typer.Option(
            "--quarantine-root",
            help="Ignored private root for content-addressed feedback quarantine.",
        ),
    ] = Path("data/private/feedback-quarantine"),
) -> None:
    """Validate and quarantine feedback; never promote it for training."""

    from signlab.feedback_packages import import_feedback_package

    try:
        result = import_feedback_package(package, quarantine_root=quarantine_root)
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Feedback package import failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Feedback package quarantined: {result.record_count} records.")
    typer.echo(f"Package SHA-256: {result.package_sha256}")
    typer.echo("Trainable: no; manual review gates remain.")


def _read_capture_source_map(path: Path) -> dict[str, str]:
    """Read the private path map without returning untrusted values in errors."""

    from signlab.contracts.canonical import parse_json_object

    try:
        payload = parse_json_object(path.read_bytes())
        pairs_are_strings = all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
        )
        if not pairs_are_strings:
            raise ValueError("source map must contain only string pairs")
        return cast(dict[str, str], payload)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("capture source map is invalid") from error


@app.command("allocate-capture-ids")
def allocate_capture_ids_command(
    output: Annotated[
        Path,
        typer.Argument(help="New private JSON file that durably stores opaque workflow IDs."),
    ],
    retry_of: Annotated[
        Path | None,
        typer.Option(
            "--retry-of",
            help="Existing identifier file whose workflow IDs a new retry must retain.",
        ),
    ] = None,
) -> None:
    """Allocate opaque IDs once, or allocate a distinct recording/attempt retry."""

    from signlab.datasets.capture import CaptureAllocationError, allocate_capture_identifiers

    try:
        result = allocate_capture_identifiers(output, retry_of=retry_of)
    except (CaptureAllocationError, OSError, TypeError, ValueError) as error:
        typer.echo("Capture identifier allocation failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Capture identifiers: {result.status}.")
    typer.echo(f"Identifier-set SHA-256: {result.identifiers_sha256}")


@app.command("validate-capture")
def validate_capture_command(
    sidecar: Annotated[
        Path,
        typer.Argument(help="Collection-sidecar/1 JSON document to validate."),
    ],
) -> None:
    """Validate capture state, attempt history, consent bindings, and review history."""

    from signlab.contracts.ingest import (
        collection_sidecar_digest,
        validate_collection_sidecar,
    )

    try:
        checked = validate_collection_sidecar(sidecar.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Collection sidecar validation failed.", err=True)
        raise typer.Exit(code=1) from error
    attempts = tuple(
        attempt for occurrence in checked.occurrences for attempt in occurrence.attempts
    )
    typer.echo(f"Collection sidecar SHA-256: {collection_sidecar_digest(checked)}")
    typer.echo(f"Collection state: {checked.state}")
    typer.echo(
        "Capture outcomes: "
        f"{sum(attempt.outcome == 'accepted' for attempt in attempts)} accepted, "
        f"{sum(attempt.outcome == 'retry' for attempt in attempts)} retry, "
        f"{sum(attempt.outcome == 'quarantined' for attempt in attempts)} quarantined, "
        f"{sum(occurrence.state == 'skipped' for occurrence in checked.occurrences)} skipped."
    )
    typer.echo(f"Annotation histories: {len(checked.annotations)}")


@app.command("append-capture-attempt")
def append_capture_attempt_command(
    sidecar: Annotated[
        Path,
        typer.Argument(help="Active or paused collection-sidecar/1 JSON document to update."),
    ],
    identifiers: Annotated[
        Path,
        typer.Option(
            "--identifiers",
            help="Existing capture-identifier-set/1 JSON for this attempt.",
        ),
    ],
    media: Annotated[
        Path,
        typer.Option("--media", help="Private captured file to hash without persisting its path."),
    ],
    outcome: Annotated[
        Literal["accepted", "retry", "quarantined"],
        typer.Option("--outcome", help="Coded outcome for this immutable attempt."),
    ],
    recorded_at: Annotated[
        str,
        typer.Option("--recorded-at", help="UTC capture timestamp in YYYY-MM-DDTHH:MM:SSZ form."),
    ],
    media_type: Annotated[
        Literal["video/mp4", "video/quicktime", "video/webm"],
        typer.Option("--media-type", help="Trusted declared media type."),
    ],
    duration_us: Annotated[
        int,
        typer.Option(
            "--duration-us",
            min=1,
            help="Trusted positive media duration in microseconds.",
        ),
    ],
    handedness: Annotated[
        Literal["left", "right", "unknown"],
        typer.Option("--handedness", help="Observed signing hand for this attempt."),
    ],
    mirror_state: Annotated[
        Literal["not_mirrored", "mirrored"],
        typer.Option("--mirror-state", help="Trusted capture mirror state."),
    ],
    rotation_degrees: Annotated[
        int,
        typer.Option("--rotation-degrees", help="Trusted rotation: 0, 90, 180, or 270."),
    ],
    reason_code: Annotated[
        str | None,
        typer.Option(
            "--reason-code",
            help="Required controlled reason for retry or quarantine; forbidden for accepted.",
        ),
    ] = None,
    consent_grant: Annotated[
        Path | None,
        typer.Option(
            "--consent-grant",
            help="Prevalidated recording-consent-grant/1 JSON required for accepted.",
        ),
    ] = None,
) -> None:
    """Hash media and atomically append one preallocated capture attempt."""

    from signlab.datasets.ledger import CaptureLedgerError, append_capture_attempt

    try:
        result = append_capture_attempt(
            sidecar,
            identifiers.read_bytes(),
            media_path=media,
            outcome=outcome,
            reason_code=reason_code,
            recorded_at=recorded_at,
            media_type=media_type,
            duration_us=duration_us,
            handedness=handedness,
            mirror_state=mirror_state,
            rotation_degrees=rotation_degrees,
            consent_grant=(consent_grant.read_bytes() if consent_grant is not None else None),
        )
    except (CaptureLedgerError, OSError, TypeError, ValueError) as error:
        typer.echo("Capture attempt update failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Capture attempt: {result.status}.")
    typer.echo(f"Capture outcome: {result.attempt.outcome}.")
    typer.echo(f"Collection sidecar SHA-256: {result.collection_sidecar_sha256}")


@app.command("import-capture")
def import_capture_command(
    sidecar: Annotated[
        Path,
        typer.Argument(help="Finalized fixture-only collection-sidecar/1 JSON document."),
    ],
    source_map: Annotated[
        Path,
        typer.Option("--source-map", help="Private JSON map of opaque source keys to paths."),
    ],
    source_root: Annotated[
        Path,
        typer.Option("--source-root", help="Explicit root for private source-map paths."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New or byte-identical raw dataset bundle directory."),
    ],
) -> None:
    """Atomically import a complete synthetic sidecar into a validated raw bundle."""

    from signlab.datasets.importer import DatasetImportError, import_collection_sidecar

    try:
        result = import_collection_sidecar(
            sidecar.read_bytes(),
            source_root=source_root,
            source_map=_read_capture_source_map(source_map),
            destination=output,
        )
    except (DatasetImportError, OSError, TypeError, ValueError) as error:
        typer.echo("Capture import failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Capture import: {result.status}.")
    typer.echo(
        "Imported outcomes: "
        f"{result.accepted_recordings} accepted, "
        f"{result.retry_attempts} retry, "
        f"{result.quarantined_attempts} quarantined, "
        f"{result.skipped_occurrences} skipped."
    )
    typer.echo(f"Raw data SHA-256: {result.manifest.raw_data_sha256}")
    typer.echo("Raw bundle integrity: verified.")


@app.command("validate-raw-dataset")
def validate_raw_dataset_command(
    manifest: Annotated[
        Path,
        typer.Argument(help="Raw-dataset-manifest/1 JSON document."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Explicit raw bundle root."),
    ],
) -> None:
    """Verify raw metadata, normalized tables, and every referenced media byte."""

    from signlab.datasets.raw_bundle import RawDatasetBundleError, validate_raw_dataset_bundle

    try:
        result = validate_raw_dataset_bundle(manifest.read_bytes(), workspace_root)
    except (RawDatasetBundleError, OSError, TypeError, ValueError) as error:
        typer.echo("Raw dataset validation failed.", err=True)
        raise typer.Exit(code=1) from error
    checked = result.validation
    typer.echo(f"Raw data SHA-256: {checked.raw_data_sha256}")
    typer.echo(f"Parquet table bytes: {checked.parquet_table_bytes}")
    typer.echo(f"Raw dataset semantics: {checked.semantic_integrity}")
    typer.echo(f"Referenced artifact bytes: {checked.artifact_byte_integrity}")
    typer.echo(f"Collection sidecar: {checked.collection_sidecar_integrity}")
    typer.echo(f"Lineage inventory: {checked.lineage_inventory_integrity}")
    typer.echo(f"Quarantine inventory: {checked.quarantine_inventory_integrity}")
    typer.echo(f"Current consent authorization: {checked.consent_authorization.replace('_', ' ')}")


@app.command("plan-external-dataset")
def plan_external_dataset_command(
    source: Annotated[
        Literal["popsign-asl-v1"],
        typer.Argument(help="Registered licensed dataset release to plan."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New or byte-identical acquisition-plan JSON file."),
    ],
) -> None:
    """Write the reviewed offline plan without downloading human media."""

    from signlab.datasets.popsign import (
        PopSignDatasetError,
        build_popsign_v1_plan,
        write_external_acquisition_plan,
    )

    try:
        plan = build_popsign_v1_plan()
        result = write_external_acquisition_plan(plan, output)
    except (PopSignDatasetError, OSError, TypeError, ValueError) as error:
        typer.echo("External dataset planning failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"External dataset plan: {result.status}.")
    typer.echo("Registered source: PopSign ASL v1.0.")
    typer.echo(f"Planned archives: {result.archive_count}")
    typer.echo(f"Acquisition plan SHA-256: {result.plan_sha256}")
    typer.echo("Network access: not used.")


def _echo_external_dataset_summary(
    *,
    status: str,
    content_sha256: str,
    archive_count: int,
    media_count: int,
    semantic_integrity: str,
    media_byte_integrity: str,
    archive_byte_integrity: str,
    license_authorization: str,
) -> None:
    """Emit only aggregate, path-free evidence for licensed external media."""

    typer.echo(f"External dataset: {status.replace('_', ' ')}.")
    typer.echo(f"External data SHA-256: {content_sha256}")
    typer.echo(f"Licensed archives: {archive_count}")
    typer.echo(f"Imported media: {media_count}")
    typer.echo(f"Dataset semantics: {semantic_integrity.replace('_', ' ')}")
    typer.echo(f"Imported media bytes: {media_byte_integrity.replace('_', ' ')}")
    typer.echo(f"Original archive bytes: {archive_byte_integrity.replace('_', ' ')}")
    typer.echo(f"License authorization: {license_authorization.replace('_', ' ')}")
    typer.echo("SignLab participant consent: not applicable to licensed public data.")


@app.command("import-popsign")
def import_popsign_command(
    plan: Annotated[
        Path,
        typer.Argument(help="Exact reviewed external-acquisition-plan/1 JSON document."),
    ],
    archive_root: Annotated[
        Path,
        typer.Option("--archive-root", help="Local root containing every planned archive."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New or byte-identical external dataset bundle."),
    ],
    accept_license: Annotated[
        str,
        typer.Option(
            "--accept-license",
            help="Explicit license acknowledgement; this release requires CC-BY-4.0.",
        ),
    ],
) -> None:
    """Import explicitly licensed local PopSign archives without network access."""

    from signlab.datasets.popsign import PopSignDatasetError, import_popsign_v1_archives

    try:
        result = import_popsign_v1_archives(
            plan.read_bytes(),
            archive_root=archive_root,
            destination=output,
            accept_license=accept_license,
        )
    except (PopSignDatasetError, OSError, TypeError, ValueError) as error:
        typer.echo("External dataset import failed.", err=True)
        raise typer.Exit(code=1) from error
    checked = result.validation
    _echo_external_dataset_summary(
        status=result.status,
        content_sha256=checked.content_sha256,
        archive_count=checked.archive_count,
        media_count=checked.media_count,
        semantic_integrity=checked.semantic_integrity,
        media_byte_integrity=checked.media_byte_integrity,
        archive_byte_integrity=checked.archive_byte_integrity,
        license_authorization=checked.license_authorization,
    )


@app.command("validate-external-dataset")
def validate_external_dataset_command(
    manifest: Annotated[
        Path,
        typer.Argument(help="External-dataset-manifest/1 JSON document."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Explicit external bundle root."),
    ],
    archive_root: Annotated[
        Path | None,
        typer.Option(
            "--archive-root",
            help="Optional local archive root for exact source-byte revalidation.",
        ),
    ] = None,
) -> None:
    """Verify licensed-media lineage, inventory, and content-addressed bytes."""

    from signlab.datasets.popsign import (
        PopSignDatasetError,
        validate_external_dataset_bundle,
    )

    try:
        checked = validate_external_dataset_bundle(
            manifest.read_bytes(),
            workspace_root,
            archive_root=archive_root,
        )
    except (PopSignDatasetError, OSError, TypeError, ValueError) as error:
        typer.echo("External dataset validation failed.", err=True)
        raise typer.Exit(code=1) from error
    _echo_external_dataset_summary(
        status="verified",
        content_sha256=checked.content_sha256,
        archive_count=checked.archive_count,
        media_count=checked.media_count,
        semantic_integrity=checked.semantic_integrity,
        media_byte_integrity=checked.media_byte_integrity,
        archive_byte_integrity=checked.archive_byte_integrity,
        license_authorization=checked.license_authorization,
    )


@app.command("build-public-corpus")
def build_public_corpus_command(
    manifest: Annotated[
        Path,
        typer.Argument(help="External-dataset-manifest/1 JSON document."),
    ],
    external_root: Annotated[
        Path,
        typer.Option("--external-root", help="Validated external dataset bundle root."),
    ],
    model_root: Annotated[
        Path,
        typer.Option("--model-root", help="Directory containing the two pinned task assets."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Local content-addressed public corpus directory."),
    ],
    archive_root: Annotated[
        Path | None,
        typer.Option(
            "--archive-root",
            help="Optional official archive root for exact source-byte revalidation.",
        ),
    ] = None,
    max_candidates_per_group: Annotated[
        int,
        typer.Option(
            "--max-candidates-per-group",
            min=1,
            help="Legacy-slice candidates tried per split and target.",
        ),
    ] = 5,
    trainable_smoke: Annotated[
        bool,
        typer.Option(
            "--trainable-smoke",
            help="Aim for fixed 10/3/3 signer-distinct quotas under 750 attempts.",
        ),
    ] = False,
) -> None:
    """Run one bounded licensed PopSign corpus through the existing pipeline."""

    from signlab.datasets.public_corpus import build_public_corpus

    def show_progress(index: int, total: int, accepted: bool) -> None:
        interval = 25 if trainable_smoke else 10
        if index == 1 or index % interval == 0 or index == total:
            unit = "attempts" if trainable_smoke else "groups"
            typer.echo(
                f"Public corpus {unit}: {index}/{total} "
                f"(latest {'selected' if accepted else 'unfilled'})."
            )

    try:
        result = build_public_corpus(
            manifest,
            external_root=external_root,
            model_root=model_root,
            output_root=output,
            archive_root=archive_root,
            max_candidates_per_group=max_candidates_per_group,
            trainable_smoke=trainable_smoke,
            progress=show_progress,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Public corpus build failed.", err=True)
        raise typer.Exit(code=1) from error
    if trainable_smoke:
        typer.echo(f"Trainable smoke decision: {result.decision.upper()}.")
        typer.echo(f"Selected usable clips: {result.selected_count}/{result.target_count}.")
        typer.echo(f"Attempted videos: {result.attempted_count}/{result.attempt_limit}.")
    else:
        typer.echo(f"Selected usable clips: {result.selected_count}/{result.group_count} groups.")
    typer.echo(f"Coded unselected or unusable clips: {result.exclusion_count}.")
    typer.echo(f"Public corpus SHA-256: {result.corpus_sha256}")
    typer.echo("Aggregate JSON and Markdown reports: written.")
    typer.echo("Claim scope: licensed isolated public data only.")


@app.command("freeze-public-corpus-split")
def freeze_public_corpus_split_command(
    manifest: Annotated[
        Path,
        typer.Argument(help="External-dataset-manifest/1 JSON document."),
    ],
    source_root: Annotated[
        Path,
        typer.Option(
            "--source-root",
            help="Immutable #79 corpus root containing the 750 retained landmarks.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New exact-80 public split directory."),
    ],
) -> None:
    """Freeze the exact PopSign split from retained landmarks without MediaPipe."""

    from signlab.datasets.public_split import freeze_public_corpus_split

    def show_progress(index: int, total: int) -> None:
        if index == 1 or index % 50 == 0 or index == total:
            typer.echo(f"Retained landmark replay: {index}/{total}.")

    try:
        result = freeze_public_corpus_split(
            manifest,
            source_root=source_root,
            output_root=output,
            progress=show_progress,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Public corpus split freeze failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Public corpus split: frozen and verified.")
    typer.echo(f"Attempt landmarks: {result.attempt_count} verified.")
    typer.echo(
        "Window and quality: "
        f"{result.pass_count} pass, {result.warning_count} warning, "
        f"{result.quarantine_count} quarantine, {result.no_window_count} no window."
    )
    typer.echo(f"Selected usable clips: {result.selected_count}/80.")
    typer.echo("Partitions: 50 train, 15 validation, 15 test.")
    typer.echo(f"Public split SHA-256: {result.split_sha256}")
    typer.echo("Claim scope: licensed isolated public data only.")


def _echo_landmark_extraction_summary(
    *,
    status: str,
    manifest_sha256: str,
    config_sha256: str,
    raw_data_sha256: str,
    sequence_count: int,
    frame_count: int,
    invalid_frame_count: int,
    integrity: str,
) -> None:
    """Emit only path-free, aggregate extraction evidence."""

    typer.echo(f"Landmark extraction: {status.replace('_', ' ')}.")
    typer.echo(f"Extraction manifest SHA-256: {manifest_sha256}")
    typer.echo(f"Extraction configuration SHA-256: {config_sha256}")
    typer.echo(f"Raw data SHA-256: {raw_data_sha256}")
    typer.echo(f"Extracted sequences: {sequence_count}")
    typer.echo(f"Landmark frames: {frame_count}")
    typer.echo(f"Invalid frames: {invalid_frame_count}")
    typer.echo(f"Extraction bundle integrity: {integrity.replace('_', ' ')}")


@app.command("extract-landmarks")
def extract_landmarks_command(
    raw_manifest: Annotated[
        Path,
        typer.Argument(help="Raw-dataset-manifest/1 JSON document."),
    ],
    raw_bundle_root: Annotated[
        Path,
        typer.Option("--raw-bundle-root", help="Explicit validated raw bundle root."),
    ],
    model_root: Annotated[
        Path,
        typer.Option("--model-root", help="Directory containing the two pinned task assets."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New or byte-identical landmark bundle directory."),
    ],
) -> None:
    """Extract an eligible synthetic raw bundle with the packaged pinned config."""

    from signlab.extraction.batch import extract_raw_dataset
    from signlab.extraction.resources import load_packaged_default_extraction_config

    try:
        result = extract_raw_dataset(
            raw_manifest.read_bytes(),
            raw_bundle_root=raw_bundle_root,
            model_root=model_root,
            config=load_packaged_default_extraction_config(),
            destination=output,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Landmark extraction failed.", err=True)
        raise typer.Exit(code=1) from error
    validation = result.validation
    _echo_landmark_extraction_summary(
        status=result.status,
        manifest_sha256=result.manifest.manifest_sha256,
        config_sha256=result.manifest.config_sha256,
        raw_data_sha256=result.manifest.raw_data_sha256,
        sequence_count=validation.sequence_count,
        frame_count=validation.frame_count,
        invalid_frame_count=validation.invalid_frame_count,
        integrity=validation.semantic_integrity,
    )


@app.command("validate-extraction")
def validate_extraction_command(
    extraction_manifest: Annotated[
        Path,
        typer.Argument(help="Landmark-extraction-manifest/1 JSON document."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Explicit landmark bundle root."),
    ],
    raw_manifest: Annotated[
        Path,
        typer.Option("--raw-manifest", help="Exact raw-dataset-manifest/1 JSON document."),
    ],
    raw_bundle_root: Annotated[
        Path,
        typer.Option("--raw-bundle-root", help="Explicit validated raw bundle root."),
    ],
) -> None:
    """Verify extraction lineage, inventory, Parquet bytes, and semantic rows."""

    from signlab.extraction.batch import validate_landmark_extraction_bundle

    try:
        result = validate_landmark_extraction_bundle(
            extraction_manifest.read_bytes(),
            workspace_root,
            raw_manifest=raw_manifest.read_bytes(),
            raw_bundle_root=raw_bundle_root,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Landmark extraction validation failed.", err=True)
        raise typer.Exit(code=1) from error
    validation = result.validation
    _echo_landmark_extraction_summary(
        status="verified",
        manifest_sha256=result.manifest.manifest_sha256,
        config_sha256=result.manifest.config_sha256,
        raw_data_sha256=result.manifest.raw_data_sha256,
        sequence_count=validation.sequence_count,
        frame_count=validation.frame_count,
        invalid_frame_count=validation.invalid_frame_count,
        integrity=validation.semantic_integrity,
    )


def _echo_landmark_quality_summary(
    *,
    status: str,
    manifest_sha256: str,
    policy_sha256: str,
    extraction_manifest_sha256: str,
    raw_dataset_manifest_sha256: str,
    sequence_count: int,
    pass_count: int,
    warning_count: int,
    quarantine_count: int,
    reject_count: int,
    dataset_status: str,
    integrity: str,
) -> None:
    """Emit only path-free, aggregate landmark-quality evidence."""

    typer.echo(f"Landmark quality: {status.replace('_', ' ')}.")
    typer.echo(f"Quality manifest SHA-256: {manifest_sha256}")
    typer.echo(f"Quality policy SHA-256: {policy_sha256}")
    typer.echo(f"Extraction manifest SHA-256: {extraction_manifest_sha256}")
    typer.echo(f"Raw dataset manifest SHA-256: {raw_dataset_manifest_sha256}")
    typer.echo(f"Assessed sequences: {sequence_count}")
    typer.echo(
        "Quality dispositions: "
        f"{pass_count} pass, "
        f"{warning_count} warning, "
        f"{quarantine_count} quarantine, "
        f"{reject_count} reject."
    )
    typer.echo(f"Dataset quality status: {dataset_status.replace('_', ' ')}")
    typer.echo(f"Quality report recomputation: {integrity.replace('_', ' ')}")


@app.command("assess-landmark-quality")
def assess_landmark_quality_command(
    extraction_manifest: Annotated[
        Path,
        typer.Argument(help="Landmark-extraction-manifest/1 JSON document."),
    ],
    extraction_root: Annotated[
        Path,
        typer.Option("--extraction-root", help="Explicit validated landmark bundle root."),
    ],
    raw_manifest: Annotated[
        Path,
        typer.Option("--raw-manifest", help="Exact raw-dataset-manifest/1 JSON document."),
    ],
    raw_bundle_root: Annotated[
        Path,
        typer.Option("--raw-bundle-root", help="Explicit validated raw bundle root."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New or byte-identical quality report bundle directory."),
    ],
) -> None:
    """Assess an eligible synthetic landmark bundle with the packaged policy."""

    from signlab.quality.batch import assess_landmark_quality
    from signlab.quality.resources import load_packaged_default_quality_policy

    try:
        result = assess_landmark_quality(
            extraction_manifest.read_bytes(),
            extraction_root=extraction_root,
            raw_manifest=raw_manifest.read_bytes(),
            raw_bundle_root=raw_bundle_root,
            policy=load_packaged_default_quality_policy(),
            destination=output,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Landmark quality assessment failed.", err=True)
        raise typer.Exit(code=1) from error
    validation = result.validation
    _echo_landmark_quality_summary(
        status=result.status,
        manifest_sha256=result.manifest.manifest_sha256,
        policy_sha256=result.manifest.policy_sha256,
        extraction_manifest_sha256=result.manifest.extraction_manifest_sha256,
        raw_dataset_manifest_sha256=result.manifest.raw_dataset_manifest_sha256,
        sequence_count=validation.sequence_count,
        pass_count=validation.pass_count,
        warning_count=validation.warning_count,
        quarantine_count=validation.quarantine_count,
        reject_count=validation.reject_count,
        dataset_status=result.manifest.dataset_report.status,
        integrity=validation.report_recomputation,
    )


@app.command("validate-landmark-quality")
def validate_landmark_quality_command(
    quality_manifest: Annotated[
        Path,
        typer.Argument(help="Landmark-quality-manifest/1 JSON document."),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Explicit quality report bundle root."),
    ],
    extraction_manifest: Annotated[
        Path,
        typer.Option(
            "--extraction-manifest",
            help="Exact landmark-extraction-manifest/1 JSON document.",
        ),
    ],
    extraction_root: Annotated[
        Path,
        typer.Option("--extraction-root", help="Explicit validated landmark bundle root."),
    ],
    raw_manifest: Annotated[
        Path,
        typer.Option("--raw-manifest", help="Exact raw-dataset-manifest/1 JSON document."),
    ],
    raw_bundle_root: Annotated[
        Path,
        typer.Option("--raw-bundle-root", help="Explicit validated raw bundle root."),
    ],
) -> None:
    """Verify quality source bindings, inventory, and every recomputed report."""

    from signlab.quality.batch import validate_landmark_quality_bundle

    try:
        result = validate_landmark_quality_bundle(
            quality_manifest.read_bytes(),
            workspace_root,
            extraction_manifest=extraction_manifest.read_bytes(),
            extraction_root=extraction_root,
            raw_manifest=raw_manifest.read_bytes(),
            raw_bundle_root=raw_bundle_root,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Landmark quality validation failed.", err=True)
        raise typer.Exit(code=1) from error
    validation = result.validation
    _echo_landmark_quality_summary(
        status="verified",
        manifest_sha256=result.manifest.manifest_sha256,
        policy_sha256=result.manifest.policy_sha256,
        extraction_manifest_sha256=result.manifest.extraction_manifest_sha256,
        raw_dataset_manifest_sha256=result.manifest.raw_dataset_manifest_sha256,
        sequence_count=validation.sequence_count,
        pass_count=validation.pass_count,
        warning_count=validation.warning_count,
        quarantine_count=validation.quarantine_count,
        reject_count=validation.reject_count,
        dataset_status=result.manifest.dataset_report.status,
        integrity=validation.report_recomputation,
    )


@app.command("configure-private-remote")
def configure_private_remote_command() -> None:
    """Configure a credential-free private S3 DVC remote in ignored local state."""

    from signlab.reproducibility.remote import (
        DvcRemoteConfigurationError,
        configure_private_dvc_remote,
    )

    try:
        result = configure_private_dvc_remote(Path.cwd())
    except (OSError, DvcRemoteConfigurationError) as error:
        typer.echo("Private DVC remote configuration failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Private DVC remote configured locally: "
        f"endpoint override {str(result.endpoint_configured).lower()}, "
        f"region override {str(result.region_configured).lower()}."
    )


@app.command("capture-reproduction-snapshot")
def capture_reproduction_snapshot_command(
    metadata_repository_role: Annotated[
        Literal["public-fixture", "protected-metadata"],
        typer.Option(
            "--repository-role",
            help=(
                "Declare whether this checkout is the public fixture or "
                "protected metadata repository."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New repository-relative JSON path beneath reports/reproduction/.",
        ),
    ] = Path("reports/reproduction/dvc-snapshot.json"),
) -> None:
    """Capture clean Git/DVC identities for later experiment tracking."""

    from signlab.reproducibility.evidence import (
        DvcEvidenceError,
        capture_dvc_snapshot,
        write_dvc_snapshot,
    )
    from signlab.reproducibility.provenance import dvc_snapshot_digest

    try:
        snapshot = capture_dvc_snapshot(
            Path.cwd(),
            metadata_repository_role=metadata_repository_role,
        )
        write_dvc_snapshot(snapshot, Path.cwd(), output.as_posix())
        snapshot_sha256 = dvc_snapshot_digest(snapshot)
    except (OSError, DvcEvidenceError, ValueError) as error:
        typer.echo("DVC reproduction snapshot could not be captured.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"DVC reproduction snapshot SHA-256: {snapshot_sha256}")


@app.command("run-reproduction-stage")
def run_reproduction_stage_command(
    stage: Annotated[
        Literal["ingest", "validate", "extract", "quality", "split", "feature"],
        typer.Argument(help="Registered public-fixture stage to execute."),
    ],
) -> None:
    """Run one deterministic stage in the synthetic DVC proof graph."""

    from signlab.reproducibility.stages import ReproductionStageError, run_reproduction_stage

    try:
        run_reproduction_stage(stage, Path.cwd())
    except (OSError, ReproductionStageError) as error:
        typer.echo("Synthetic reproduction stage failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Synthetic reproduction stage completed: {stage}.")


@app.command("validate-resources")
def validate_dataset_resources() -> None:
    """Validate all packaged data-foundation resources."""

    from signlab.datasets.external_resources import (
        validate_packaged_external_dataset_resources,
    )
    from signlab.datasets.ingest_resources import validate_packaged_ingest_resources
    from signlab.datasets.resources import validate_packaged_dataset_resources
    from signlab.extraction.resources import validate_packaged_extraction_resources
    from signlab.features.resources import validate_packaged_feature_resources
    from signlab.quality.resources import validate_packaged_quality_resources

    try:
        validate_packaged_dataset_resources()
        validate_packaged_external_dataset_resources()
        validate_packaged_ingest_resources()
        validate_packaged_extraction_resources()
        validate_packaged_quality_resources()
        validate_packaged_feature_resources()
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Packaged dataset resource validation failed.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Packaged dataset, external, ingest, extraction, quality, and feature resources are valid."
    )


@app.command("write-example-dataset")
def write_example_dataset(
    output: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            resolve_path=True,
            help="New or empty output directory for the synthetic bundle.",
        ),
    ],
) -> None:
    """Write a synthetic bundle and verify its tables and relationships."""

    from signlab.datasets.bundle import write_dataset_bundle
    from signlab.datasets.resources import build_example_dataset_bundle

    try:
        example = build_example_dataset_bundle()
        write_dataset_bundle(example.manifest, example.tables, output)
    except (OSError, TypeError, ValueError) as error:
        typer.echo("Synthetic dataset bundle could not be written and verified.", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Synthetic bundle written; Parquet table bytes and relationships verified.")


@app.command("validate-dataset")
def validate_dataset(
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Table-backed dataset-manifest/2 JSON document.",
        ),
    ],
    workspace_root: Annotated[
        Path,
        typer.Option(
            "--workspace-root",
            help="Explicit root for workspace-relative Parquet table locators.",
        ),
    ],
    split: Annotated[
        Path | None,
        typer.Option(
            "--split",
            help="Optional exact split-manifest/1 to reconcile.",
        ),
    ] = None,
    verify_row_artifacts: Annotated[
        bool,
        typer.Option(
            "--verify-row-artifacts",
            help="Stream-check every recording, materialized clip, and derived artifact.",
        ),
    ] = False,
) -> None:
    """Verify Parquet bytes, schemas, semantic relationships, and optional split."""

    # Keep PyArrow off the import path for unrelated CLI commands.
    from signlab.contracts.pipeline import validate_split_manifest
    from signlab.datasets.bundle import validate_dataset_bundle

    try:
        manifest_document = manifest.read_bytes()
        checked_split = validate_split_manifest(split.read_bytes()) if split is not None else None
        result = validate_dataset_bundle(
            manifest_document,
            workspace_root,
            split=checked_split,
            verify_row_artifacts=verify_row_artifacts,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo(
            "Dataset validation failed: manifest, table bytes, or relationships are invalid.",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"Dataset data SHA-256: {result.data_sha256}")
    typer.echo(f"Parquet table bytes: {result.parquet_table_bytes.replace('_', ' ')}")
    typer.echo(f"Dataset semantic integrity: {result.semantic_integrity.replace('_', ' ')}")
    typer.echo(f"Referenced row artifacts: {result.artifact_byte_integrity.replace('_', ' ')}")
    typer.echo(f"Split compatibility: {result.split_compatibility.replace('_', ' ')}")
    typer.echo(f"Current consent authorization: {result.consent_authorization.replace('_', ' ')}")


@app.command("export-legacy")
def export_legacy(
    legacy_root: Annotated[
        Path,
        typer.Option(
            "--legacy-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Legacy project root to inspect read-only.",
        ),
    ],
    audit_snapshot: Annotated[
        Path,
        typer.Option(
            "--audit-snapshot",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Immutable legacy-state audit snapshot.",
        ),
    ],
    public_output: Annotated[
        Path,
        typer.Option(
            "--public-output",
            resolve_path=True,
            help="Empty target for portfolio-safe evidence.",
        ),
    ],
    quarantine_output: Annotated[
        Path,
        typer.Option(
            "--quarantine-output",
            resolve_path=True,
            help="Empty ignored target for private content-addressed objects.",
        ),
    ],
) -> None:
    """Export sanitized evidence and an ignored private quarantine."""
    try:
        summary = export_legacy_evidence(
            legacy_root=legacy_root,
            audit_snapshot=audit_snapshot,
            public_output=public_output,
            quarantine_output=quarantine_output,
        )
    except LegacyExportError as error:
        typer.echo(f"Legacy export failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "Legacy export complete: "
        f"{summary.runs} runs, {summary.attempts} attempts, "
        f"{summary.promoted_models} promoted models, "
        f"{summary.quarantined_segments} quarantined segments."
    )


@app.command("validate-legacy")
def validate_legacy(
    public_root: Annotated[
        Path,
        typer.Option(
            "--public-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Public legacy-export root.",
        ),
    ],
    quarantine_root: Annotated[
        Path | None,
        typer.Option(
            "--quarantine-root",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional private quarantine root.",
        ),
    ] = None,
) -> None:
    """Validate public evidence and optionally verify every private object."""
    try:
        summary = validate_legacy_export(
            public_root=public_root,
            quarantine_root=quarantine_root,
        )
    except LegacyExportError as error:
        typer.echo(f"Legacy export validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    private_status = " and private quarantine" if summary.quarantine_verified else ""
    typer.echo(f"Validated {summary.runs} runs and {summary.attempts} attempts{private_status}.")
