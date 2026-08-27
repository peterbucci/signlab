"""Offline, deterministic acquisition and import boundary for PopSign ASL v1.0.

This module deliberately does not download anything.  It turns the reviewed
package resources into an operator-facing archive plan, then imports only the
local archives named by that plan.  Upstream member names are used transiently
to derive opaque identities and never appear in the published manifest.
"""

from __future__ import annotations

import hashlib
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Final, Literal, cast

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.external_dataset import (
    ExternalAcquisitionPlanV1,
    ExternalArchivePlanV1,
    ExternalArchiveRecordV1,
    ExternalDatasetContractError,
    ExternalDatasetManifestV1,
    ExternalLicenseAcknowledgementV1,
    ExternalMediaRecordV1,
    ExternalSplit,
    ExternalTargetLabel,
    SourceLabel,
    external_acquisition_plan_digest,
    external_dataset_manifest_digest,
    validate_external_acquisition_plan,
    validate_external_dataset_manifest,
)
from signlab.datasets.external_resources import (
    ExternalDatasetResourceError,
    external_resource_reference,
    load_popsign_source,
    load_signlab_five_popsign_selection,
    render_external_dataset_json,
)

type ExternalAcquisitionPlanInput = (
    ExternalAcquisitionPlanV1 | str | bytes | bytearray | Mapping[str, object]
)
type ExternalDatasetManifestInput = (
    ExternalDatasetManifestV1 | str | bytes | bytearray | Mapping[str, object]
)
type ExternalDatasetErrorCategory = Literal[
    "plan.invalid",
    "plan.conflict",
    "license.denied",
    "archive.root_invalid",
    "archive.missing",
    "archive.bytes_invalid",
    "archive.structure_invalid",
    "archive.limit_exceeded",
    "archive.member_invalid",
    "split.leakage",
    "destination.invalid",
    "destination.conflict",
    "publication.failed",
    "manifest.invalid",
    "bundle.inventory_invalid",
    "media.bytes_invalid",
]

EXTERNAL_DATASET_MANIFEST_FILENAME: Final = "external-dataset-manifest.json"
POPSIGN_PLAN_ID: Final = "popsign_v1_signlab_five"
POPSIGN_DATASET_ID: Final = "popsign_v1_signlab_five"
POPSIGN_LICENSE_ACKNOWLEDGEMENT: Final = "CC-BY-4.0"

_CHUNK_SIZE: Final = 1024 * 1024
_MAX_PLAN_BYTES: Final = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 32 * 1024 * 1024 * 1024
_MAX_TAR_ENTRIES: Final = 100_000
_MAX_MEDIA_PER_ARCHIVE: Final = 50_000
_MAX_MEMBER_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES: Final = 64 * 1024 * 1024 * 1024
_OPAQUE_DIGEST_PREFIX: Final = b"signlab.external.popsign/1\0"
_PROVIDER_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$")

_ERROR_MESSAGES: Final[dict[ExternalDatasetErrorCategory, str]] = {
    "plan.invalid": "the external acquisition plan is invalid or unsupported",
    "plan.conflict": "the acquisition-plan destination contains different content",
    "license.denied": "the required external dataset license was not acknowledged",
    "archive.root_invalid": "the local external archive root is invalid",
    "archive.missing": "a required external dataset archive is unavailable",
    "archive.bytes_invalid": "external archive bytes changed or could not be verified",
    "archive.structure_invalid": "an external archive has an unsafe or invalid structure",
    "archive.limit_exceeded": "an external archive exceeds a configured safety limit",
    "archive.member_invalid": "an external archive member is invalid",
    "split.leakage": "one external signer appears in more than one source split",
    "destination.invalid": "the external dataset destination is invalid",
    "destination.conflict": "the external dataset destination contains different content",
    "publication.failed": "the external dataset bundle could not be published",
    "manifest.invalid": "the external dataset manifest is invalid or incompatible",
    "bundle.inventory_invalid": "the external dataset bundle inventory is invalid",
    "media.bytes_invalid": "external dataset media bytes could not be verified",
}


class PopSignDatasetError(ValueError):
    """Stable, path-free failure at the PopSign application boundary."""

    def __init__(self, category: ExternalDatasetErrorCategory) -> None:
        self.category = category
        self.code = f"dataset.external.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class ExternalAcquisitionPlanWriteResult:
    """Path-free evidence that a deterministic plan was written or retained."""

    status: Literal["published", "unchanged"]
    plan_sha256: str
    archive_count: int


@dataclass(frozen=True, slots=True)
class ExternalDatasetValidationResult:
    """Positive, aggregate evidence returned after a full offline check."""

    content_sha256: str
    archive_count: int
    media_count: int
    semantic_integrity: Literal["verified"]
    media_byte_integrity: Literal["verified"]
    archive_byte_integrity: Literal["verified", "not_checked"]
    license_authorization: Literal["verified"]


@dataclass(frozen=True, slots=True)
class ImportedExternalDatasetBundle:
    """One newly published or already-identical licensed-media bundle."""

    status: Literal["published", "unchanged"]
    manifest: ExternalDatasetManifestV1
    validation: ExternalDatasetValidationResult


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int]


class _HashingReader:
    """Minimal read-only wrapper that hashes the exact tar stream consumed."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        captured = self._stream.read(size)
        self._digest.update(captured)
        self.size_bytes += len(captured)
        return captured

    @property
    def sha256(self) -> str:
        return f"sha256:{self._digest.hexdigest()}"


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _bounded_document(document: object, limit: int, category: ExternalDatasetErrorCategory) -> None:
    if isinstance(document, (bytes, bytearray)) and len(document) > limit:
        raise PopSignDatasetError(category)
    if isinstance(document, str) and len(document.encode("utf-8", errors="ignore")) > limit:
        raise PopSignDatasetError(category)


def _validate_exact_plan(document: ExternalAcquisitionPlanInput) -> ExternalAcquisitionPlanV1:
    _bounded_document(document, _MAX_PLAN_BYTES, "plan.invalid")
    try:
        checked = validate_external_acquisition_plan(document)
        if checked != build_popsign_v1_plan():
            raise PopSignDatasetError("plan.invalid")
        return checked
    except PopSignDatasetError:
        raise
    except (
        ExternalDatasetContractError,
        ExternalDatasetResourceError,
        TypeError,
        ValueError,
    ) as error:
        raise PopSignDatasetError("plan.invalid") from error


def build_popsign_v1_plan() -> ExternalAcquisitionPlanV1:
    """Build the exact offline 15-archive plan for the reviewed five targets."""

    try:
        source = load_popsign_source()
        selection = load_signlab_five_popsign_selection()
        source_reference = external_resource_reference(source)
        selection_reference = external_resource_reference(selection)
        archives = tuple(
            ExternalArchivePlanV1(
                schema_version="external-archive-plan/1",
                archive_id=(f"popsign_v1_{selection.category}_{split}_{mapping.source_label}"),
                category=selection.category,
                split=split,
                source_label=mapping.source_label,
                download_url=source.download_url_template.format(
                    download_id=source.download_id,
                    category=selection.category,
                    split=split,
                    source_label=mapping.source_label,
                ),
                local_archive=WorkspaceRelativeLocatorV1(
                    kind="workspace_relative",
                    path=(f"archives/{selection.category}/{split}/{mapping.source_label}.tar"),
                ),
                archive_format="tar",
                publisher_sha256=None,
                integrity_basis="trust_on_first_use_then_sha256",
            )
            for split in selection.splits
            for mapping in sorted(selection.mappings, key=lambda item: item.source_label)
        )
        if len(archives) != 15:
            raise PopSignDatasetError("plan.invalid")
        payload: dict[str, object] = {
            "schema_version": "external-acquisition-plan/1",
            "plan_id": POPSIGN_PLAN_ID,
            "version": "1.0.0",
            "source": source_reference.model_dump(mode="json", round_trip=True),
            "selection": selection_reference.model_dump(mode="json", round_trip=True),
            "network_access": "forbidden",
            "preview_media": "forbidden",
            "required_license_acknowledgement": POPSIGN_LICENSE_ACKNOWLEDGEMENT,
            "archives": [item.model_dump(mode="json", round_trip=True) for item in archives],
        }
        payload["plan_sha256"] = external_acquisition_plan_digest(payload)
        return ExternalAcquisitionPlanV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except PopSignDatasetError:
        raise
    except (
        ExternalDatasetContractError,
        ExternalDatasetResourceError,
        TypeError,
        ValueError,
    ) as error:
        raise PopSignDatasetError("plan.invalid") from error


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_or_identical(
    destination: Path,
    content: bytes,
) -> Literal["published", "unchanged"]:
    temporary: Path | None = None
    try:
        if not destination.name or _is_linklike(destination):
            raise PopSignDatasetError("plan.conflict")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _is_linklike(destination.parent) or not destination.parent.is_dir():
            raise PopSignDatasetError("plan.conflict")
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == content:
                return "unchanged"
            raise PopSignDatasetError("plan.conflict")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if (
                not _is_linklike(destination)
                and destination.is_file()
                and destination.read_bytes() == content
            ):
                return "unchanged"
            raise PopSignDatasetError("plan.conflict") from None
        with suppress(OSError):
            _fsync_directory(destination.parent)
        return "published"
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("publication.failed") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def write_external_acquisition_plan(
    plan: ExternalAcquisitionPlanInput,
    destination: str | os.PathLike[str],
) -> ExternalAcquisitionPlanWriteResult:
    """Write the exact plan to a new path, or accept byte-identical existing bytes."""

    checked = _validate_exact_plan(plan)
    content = render_external_dataset_json(checked).encode("utf-8")
    status = _write_new_or_identical(Path(destination), content)
    return ExternalAcquisitionPlanWriteResult(
        status=status,
        plan_sha256=checked.plan_sha256,
        archive_count=len(checked.archives),
    )


def _require_archive_root(archive_root: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(archive_root)
        if _is_linklike(candidate):
            raise PopSignDatasetError("archive.root_invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise PopSignDatasetError("archive.root_invalid")
        return root
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("archive.root_invalid") from error


def _resolve_archive(root: Path, locator: WorkspaceRelativeLocatorV1) -> Path:
    try:
        candidate = root
        for segment in locator.path.split("/"):
            candidate = candidate / segment
            if _is_linklike(candidate):
                raise PopSignDatasetError("archive.bytes_invalid")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise PopSignDatasetError("archive.bytes_invalid")
        return resolved
    except FileNotFoundError as error:
        raise PopSignDatasetError("archive.missing") from error
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("archive.bytes_invalid") from error


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _fingerprint_file(
    path: Path,
    *,
    maximum_bytes: int,
    category: ExternalDatasetErrorCategory,
) -> _FileFingerprint:
    try:
        if _is_linklike(path):
            raise PopSignDatasetError(category)
        before = path.stat()
        if not path.is_file() or before.st_size <= 0 or before.st_size > maximum_bytes:
            raise PopSignDatasetError(category)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                size += len(chunk)
                if size > maximum_bytes:
                    raise PopSignDatasetError(category)
                digest.update(chunk)
        after = path.stat()
        if (
            _is_linklike(path)
            or _stat_identity(before) != _stat_identity(after)
            or size != after.st_size
        ):
            raise PopSignDatasetError(category)
        return _FileFingerprint(
            sha256=f"sha256:{digest.hexdigest()}",
            size_bytes=size,
            identity=_stat_identity(after),
        )
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError(category) from error


def _normalized_member_name(name: str) -> str:
    try:
        if (
            not name
            or len(name) > 1024
            or "\\" in name
            or "\x00" in name
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name)
            or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).drive
        ):
            raise PopSignDatasetError("archive.structure_invalid")
        parts: list[str] = []
        for part in name.split("/"):
            if part == ".":
                continue
            if not part or part == "..":
                raise PopSignDatasetError("archive.structure_invalid")
            parts.append(part)
        if not parts:
            raise PopSignDatasetError("archive.structure_invalid")
        normalized = "/".join(parts)
        normalized.encode("utf-8", errors="strict")
        return normalized
    except PopSignDatasetError:
        raise
    except (UnicodeError, ValueError) as error:
        raise PopSignDatasetError("archive.structure_invalid") from error


def _provider_tokens(normalized_name: str) -> tuple[str, str]:
    basename = normalized_name.rsplit("/", 1)[-1]
    if not basename.endswith("-.mp4"):
        raise PopSignDatasetError("archive.member_invalid")
    stem = basename[: -len("-.mp4")]
    parts = stem.split("--")
    if len(parts) != 2:
        raise PopSignDatasetError("archive.member_invalid")
    participant, recording = parts
    if (
        not 1 <= len(participant) <= 128
        or not 1 <= len(recording) <= 256
        or _PROVIDER_TOKEN.fullmatch(participant) is None
        or _PROVIDER_TOKEN.fullmatch(recording) is None
    ):
        raise PopSignDatasetError("archive.member_invalid")
    return participant, recording


def _domain_digest(domain: str, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(_OPAQUE_DIGEST_PREFIX)
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for part in parts:
        captured = part.encode("utf-8", errors="strict")
        digest.update(len(captured).to_bytes(8, "big"))
        digest.update(captured)
    return digest.hexdigest()


def _opaque_id(prefix: Literal["participant", "recording", "sample"], *parts: str) -> str:
    return f"{prefix}_{_domain_digest(prefix, *parts)[:32]}"


def _copy_or_verify_content_addressed(
    temporary: Path,
    staging: Path,
    sha256: str,
    size_bytes: int,
) -> str:
    digest = sha256.removeprefix("sha256:")
    relative = f"media/sha256/{digest[:2]}/{digest}.mp4"
    destination = staging.joinpath(*relative.split("/"))
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or _is_linklike(destination):
            checked = _fingerprint_file(
                destination,
                maximum_bytes=_MAX_MEMBER_BYTES,
                category="archive.member_invalid",
            )
            if checked.sha256 != sha256 or checked.size_bytes != size_bytes:
                raise PopSignDatasetError("archive.member_invalid")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
            with suppress(OSError):
                _fsync_directory(destination.parent)
        return relative
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("archive.member_invalid") from error


def _read_media_member(
    stream: BinaryIO,
    *,
    declared_size: int,
    staging: Path | None,
) -> tuple[str, int, str | None]:
    temporary: Path | None = None
    try:
        with ExitStack() as resources:
            output: BinaryIO | None = None
            if staging is not None:
                created = resources.enter_context(
                    tempfile.NamedTemporaryFile(
                        mode="w+b",
                        dir=staging,
                        prefix=".external-member-",
                        delete=False,
                    )
                )
                output = cast(BinaryIO, created)
                temporary = Path(created.name)
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                size += len(chunk)
                if size > declared_size or size > _MAX_MEMBER_BYTES:
                    raise PopSignDatasetError("archive.limit_exceeded")
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
            if size != declared_size or size <= 0:
                raise PopSignDatasetError("archive.member_invalid")
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        sha256 = f"sha256:{digest.hexdigest()}"
        relative: str | None = None
        if staging is not None and temporary is not None:
            relative = _copy_or_verify_content_addressed(temporary, staging, sha256, size)
            temporary = None
        return sha256, size, relative
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("archive.member_invalid") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _media_record(
    archive: ExternalArchivePlanV1,
    *,
    normalized_name: str,
    participant_token: str,
    recording_token: str,
    target_label: ExternalTargetLabel,
    sha256: str,
    size_bytes: int,
    locator_path: str,
) -> ExternalMediaRecordV1:
    source_member_fingerprint = (
        f"sha256:{_domain_digest('source-member', archive.archive_id, normalized_name)}"
    )
    identity_scope = ("popsign-asl", "1.0.0")
    participant_id = _opaque_id("participant", *identity_scope, participant_token)
    recording_id = _opaque_id(
        "recording",
        *identity_scope,
        participant_token,
        recording_token,
        archive.archive_id,
        source_member_fingerprint,
    )
    sample_id = _opaque_id(
        "sample",
        *identity_scope,
        archive.archive_id,
        source_member_fingerprint,
        sha256,
    )
    return ExternalMediaRecordV1(
        schema_version="external-media-record/1",
        sample_id=sample_id,
        recording_id=recording_id,
        participant_id=participant_id,
        archive_id=archive.archive_id,
        source_member_fingerprint=source_member_fingerprint,
        category=archive.category,
        source_split=archive.split,
        source_label=archive.source_label,
        target_label_id=target_label,
        media_type="video/mp4",
        sha256=sha256,
        size_bytes=size_bytes,
        locator=WorkspaceRelativeLocatorV1(
            kind="workspace_relative",
            path=locator_path,
        ),
        eligible_for_extraction=True,
    )


def _scan_archive(
    archive_plan: ExternalArchivePlanV1,
    archive_path: Path,
    *,
    target_label: ExternalTargetLabel,
    staging: Path | None,
) -> tuple[ExternalArchiveRecordV1, tuple[ExternalMediaRecordV1, ...]]:
    fingerprint = _fingerprint_file(
        archive_path,
        maximum_bytes=_MAX_ARCHIVE_BYTES,
        category="archive.bytes_invalid",
    )
    members: list[ExternalMediaRecordV1] = []
    seen_names: set[str] = set()
    entry_count = 0
    uncompressed_size = 0
    try:
        with archive_path.open("rb") as raw_stream:
            hashing_reader = _HashingReader(raw_stream)
            with tarfile.open(
                fileobj=cast(BinaryIO, hashing_reader),
                mode="r|",
            ) as opened:
                for member in opened:
                    entry_count += 1
                    if entry_count > _MAX_TAR_ENTRIES:
                        raise PopSignDatasetError("archive.limit_exceeded")
                    normalized_name = _normalized_member_name(member.name)
                    folded = normalized_name.casefold()
                    if folded in seen_names:
                        raise PopSignDatasetError("archive.structure_invalid")
                    seen_names.add(folded)
                    if member.issym() or member.islnk():
                        raise PopSignDatasetError("archive.structure_invalid")
                    if member.isdir():
                        continue
                    if not member.isreg():
                        raise PopSignDatasetError("archive.structure_invalid")
                    if not normalized_name.endswith(".mp4"):
                        raise PopSignDatasetError("archive.member_invalid")
                    if member.size <= 0 or member.size > _MAX_MEMBER_BYTES:
                        raise PopSignDatasetError("archive.limit_exceeded")
                    if len(members) >= _MAX_MEDIA_PER_ARCHIVE:
                        raise PopSignDatasetError("archive.limit_exceeded")
                    uncompressed_size += member.size
                    if uncompressed_size > _MAX_UNCOMPRESSED_BYTES:
                        raise PopSignDatasetError("archive.limit_exceeded")
                    participant_token, recording_token = _provider_tokens(normalized_name)
                    extracted = opened.extractfile(member)
                    if extracted is None:
                        raise PopSignDatasetError("archive.member_invalid")
                    with extracted:
                        sha256, size_bytes, locator = _read_media_member(
                            cast(BinaryIO, extracted),
                            declared_size=member.size,
                            staging=staging,
                        )
                    if locator is None:
                        digest = sha256.removeprefix("sha256:")
                        locator = f"media/sha256/{digest[:2]}/{digest}.mp4"
                    members.append(
                        _media_record(
                            archive_plan,
                            normalized_name=normalized_name,
                            participant_token=participant_token,
                            recording_token=recording_token,
                            target_label=target_label,
                            sha256=sha256,
                            size_bytes=size_bytes,
                            locator_path=locator,
                        )
                    )
            for _chunk in iter(lambda: hashing_reader.read(_CHUNK_SIZE), b""):
                pass
        if (
            hashing_reader.sha256 != fingerprint.sha256
            or hashing_reader.size_bytes != fingerprint.size_bytes
            or _stat_identity(archive_path.stat()) != fingerprint.identity
            or _is_linklike(archive_path)
        ):
            raise PopSignDatasetError("archive.bytes_invalid")
        if not members or uncompressed_size <= 0:
            raise PopSignDatasetError("archive.member_invalid")
        archive_record = ExternalArchiveRecordV1(
            schema_version="external-archive-record/1",
            archive_id=archive_plan.archive_id,
            category=archive_plan.category,
            split=archive_plan.split,
            source_label=archive_plan.source_label,
            local_archive=archive_plan.local_archive,
            sha256=fingerprint.sha256,
            size_bytes=fingerprint.size_bytes,
            member_count=len(members),
            uncompressed_size_bytes=uncompressed_size,
            publisher_checksum_available=False,
            integrity_basis="local_sha256_after_download",
        )
        return archive_record, tuple(members)
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, tarfile.TarError, TypeError, ValueError) as error:
        raise PopSignDatasetError("archive.structure_invalid") from error


def _scan_all_archives(
    plan: ExternalAcquisitionPlanV1,
    archive_root: Path,
    *,
    staging: Path | None,
) -> tuple[tuple[ExternalArchiveRecordV1, ...], tuple[ExternalMediaRecordV1, ...]]:
    selection = load_signlab_five_popsign_selection()
    targets: dict[SourceLabel, ExternalTargetLabel] = {
        mapping.source_label: mapping.target_label_id for mapping in selection.mappings
    }
    archives: list[ExternalArchiveRecordV1] = []
    media: list[ExternalMediaRecordV1] = []
    signer_splits: dict[str, ExternalSplit] = {}
    sample_ids: set[str] = set()
    for archive_plan in plan.archives:
        try:
            target_label = targets[archive_plan.source_label]
        except KeyError as error:
            raise PopSignDatasetError("plan.invalid") from error
        archive_record, captured_media = _scan_archive(
            archive_plan,
            _resolve_archive(archive_root, archive_plan.local_archive),
            target_label=target_label,
            staging=staging,
        )
        archives.append(archive_record)
        for item in captured_media:
            if item.sample_id in sample_ids:
                raise PopSignDatasetError("archive.member_invalid")
            sample_ids.add(item.sample_id)
            prior_split = signer_splits.setdefault(item.participant_id, item.source_split)
            if prior_split != item.source_split:
                raise PopSignDatasetError("split.leakage")
            media.append(item)
    return tuple(archives), tuple(sorted(media, key=lambda item: item.sample_id))


def _build_manifest(
    plan: ExternalAcquisitionPlanV1,
    archives: tuple[ExternalArchiveRecordV1, ...],
    media: tuple[ExternalMediaRecordV1, ...],
) -> ExternalDatasetManifestV1:
    selection = load_signlab_five_popsign_selection()
    payload: dict[str, object] = {
        "schema_version": "external-dataset-manifest/1",
        "dataset_id": POPSIGN_DATASET_ID,
        "version": "1.0.0",
        "source": plan.source.model_dump(mode="json", round_trip=True),
        "selection": plan.selection.model_dump(mode="json", round_trip=True),
        "acquisition_plan_sha256": plan.plan_sha256,
        "taxonomy": selection.taxonomy.model_dump(mode="json", round_trip=True),
        "license_acknowledgement": ExternalLicenseAcknowledgementV1(
            schema_version="external-license-acknowledgement/1",
            license_id="CC-BY-4.0",
            accepted=True,
            authorization_basis="licensed_public_dataset",
            signlab_participant_consent="not_applicable",
        ).model_dump(mode="json", round_trip=True),
        "contains_identifiable_human_video": True,
        "source_metadata_retained": False,
        "claim_scope": "isolated_predefined_gesture_research_only",
        "archives": [item.model_dump(mode="json", round_trip=True) for item in archives],
        "media": [item.model_dump(mode="json", round_trip=True) for item in media],
    }
    try:
        payload["content_sha256"] = external_dataset_manifest_digest(payload)
        return ExternalDatasetManifestV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except (ExternalDatasetContractError, TypeError, ValueError) as error:
        raise PopSignDatasetError("manifest.invalid") from error


def _require_bundle_root(workspace_root: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(workspace_root)
        if _is_linklike(candidate):
            raise PopSignDatasetError("bundle.inventory_invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise PopSignDatasetError("bundle.inventory_invalid")
        return root
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("bundle.inventory_invalid") from error


def _resolve_bundle_file(root: Path, relative: str) -> Path:
    try:
        candidate = root
        for segment in relative.split("/"):
            candidate = candidate / segment
            if _is_linklike(candidate):
                raise PopSignDatasetError("media.bytes_invalid")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise PopSignDatasetError("media.bytes_invalid")
        return resolved
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("media.bytes_invalid") from error


def _bundle_inventory(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        for directory, directory_names, filenames in os.walk(root, topdown=True):
            current = Path(directory)
            for name in directory_names:
                child = current / name
                if _is_linklike(child):
                    raise PopSignDatasetError("bundle.inventory_invalid")
            for name in filenames:
                child = current / name
                if _is_linklike(child) or not child.is_file():
                    raise PopSignDatasetError("bundle.inventory_invalid")
                files.add(child.relative_to(root).as_posix())
        return files
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("bundle.inventory_invalid") from error


def _validate_manifest_bindings(manifest: ExternalDatasetManifestV1) -> None:
    try:
        plan = build_popsign_v1_plan()
        selection = load_signlab_five_popsign_selection()
        if (
            manifest.dataset_id != POPSIGN_DATASET_ID
            or manifest.version != "1.0.0"
            or manifest.source != plan.source
            or manifest.selection != plan.selection
            or manifest.acquisition_plan_sha256 != plan.plan_sha256
            or manifest.taxonomy != selection.taxonomy
            or tuple(item.archive_id for item in manifest.archives)
            != tuple(item.archive_id for item in plan.archives)
        ):
            raise PopSignDatasetError("manifest.invalid")
        planned = {item.archive_id: item for item in plan.archives}
        targets = {item.source_label: item.target_label_id for item in selection.mappings}
        for archive in manifest.archives:
            expected = planned[archive.archive_id]
            if (
                archive.category,
                archive.split,
                archive.source_label,
                archive.local_archive,
            ) != (
                expected.category,
                expected.split,
                expected.source_label,
                expected.local_archive,
            ):
                raise PopSignDatasetError("manifest.invalid")
        if any(targets.get(item.source_label) != item.target_label_id for item in manifest.media):
            raise PopSignDatasetError("manifest.invalid")
    except PopSignDatasetError:
        raise
    except (ExternalDatasetResourceError, KeyError, TypeError, ValueError) as error:
        raise PopSignDatasetError("manifest.invalid") from error


def validate_external_dataset_bundle(
    manifest: ExternalDatasetManifestInput,
    workspace_root: str | os.PathLike[str],
    *,
    archive_root: str | os.PathLike[str] | None = None,
) -> ExternalDatasetValidationResult:
    """Verify exact manifest/media bytes and, optionally, all local source archives."""

    _bounded_document(manifest, _MAX_MANIFEST_BYTES, "manifest.invalid")
    try:
        checked = validate_external_dataset_manifest(manifest)
    except (ExternalDatasetContractError, TypeError, ValueError) as error:
        raise PopSignDatasetError("manifest.invalid") from error
    _validate_manifest_bindings(checked)
    root = _require_bundle_root(workspace_root)
    expected_manifest_bytes = render_external_dataset_json(checked).encode("utf-8")
    try:
        manifest_path = root / EXTERNAL_DATASET_MANIFEST_FILENAME
        if (
            _is_linklike(manifest_path)
            or not manifest_path.is_file()
            or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES
            or manifest_path.read_bytes() != expected_manifest_bytes
        ):
            raise PopSignDatasetError("manifest.invalid")
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("manifest.invalid") from error

    expected_files = {EXTERNAL_DATASET_MANIFEST_FILENAME}
    verified_media: dict[str, _FileFingerprint] = {}
    for item in checked.media:
        expected_files.add(item.locator.path)
        fingerprint = verified_media.get(item.locator.path)
        if fingerprint is None:
            fingerprint = _fingerprint_file(
                _resolve_bundle_file(root, item.locator.path),
                maximum_bytes=_MAX_MEMBER_BYTES,
                category="media.bytes_invalid",
            )
            verified_media[item.locator.path] = fingerprint
        if fingerprint.sha256 != item.sha256 or fingerprint.size_bytes != item.size_bytes:
            raise PopSignDatasetError("media.bytes_invalid")
    if _bundle_inventory(root) != expected_files:
        raise PopSignDatasetError("bundle.inventory_invalid")

    archive_integrity: Literal["verified", "not_checked"] = "not_checked"
    if archive_root is not None:
        plan = build_popsign_v1_plan()
        scanned_archives, scanned_media = _scan_all_archives(
            plan,
            _require_archive_root(archive_root),
            staging=None,
        )
        if scanned_archives != checked.archives or scanned_media != checked.media:
            raise PopSignDatasetError("archive.bytes_invalid")
        archive_integrity = "verified"
    return ExternalDatasetValidationResult(
        content_sha256=checked.content_sha256,
        archive_count=len(checked.archives),
        media_count=len(checked.media),
        semantic_integrity="verified",
        media_byte_integrity="verified",
        archive_byte_integrity=archive_integrity,
        license_authorization="verified",
    )


def _write_manifest_last(staging: Path, manifest: ExternalDatasetManifestV1) -> None:
    content = render_external_dataset_json(manifest).encode("utf-8")
    try:
        path = staging / EXTERNAL_DATASET_MANIFEST_FILENAME
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("publication.failed") from error


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _load_existing_manifest(destination: Path) -> ExternalDatasetManifestV1:
    try:
        manifest_path = destination / EXTERNAL_DATASET_MANIFEST_FILENAME
        if _is_linklike(manifest_path) or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise PopSignDatasetError("destination.conflict")
        return validate_external_dataset_manifest(manifest_path.read_bytes())
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("destination.conflict") from error


def _existing_is_identical(
    destination: Path,
    manifest: ExternalDatasetManifestV1,
) -> ExternalDatasetValidationResult:
    existing = _load_existing_manifest(destination)
    if existing != manifest:
        raise PopSignDatasetError("destination.conflict")
    try:
        return validate_external_dataset_bundle(existing, destination)
    except PopSignDatasetError as error:
        raise PopSignDatasetError("destination.conflict") from error


def _publish_or_reconcile(
    staging: Path,
    destination: Path,
    manifest: ExternalDatasetManifestV1,
) -> tuple[Literal["published", "unchanged"], ExternalDatasetValidationResult]:
    try:
        if _is_linklike(destination) or (destination.exists() and not destination.is_dir()):
            raise PopSignDatasetError("destination.invalid")
        if destination.exists() and any(destination.iterdir()):
            return "unchanged", _existing_is_identical(destination, manifest)
        removed_empty = False
        if destination.exists():
            destination.rmdir()
            removed_empty = True
        try:
            staging.rename(destination)
        except OSError:
            competing_destination = destination.exists()
            if removed_empty and not competing_destination:
                with suppress(OSError):
                    destination.mkdir()
            if competing_destination:
                return "unchanged", _existing_is_identical(destination, manifest)
            raise
        with suppress(OSError):
            _fsync_directory(destination.parent)
        return "published", validate_external_dataset_bundle(manifest, destination)
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise PopSignDatasetError("publication.failed") from error


def _require_destination(
    destination: str | os.PathLike[str],
    archive_root: Path,
) -> Path:
    try:
        output = Path(destination)
        if not output.name or _is_linklike(output):
            raise PopSignDatasetError("destination.invalid")
        output.parent.mkdir(parents=True, exist_ok=True)
        if _is_linklike(output.parent) or not output.parent.is_dir():
            raise PopSignDatasetError("destination.invalid")
        resolved_output = output.parent.resolve(strict=True) / output.name
        if archive_root.is_relative_to(resolved_output) or resolved_output.is_relative_to(
            archive_root
        ):
            raise PopSignDatasetError("destination.invalid")
        if output.exists() and not output.is_dir():
            raise PopSignDatasetError("destination.invalid")
        return output
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("destination.invalid") from error


def import_popsign_v1_archives(
    plan: ExternalAcquisitionPlanInput,
    *,
    archive_root: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    accept_license: str,
) -> ImportedExternalDatasetBundle:
    """Import already-downloaded PopSign archives into one atomic licensed bundle."""

    checked_plan = _validate_exact_plan(plan)
    if accept_license != POPSIGN_LICENSE_ACKNOWLEDGEMENT:
        raise PopSignDatasetError("license.denied")
    source_root = _require_archive_root(archive_root)
    output = _require_destination(destination, source_root)
    try:
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            archives, media = _scan_all_archives(
                checked_plan,
                source_root,
                staging=staging,
            )
            manifest = _build_manifest(checked_plan, archives, media)
            _write_manifest_last(staging, manifest)
            validation = validate_external_dataset_bundle(manifest, staging)
            _fsync_tree(staging)
            status, validation = _publish_or_reconcile(staging, output, manifest)
    except PopSignDatasetError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PopSignDatasetError("publication.failed") from error
    return ImportedExternalDatasetBundle(
        status=status,
        manifest=manifest,
        validation=replace(validation, archive_byte_integrity="verified"),
    )


__all__ = [
    "EXTERNAL_DATASET_MANIFEST_FILENAME",
    "POPSIGN_DATASET_ID",
    "POPSIGN_LICENSE_ACKNOWLEDGEMENT",
    "POPSIGN_PLAN_ID",
    "ExternalAcquisitionPlanWriteResult",
    "ExternalDatasetValidationResult",
    "ImportedExternalDatasetBundle",
    "PopSignDatasetError",
    "build_popsign_v1_plan",
    "import_popsign_v1_archives",
    "validate_external_dataset_bundle",
    "write_external_acquisition_plan",
]
