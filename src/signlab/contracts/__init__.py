"""Versioned, UI-independent contracts shared across the SignLab pipeline."""

from signlab.contracts.taxonomy import (
    BUILTIN_TAXONOMY_DIGEST,
    GestureTaxonomy,
    TaxonomyContractError,
    TaxonomyRef,
    load_builtin_taxonomy,
    normalize_legacy_label,
    taxonomy_reference,
    validate_packaged_taxonomy_resources,
    validate_taxonomy,
    validate_training_label_counts,
    validate_training_label_map,
    validate_training_taxonomy_binding,
)

__all__ = [
    "BUILTIN_TAXONOMY_DIGEST",
    "GestureTaxonomy",
    "TaxonomyContractError",
    "TaxonomyRef",
    "load_builtin_taxonomy",
    "normalize_legacy_label",
    "taxonomy_reference",
    "validate_packaged_taxonomy_resources",
    "validate_taxonomy",
    "validate_training_label_counts",
    "validate_training_label_map",
    "validate_training_taxonomy_binding",
]
