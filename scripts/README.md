# Repository scripts

Scripts automate repository maintenance, evidence verification, packaging checks,
and reproducible developer workflows. Pipeline behavior belongs in importable
`src/signlab` services; scripts remain thin adapters and must never embed credentials
or private data locations.

Generated contract resources come only from their authoritative Python models and
synthetic builders. Run `generate_taxonomy_schemas.py`,
`generate_governance_resources.py`, or `generate_contract_resources.py` after the
corresponding source changes, then review and commit the deterministic output.
Dataset table JSON Schemas, examples, and Arrow-schema snapshots come from
`generate_dataset_resources.py`; it deliberately does not commit Parquet bytes.
