from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "docs/contracts/web-release-headers.v1.json"


def _contract() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))


def _csp(value: str) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    for raw_directive in value.split(";"):
        parts = raw_directive.strip().split()
        assert parts
        name, *sources = parts
        assert name not in directives
        directives[name] = sources
    return directives


def test_release_header_contract_is_exact_and_restrictive() -> None:
    contract = _contract()

    assert contract["format"] == "signlab-web-release-headers/1"
    assert contract["environment"] == "production"
    assert contract["applies_to"] == "all_html_and_static_asset_responses"
    assert contract["deployment_owner"] == "#57"
    assert contract["headers"] == {
        "Content-Security-Policy": (
            "default-src 'none'; base-uri 'none'; connect-src 'self'; font-src 'none'; "
            "form-action 'none'; frame-ancestors 'none'; img-src 'self' data:; "
            "manifest-src 'self'; media-src 'self' blob:; object-src 'none'; "
            "script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self'"
        ),
        "Cross-Origin-Resource-Policy": "same-origin",
        "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }

    directives = _csp(contract["headers"]["Content-Security-Policy"])
    assert directives == {
        "default-src": ["'none'"],
        "base-uri": ["'none'"],
        "connect-src": ["'self'"],
        "font-src": ["'none'"],
        "form-action": ["'none'"],
        "frame-ancestors": ["'none'"],
        "img-src": ["'self'", "data:"],
        "manifest-src": ["'self'"],
        "media-src": ["'self'", "blob:"],
        "object-src": ["'none'"],
        "script-src": ["'self'", "'wasm-unsafe-eval'"],
        "style-src": ["'self'"],
        "worker-src": ["'self'"],
    }
    sources = [source for values in directives.values() for source in values]
    assert "*" not in sources
    assert "'unsafe-eval'" not in sources
    assert "'unsafe-inline'" not in sources
    assert all(not source.startswith("http") for source in sources)


def test_cross_origin_isolation_decision_matches_runtime() -> None:
    isolation = _contract()["cross_origin_isolation"]
    assert isolation["required"] is False
    assert {"Cross-Origin-Embedder-Policy", "Cross-Origin-Opener-Policy"}.isdisjoint(
        _contract()["headers"]
    )
    assert "numThreads=1" in isolation["reason"]

    runtime = (REPOSITORY_ROOT / "apps/web/src/inference/candidateInferenceSession.ts").read_text(
        encoding="utf-8"
    )
    assert 'import * as ort from "onnxruntime-web/wasm";' in runtime
    assert "ort.env.wasm.numThreads = 1;" in runtime
    assert 'executionProviders: ["wasm"]' in runtime


def test_threat_model_marks_deployment_and_development_boundaries() -> None:
    threat_model = (REPOSITORY_ROOT / "docs/browser-security.md").read_text(encoding="utf-8")
    normalized = " ".join(threat_model.split())

    for statement in (
        "Development may fetch the two exact digest-pinned task files",
        "is not part of the production privacy claim",
        "Story #57 owns copying the reviewed bytes to the release origin",
        "no account, application server, analytics, telemetry",
        "one in flight and the newest waiting frame",
        "Maximum serialized size is 16 MiB",
        "IndexedDB is not encrypted",
    ):
        assert statement in normalized
