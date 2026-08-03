from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.poomsae_scoring.contracts import ScoringContractError, _UniqueKeyLoader


AUTHORITY_TIERS = {
    "current_official_rule",
    "current_official_technique",
    "historical_official",
    "national_secondary",
    "training_secondary",
    "primary_research",
    "unknown",
}
INTENDED_USES = {
    "score_budget",
    "deduction_semantics",
    "numeric_tolerance",
    "movement_sequence",
    "technique_definition",
    "metric_design",
    "method_research",
}


def load_source_intake(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ScoringContractError(f"source intake manifest not found: {source}")
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ScoringContractError(f"invalid source intake YAML: {exc}") from exc
    return validate_source_intake(payload)


def validate_source_intake(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("source intake manifest must be a mapping")
    data = deepcopy(payload)
    _exact_keys(
        data,
        {
            "schema_version",
            "source_id",
            "status",
            "authority",
            "title",
            "document_date",
            "effective_date",
            "retrieved_at",
            "authority_tier",
            "language",
            "access",
            "local_path",
            "expected_sha256",
            "intended_uses",
            "activation_request",
            "notes",
        },
        "source intake manifest",
    )
    if data["schema_version"] != 1 or data["status"] != "candidate":
        raise ScoringContractError("source intake must remain schema 1 candidate")
    for key in (
        "source_id",
        "authority",
        "title",
        "document_date",
        "effective_date",
        "retrieved_at",
        "language",
        "local_path",
        "notes",
    ):
        _nonempty(data[key], key)
    if any(character.isspace() for character in data["source_id"]):
        raise ScoringContractError("source_id cannot contain whitespace")
    if data["authority_tier"] not in AUTHORITY_TIERS:
        raise ScoringContractError("source intake authority_tier is unsupported")
    if data["access"] not in {"public", "paid", "restricted", "user_supplied"}:
        raise ScoringContractError("source intake access is unsupported")
    expected_sha256 = data["expected_sha256"]
    if expected_sha256 is not None and not _is_sha256(expected_sha256):
        raise ScoringContractError("expected_sha256 must be null or a 64-character hex digest")
    if not isinstance(data["intended_uses"], list) or not data["intended_uses"]:
        raise ScoringContractError("intended_uses must be a non-empty list")
    if len(set(data["intended_uses"])) != len(data["intended_uses"]):
        raise ScoringContractError("intended_uses cannot contain duplicates")
    unsupported = set(data["intended_uses"]) - INTENDED_USES
    if unsupported:
        raise ScoringContractError(f"unsupported intended_uses: {sorted(unsupported)}")
    request = data["activation_request"]
    _exact_keys(
        request,
        {"requested", "requested_claims", "numeric_thresholds_requested"},
        "activation_request",
    )
    if not isinstance(request["requested"], bool) or not isinstance(
        request["numeric_thresholds_requested"], bool
    ):
        raise ScoringContractError("activation_request flags must be boolean")
    if not isinstance(request["requested_claims"], list):
        raise ScoringContractError("requested_claims must be a list")
    for claim in request["requested_claims"]:
        _nonempty(claim, "requested_claim")
    if request["numeric_thresholds_requested"] and "numeric_tolerance" not in data["intended_uses"]:
        raise ScoringContractError(
            "numeric_thresholds_requested requires numeric_tolerance in intended_uses"
        )
    return data


def inspect_source_intake(payload: dict[str, Any], *, workspace_root: str | Path) -> dict[str, Any]:
    data = validate_source_intake(payload)
    root = Path(workspace_root).resolve()
    source_path = Path(data["local_path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    exists = source_path.is_file()
    size = source_path.stat().st_size if exists else None
    digest = _sha256(source_path) if exists else None
    expected = data["expected_sha256"]
    hash_status = (
        "not_provided"
        if expected is None
        else "match"
        if digest == expected.lower()
        else "mismatch"
    )
    pdf_magic = False
    if exists:
        with source_path.open("rb") as handle:
            pdf_magic = handle.read(5) == b"%PDF-"

    blockers: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    if not exists:
        block("source_file_missing", "The declared local source file does not exist.")
    elif not pdf_magic:
        block("source_not_pdf", "The declared source does not have a PDF file signature.")
    if expected is None:
        block("expected_hash_missing", "An expected SHA-256 must be frozen before activation review.")
    elif hash_status == "mismatch":
        block("source_hash_mismatch", "The source file does not match the frozen SHA-256.")
    if data["authority_tier"] == "unknown":
        block("authority_unverified", "Source authority must be classified before activation review.")
    if data["activation_request"]["numeric_thresholds_requested"] and data["authority_tier"] not in {
        "current_official_rule",
        "current_official_technique",
    }:
        block(
            "numeric_threshold_authority_insufficient",
            "Historical, secondary or research sources cannot activate current judging tolerances.",
        )

    ready_for_manual_review = data["activation_request"]["requested"] and not blockers
    return {
        "schema_version": 1,
        "status": "ready_for_manual_activation_review" if ready_for_manual_review else "candidate_blocked",
        "source_id": data["source_id"],
        "source": {
            "path": str(source_path),
            "exists": exists,
            "size_bytes": size,
            "sha256": digest,
            "expected_sha256": expected,
            "hash_status": hash_status,
            "pdf_signature_valid": pdf_magic,
            "authority": data["authority"],
            "authority_tier": data["authority_tier"],
            "document_date": data["document_date"],
            "effective_date": data["effective_date"],
        },
        "intended_uses": data["intended_uses"],
        "activation": {
            "requested": data["activation_request"]["requested"],
            "automatic_activation_allowed": False,
            "ready_for_manual_review": ready_for_manual_review,
            "numeric_thresholds_requested": data["activation_request"][
                "numeric_thresholds_requested"
            ],
            "blockers": blockers,
            "required_next_step": (
                "manual_article_page_and_claim_review"
                if ready_for_manual_review
                else "resolve_blockers_without_activating_rules"
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ScoringContractError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringContractError(f"{label} must be a non-empty string")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )

