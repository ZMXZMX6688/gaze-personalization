#!/usr/bin/env python3
"""Block confirmation runs unless cohort novelty and frozen-spec checks pass."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from validate_confirmation_protocol import validate


def fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def prepare(protocol: dict, selection: dict, cohort: dict) -> dict:
    validate(protocol)
    if not cohort.get("confirmation_id", "").strip():
        raise ValueError("confirmation_id must be non-empty")
    checkpoint_hash = cohort["checkpoint_sha256"]
    if len(checkpoint_hash) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_hash):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hexadecimal characters")
    sids = cohort["subject_ids"]
    if not sids or len(sids) != len(set(sids)):
        raise ValueError("Confirmation subject_ids must be non-empty and unique")
    overlap = sorted(set(sids) & set(selection["subject_ids"]))
    if overlap:
        raise ValueError(f"Candidate-selection subject overlap: {overlap}")
    checkpoint_new = checkpoint_hash not in selection["checkpoint_sha256"]
    device_new = cohort["device_family"] not in selection["device_families"]
    if not (checkpoint_new or device_new):
        raise ValueError("Confirmation requires an unseen checkpoint or device family")
    return {
        "status": "LOCKED_FOR_CONFIRMATION",
        "confirmation_id": cohort["confirmation_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": fingerprint(protocol),
        "selection_manifest_sha256": fingerprint(selection),
        "cohort_sha256": fingerprint(cohort),
        "subject_count": len(sids),
        "checkpoint_is_new": checkpoint_new,
        "device_is_new": device_new,
        "protocol": protocol,
        "cohort": cohort,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    selection = json.loads(args.selection_manifest.read_text())
    cohort = json.loads(args.cohort.read_text())
    lock = prepare(protocol, selection, cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"PASS: confirmation run locked: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
