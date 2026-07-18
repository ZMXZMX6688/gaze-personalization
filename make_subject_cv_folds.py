#!/usr/bin/env python3
"""Create deterministic subject-disjoint cross-validation folds."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from personalize_from_universal import discover_sids


def make_subject_cv_folds(
    sids: Sequence[str],
    n_folds: int = 5,
    val_size: int = 5,
    seed: int = 42,
) -> List[Dict[str, object]]:
    if n_folds < 2 or n_folds > len(sids):
        raise ValueError("n_folds must be between 2 and the number of SIDs")
    if val_size <= 0 or val_size >= len(sids):
        raise ValueError("val_size must be positive and leave training subjects")
    shuffled = list(sids)
    random.Random(seed).shuffle(shuffled)
    test_folds = [list(values) for values in np.array_split(shuffled, n_folds)]
    folds = []
    for fold_index, test_sids in enumerate(test_folds):
        remaining = [sid for sid in shuffled if sid not in set(test_sids)]
        fold_rng = random.Random(seed + 1009 * (fold_index + 1))
        fold_rng.shuffle(remaining)
        val_sids = remaining[:val_size]
        train_sids = remaining[val_size:]
        if not train_sids:
            raise ValueError("Fold configuration leaves no training subjects")
        folds.append({
            "fold": fold_index,
            "train_sids": train_sids,
            "val_sids": val_sids,
            "test_sids": test_sids,
        })

    covered = [sid for fold in folds for sid in fold["test_sids"]]
    if sorted(covered) != sorted(sids) or len(covered) != len(set(covered)):
        raise AssertionError("Test folds do not cover every SID exactly once")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sids = discover_sids(args.data_dir)
    folds = make_subject_cv_folds(
        sids,
        n_folds=args.folds,
        val_size=args.val_size,
        seed=args.seed,
    )
    payload = {
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "n_folds": args.folds,
        "val_size": args.val_size,
        "subjects": len(sids),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
    for fold in folds:
        print(
            f"fold={fold['fold']} train={len(fold['train_sids'])} "
            f"val={len(fold['val_sids'])} test={len(fold['test_sids'])}"
        )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
