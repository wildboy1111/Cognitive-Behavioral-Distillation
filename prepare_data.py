"""Pack aligned trial manifests and generate explicit cross-validation splits."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def pack(args):
    with open(args.manifest, encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"image", "label", "trial"}.issubset(rows[0]):
        raise ValueError(
            "Manifest needs image,label,trial columns and at least one row."
        )
    indices = np.array([int(row["trial"]) for row in rows])
    if len(set(indices)) != len(indices) or indices.min() < 0:
        raise ValueError("trial must contain unique, nonnegative source-row indices.")
    labels = np.array(
        [int(row["label"]) - args.label_offset for row in rows], dtype=np.int64
    )
    if labels.min() < 0:
        raise ValueError("Mapped labels must be nonnegative.")
    with h5py.File(args.signals, "r") as handle:
        eeg, et = handle[args.eeg_key][:], handle[args.et_key][:]
    if eeg.ndim != 3 or et.ndim != 3 or indices.max() >= min(len(eeg), len(et)):
        raise ValueError("Expected EEG [N,C,T], ET [N,T,F] and valid trial indices.")
    eeg, et = eeg[indices].astype(np.float32), et[indices].astype(np.float32)
    if args.transpose_et:
        et = et.transpose(0, 2, 1)
    if not np.isfinite(eeg).all() or not np.isfinite(et).all():
        raise ValueError("Signals contain NaN/Inf; clean the source data first.")
    output = Path(args.output)
    if output.resolve() == Path(args.signals).resolve():
        raise ValueError("Output must differ from the source HDF5 file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as handle:
        handle.create_dataset(
            "images",
            data=[row["image"].replace("\\", "/") for row in rows],
            dtype=h5py.string_dtype(),
        )
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("eeg", data=eeg, compression="gzip")
        handle.create_dataset("et", data=et, compression="gzip")
        if "group" in rows[0]:
            handle.create_dataset(
                "groups", data=[row["group"] for row in rows], dtype=h5py.string_dtype()
            )
    print(f"Saved {len(rows)} aligned trials to {output}")


def split(args):
    with h5py.File(args.data, "r") as handle:
        labels = handle["labels"][:]
        groups = handle["groups"].asstr()[:] if "groups" in handle else None
    indices = np.arange(len(labels))
    folds = []
    if args.test_group is not None:
        if groups is None:
            raise ValueError("--test-group requires a groups dataset.")
        test = indices[groups == args.test_group]
        remaining = sorted(set(groups) - {args.test_group})
        if len(test) == 0 or len(remaining) < 2:
            raise ValueError("Need a nonempty test group and >=2 remaining groups.")
        for group in remaining:
            val = indices[groups == group]
            train = indices[(groups != args.test_group) & (groups != group)]
            folds.append(
                {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
            )
    else:
        # Match the original fixed-test + shuffled KFold protocol.
        pool, test = train_test_split(
            indices, test_size=args.test_size, stratify=labels, random_state=args.seed
        )
        for train, val in KFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(pool):
            folds.append(
                {
                    "train": pool[train].tolist(),
                    "val": pool[val].tolist(),
                    "test": test.tolist(),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "folds": folds}, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(folds)} folds to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser(
        "pack", help="Align preprocessed signals using a CSV manifest"
    )
    packing.add_argument("--signals", required=True)
    packing.add_argument("--eeg-key", required=True)
    packing.add_argument("--et-key", required=True)
    packing.add_argument("--manifest", required=True)
    packing.add_argument("--output", default="data/trials.h5")
    packing.add_argument("--label-offset", type=int, default=0)
    packing.add_argument(
        "--transpose-et",
        action="store_true",
        help="Convert source ET [N,F,T] to [N,T,F]",
    )
    splitting = commands.add_parser(
        "split", help="Fixed test set with validation folds"
    )
    splitting.add_argument("--data", default="data/trials.h5")
    splitting.add_argument("--output", default="data/splits.json")
    splitting.add_argument(
        "--test-group",
        help="Hold out this group; leave one remaining group out per fold",
    )
    splitting.add_argument("--test-size", type=float, default=0.2)
    splitting.add_argument("--folds", type=int, default=5)
    splitting.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    (pack if args.command == "pack" else split)(args)


if __name__ == "__main__":
    main()
