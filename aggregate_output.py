#!/usr/bin/env python3
"""Concatenate eval_model.py result CSVs into one CSV.

Every *.csv under DIR (searched recursively) is read and stacked; columns are
the union across files, blank where a file lacks one. Three columns are added
from each file's path, which is expected to follow the launchers' layout
<lang>/<model>/<model>_bs<N>_lr<lr>.csv:

    lang    name of the file's grandparent directory
    model   name of the file's parent directory
    source  path of the file relative to DIR

Example:
    python aggregate_output.py output/model_evaluations -o aggregated_results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def read_results(path: Path, root: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="warn")
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file {path}", file=sys.stderr)
        return None
    if "source" in df.columns:
        print(f"Skipping {path}: it is already an aggregate", file=sys.stderr)
        return None
    df.insert(0, "lang", path.parent.parent.name)
    df.insert(1, "model", path.parent.name)
    df.insert(2, "source", str(path.relative_to(root)))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="Directory to search for result CSVs")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV (default: stdout)")
    args = parser.parse_args()

    if not args.dir.is_dir():
        parser.error(f"{args.dir} is not a directory")

    out = args.output.resolve() if args.output else None
    paths = sorted(p for p in args.dir.rglob("*.csv") if p.resolve() != out)
    frames = [df for p in paths if (df := read_results(p, args.dir)) is not None]
    if not frames:
        sys.exit(f"No result CSVs found under {args.dir}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(args.output if args.output else sys.stdout, index=False)
    print(f"Aggregated {len(frames)} files, {len(combined)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
