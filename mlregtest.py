#!/usr/bin/env python3
# evaluate_models.py
"""
Grid-search over PFA-style models on every *mlregtest* corpus and plot results.

Usage:
  python evaluate_models.py                       # uses defaults
  python evaluate_models.py --models sl2,pfsa     # pick a subset
  python evaluate_models.py --epochs 300          # different hyper-params
"""

import argparse, csv, glob, os, re, subprocess, sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  CONFIG                                                                     #
# --------------------------------------------------------------------------- #
MLREG_DIR   = Path("data/mlregtest")
OUT_DIR     = Path("output/model_evaluations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = ["ptsl2", "ssm", "pfsa", "wfsa", "sl2", "sp2", "soft_tsl2"]

GRID = {                    # feel free to add more values from CLI
    "batch_size": [32],
    "num_epochs": [100],
    "lr": [1e-3],
}
# Regex to capture a *float* immediately after “Loss:”  or “loss=”
LOSS_RE = re.compile(r"(?:Loss:|loss=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

SUMMARY_CSV = OUT_DIR / "summary.csv"

# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
def iter_datasets():
    """Yield (train_file, test_file, tag) tuples for every mlregtest corpus."""
    train_files = sorted(MLREG_DIR.glob("*_Train.txt"))
    for train in train_files:
        tag = train.stem.replace("_Train", "")
        test = train.with_name(f"{tag}_TestLR.txt")
        if test.exists():
            yield train, test, tag
        else:
            print(f"[!] Warning: no test file for {tag}", file=sys.stderr)


def run_one(model, train, test, bs, epochs, lr):
    tag = train.stem.replace("_Train", "")
    out_path = OUT_DIR / f"{tag}_{model}_bs{bs}_ep{epochs}_lr{lr}.txt"

    cmd = [
        sys.executable, "eval_model.py",
        model, str(train), str(test),
        "--batch_size", str(bs),
        "--num_epochs", str(epochs),
        "--lr", str(lr),
    ]
    if out_path.exists():
        return out_path  # skip if already done
    with open(out_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    return out_path


def extract_loss(log_file):
    loss = None
    for line in Path(log_file).read_text().splitlines():
        m = LOSS_RE.search(line)
        if m:
            try:
                loss = float(m.group(1))
            except ValueError:
                pass
    return loss


def append_summary(rowdict):
    first = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rowdict.keys())
        if first:
            w.writeheader()
        w.writerow(rowdict)


# --------------------------------------------------------------------------- #
#  MAIN                                                                       #
# --------------------------------------------------------------------------- #
def main(args):
    models = args.models.split(",")
    GRID["batch_size"]  = [args.batch_size]
    GRID["num_epochs"]  = [args.epochs]
    GRID["lr"]          = [args.lr]

    print(f"Datasets  : {sum(1 for _ in iter_datasets())}")
    print(f"Models    : {models}")
    print(f"Grid size : {np.prod([len(v) for v in GRID.values()])}")

    for train, test, tag in iter_datasets():
        for model, (bs, ep, lr) in product(models, product(*GRID.values())):
            print(f"\n[{tag}] {model}  bs={bs}  ep={ep}  lr={lr}")
            log_file = run_one(model, train, test, bs, ep, lr)
            loss = extract_loss(log_file)
            print(f"  ↳ final loss = {loss}")
            append_summary({
                "dataset": tag,
                "model": model,
                "batch": bs,
                "epochs": ep,
                "lr": lr,
                "loss": loss,
                "log": log_file,
            })

    plot_results()


# --------------------------------------------------------------------------- #
#  PLOTTING                                                                   #
# --------------------------------------------------------------------------- #
def plot_results():
    import pandas as pd

    df = pd.read_csv(SUMMARY_CSV)
    best = df.sort_values("loss").groupby(["dataset", "model"]).first().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    for ds in best["dataset"].unique():
        sub = best[best.dataset == ds]
        ax.scatter(sub["model"], sub["loss"], label=ds, s=80, alpha=.8)

    ax.set_ylabel("Best loss (lower = better)")
    ax.set_xlabel("Model class")
    ax.set_title("Best configuration per model / dataset")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=150)
    print(f"\nPlot saved to {OUT_DIR/'comparison.png'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help=f"comma-separated subset (default: {','.join(DEFAULT_MODELS)})")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()
    main(args)
