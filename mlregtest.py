#!/usr/bin/env python3
# evaluate_models.py  – multi-split version

import argparse, csv, os, re, subprocess, sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  CONFIG                                                                     #
# --------------------------------------------------------------------------- #
MLREG_DIR   = Path("data/mlregtest")
OUT_DIR     = Path("output/model_evaluations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DEFAULT_MODELS = ["ptsl2", "ssm", "pfsa", "wfsa", "sl2", "sp2", "soft_tsl2"]
DEFAULT_MODELS = ["sl2"]

GRID = { "batch_size": [32], "num_epochs": [10], "lr": [1e-3] }

TEST_SUFS = {                     # suffix
    "_TestLR.txt": "LR",
    "_TestLA.txt": "LA",
    "_TestSA.txt": "SA",
    "_TestSR.txt": "SR",
}

LOSS_RE = re.compile(r"(?:Loss:|loss=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
SUMMARY_CSV = OUT_DIR / "summary.csv"

# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
def iter_pairs():
    """Yield (train, test, dataset_tag, split_code)."""
    for train in sorted(MLREG_DIR.glob("*_Train.txt")):
        tag = train.stem.replace("_Train", "")
        for suf, code in TEST_SUFS.items():
            test = train.with_name(f"{tag}{suf}")
            if test.exists():
                yield train, test, tag, code

def run_one(model, train, test, tag, split, bs, epochs, lr):
    log_file = OUT_DIR / f"{tag}_{split}_{model}_bs{bs}_ep{epochs}_lr{lr}.txt"

    if log_file.exists():                       # skip if already done
        return log_file

    cmd = [
        sys.executable, "eval_model.py",
        model, str(train), str(test),
        "--batch_size", str(bs),
        "--num_epochs", str(epochs),
        "--lr", str(lr),
    ]
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f,
                       stderr=subprocess.STDOUT, check=False)
    return log_file

def extract_loss(path):
    loss = None
    for line in Path(path).read_text().splitlines():
        m = LOSS_RE.search(line)
        if m:
            try:
                loss = float(m.group(1))
            except ValueError:
                pass
    return loss

def append_summary(row):
    first = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first:
            w.writeheader()
        w.writerow(row)

# --------------------------------------------------------------------------- #
#  MAIN                                                                       #
# --------------------------------------------------------------------------- #
def main(args):
    models = args.models.split(",")

    GRID["batch_size"] = [args.batch_size]
    GRID["num_epochs"] = [args.epochs]
    GRID["lr"]         = [args.lr]

    grid_vals   = list(product(*GRID.values()))         # pre-compute grid
    pair_list   = list(iter_pairs())                    # materialise generator
    total_runs  = len(models) * len(grid_vals) * len(pair_list)

    print(f"Train files : {len(list(MLREG_DIR.glob('*_Train.txt')))}")
    print(f"Model types : {models}")
    print(f"Grid size   : {len(grid_vals)}")
    print(f"Splits      : {list(TEST_SUFS.values())}")
    print(f"Total runs  : {total_runs}")

    with tqdm(total=total_runs, desc="Model runs") as bar:
        for train, test, tag, split in pair_list:
            for model in models:
                for bs, ep, lr in grid_vals:

                    bar.set_postfix(
                        ds=f"{tag}/{split}",
                        model=model,
                        bs=bs,
                        ep=ep,
                        lr=lr,
                        refresh=False
                    )

                    log_file = run_one(model, train, test, tag, split,
                                       bs, ep, lr)
                    loss = extract_loss(log_file)
                    bar.set_postfix(loss=f"{loss:.4g}" if loss else "NA",
                                    refresh=False)

                    append_summary({
                        "dataset": tag,
                        "split":   split,
                        "model":   model,
                        "batch":   bs,
                        "epochs":  ep,
                        "lr":      lr,
                        "loss":    loss,
                        "log":     log_file,
                    })

                    bar.update()

    # plot_results()

# --------------------------------------------------------------------------- #
#  PLOT                                                                       #
# --------------------------------------------------------------------------- #
# def plot_results():
#     df = pd.read_csv(SUMMARY_CSV)
#     best = (
#         df.sort_values("loss")
#           .groupby(["dataset", "split", "model"])
#           .first()
#           .reset_index()
#     )

#     fig, ax = plt.subplots(figsize=(11, 4))
#     lbls = []
#     for (_, split), sub in best.groupby(["dataset", "split"]):
#         ax.scatter(sub["model"], sub["loss"],
#                    label=f"{sub.dataset.iloc[0]}-{split}",
#                    s=80, alpha=.8)
#         lbls.append(f"{sub.dataset.iloc[0]}-{split}")

#     ax.set_ylabel("Best loss (lower = better)")
#     ax.set_xlabel("Model class")
#     ax.set_title("Best configuration per model / dataset / split")
#     ax.legend(title="Corpus-Split", bbox_to_anchor=(1.02, 1), loc="upper left")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     fig.savefig(OUT_DIR / "comparison.png", dpi=150)
#     print(f"\nPlot saved to {OUT_DIR/'comparison.png'}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=10)
    ap.add_argument("--lr",         type=float, default=1e-3)
    main(ap.parse_args())


# pre-compute the number of iteration and pass it to tqdm
