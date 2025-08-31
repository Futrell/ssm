#!/usr/bin/env python3
import os
import re
import csv

ROOT_DIR = "output/model_evaluations"

# Matches: {model}_bs{batch}_ep{epoch}_lr{lr}.txt
FILENAME_RE = re.compile(
    r"^(?P<model>.+)_bs(?P<batch>\d+)_ep(?P<epoch>\d+)_lr(?P<lr>[0-9.]+)\.txt$"
)

def find_files(root):
    """Yield (lang, model_dir, file_path, meta) for each matching file."""
    for lang in os.listdir(root):
        lang_path = os.path.join(root, lang)
        if not os.path.isdir(lang_path):
            continue
        for model_dir in os.listdir(lang_path):
            model_path = os.path.join(lang_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            for fname in os.listdir(model_path):
                m = FILENAME_RE.match(fname)
                if not m:
                    continue
                fpath = os.path.join(model_path, fname)
                meta = m.groupdict()
                yield lang, model_dir, fpath, {
                    "lang": lang,
                    "model": meta["model"],
                    "batch": int(meta["batch"]),
                    #"epoch": int(meta["epoch"]),
                    "lr": float(meta["lr"]),
                }

def sniff_dialect(path):
    """Sniff CSV dialect; fall back to comma."""
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        # Try tab as a common alternative
        class _TSV(csv.Dialect):
            delimiter = "\t"
            quotechar = '"'
            doublequote = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
            escapechar = None
            skipinitialspace = False
        # crude heuristic: if there are lots of tabs, use TSV; else comma
        if sample.count("\t") > sample.count(","):
            return _TSV()
        return csv.get_dialect("excel")

def collect_all_headers(files, new_cols):
    """One pass to compute union of headers across files + new metadata cols."""
    headers = set(new_cols)
    for _, _, fpath, _meta in files:
        dialect = sniff_dialect(fpath)
        with open(fpath, "r", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames:
                headers.update(reader.fieldnames)
    # Put metadata columns first, then the rest in stable order
    remaining = [h for h in sorted(headers) if h not in new_cols]
    return new_cols + remaining

def main():
    files = list(find_files(ROOT_DIR))
    if not files:
        print("No matching files found.")
        return

    meta_cols = ["lang", "model", "batch", "lr"]
    out_headers = collect_all_headers(files, meta_cols)

    out_path = "aggregated_results.csv"
    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_headers)
        writer.writeheader()

        for lang, model_dir, fpath, meta in files:
            dialect = sniff_dialect(fpath)
            with open(fpath, "r", newline="") as in_f:
                reader = csv.DictReader(in_f, dialect=dialect)
                for row in reader:
                    # Merge row with metadata; fill missing keys with ""
                    merged = {k: "" for k in out_headers}
                    merged.update(row if row is not None else {})
                    merged.update(meta)
                    writer.writerow(merged)

    print(f"Wrote {out_path} with {len(files)} source files.")

if __name__ == "__main__":
    main()
