#!/usr/bin/env python3
"""
MLRegTest → converted_mlregtest

• LearningData.txt  = Train ∪ Dev (∪ Valid/Validation)
• TestingData.txt   = TestLR ∪ TestSR                    (order-preserving, dedup)
• TestingPairs.tsv  = from TestSA ∪ TestLA ONLY, paired by ORDER:
                      first grammatical with first ungrammatical, etc. (≤1000)

Space-normalizing, order-preserving; no .bak outputs.
"""

import csv, argparse, sys, re, random
from pathlib import Path

# ---------- labels & normalization ----------
POS_LABS = {"g","grammatical","pos","positive","1","true","t","yes","y"}
NEG_LABS = {"u","ungrammatical","neg","negative","0","false","f","no","n"}

def _norm_label(s: str):
    if s is None: return "grammatical"
    s = (s or "").strip().lower()
    if s in POS_LABS: return "grammatical"
    if s in NEG_LABS: return "ungrammatical"
    if s.startswith("gram"): return "grammatical"
    if s.startswith("ungram"): return "ungrammatical"
    return "grammatical"

_WS_COMMA_RE = re.compile(r"[,\s]+")
def _space_normalize(seq: str) -> str:
    if seq is None: return ""
    s = _WS_COMMA_RE.sub(" ", seq.strip())
    s = " ".join(t for t in s.split(" ") if t)
    if " " not in s:
        s = " ".join(list(s))
    return s

def _read_rows(path: Path, sep="\t"):
    rows = []
    if not path or not path.exists(): return rows
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=sep):
            if row and any((cell or "").strip() for cell in row):
                rows.append([(cell or "").strip() for cell in row])
    return rows

def _rows_from_labeled_file(path: Path):
    out = []
    for r in _read_rows(path):
        if not r: continue
        tok = _space_normalize(r[0])
        lab = _norm_label(r[1] if len(r)>1 else None)
        if tok: out.append((tok, lab))
    return out

def _dedup_preserve_order(items):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it); out.append(it)
    return out

def _strip_prefix(name: str) -> str:
    return re.sub(r'^(?:\d+\.)+', '', name)

# ---------- discovery ----------
SUFFIXES = [
    # learning
    "Train","Dev","Development","Valid","Validation","LearningData",
    # tests we use
    "TestLR","TestSR","TestLA","TestSA",
    # legacy/fallbacks
    "TestingData","Test","TestingPairs","TestPairs",
]

def _discover_ids_flat(root: Path):
    ids = set(); suf_pat = "|".join(SUFFIXES)
    for p in root.iterdir():
        if not p.is_file(): continue
        stem = _strip_prefix(p.stem)
        m = re.match(rf'^(?P<id>.+)_(?P<suf>{suf_pat})$', stem, re.IGNORECASE)
        if m: ids.add(m.group('id'))
    return sorted(ids)

def _discover_ids_subdirs(root: Path):
    return [d.name for d in sorted(root.iterdir()) if d.is_dir()]

def _match_in_dir(dir_path: Path, id_core: str):
    suf_pat = "|".join(SUFFIXES)
    out = {k: None for k in SUFFIXES}
    for p in sorted(dir_path.iterdir()):
        if not p.is_file(): continue
        stem = _strip_prefix(p.stem)
        m = re.match(rf'^{re.escape(id_core)}_(?P<suf>{suf_pat})$', stem, re.IGNORECASE)
        if m: out[m.group("suf")] = p
    return out

def _match_in_subdir(dir_path: Path):
    suf_pat = "|".join(SUFFIXES)
    out = {k: None for k in SUFFIXES}
    for p in sorted(dir_path.iterdir()):
        if not p.is_file(): continue
        stem = _strip_prefix(p.stem)
        m = re.match(rf'^(?P<id>.+)_(?P<suf>{suf_pat})$', stem, re.IGNORECASE)
        if m: out[m.group("suf")] = p
    return out

# ---------- builders ----------
def _collect_learning_rows(cluster):
    train_p = cluster.get("Train")
    dev_p   = cluster.get("Dev") or cluster.get("Development")
    val_p   = cluster.get("Valid") or cluster.get("Validation")
    legacy  = cluster.get("LearningData")

    rows = []
    for p in [train_p, dev_p, val_p]:
        if p and p.exists(): rows.extend(_rows_from_labeled_file(p))
    rows = _dedup_preserve_order(rows)

    if not rows and legacy and legacy.exists():
        rows = _dedup_preserve_order(_rows_from_labeled_file(legacy))

    if not rows:
        # last resort: labeled rows from any legacy testing/pairs
        for key in ["TestingData","Test"]:
            p = cluster.get(key)
            if p and p.exists(): rows.extend(_rows_from_labeled_file(p))
        pairs_p = cluster.get("TestingPairs") or cluster.get("TestPairs")
        if pairs_p and pairs_p.exists():
            for r in _read_rows(pairs_p):
                if len(r) >= 2:
                    g = _space_normalize(r[0]); u = _space_normalize(r[1])
                    if g: rows.append((g,"grammatical"))
                    if u: rows.append((u,"ungrammatical"))
        rows = _dedup_preserve_order(rows)
    return rows

def _collect_testing_rows_LR_SR(cluster, learning_rows):
    """TestingData = TestLR ∪ TestSR (NO SA/LA)."""
    rows = []
    any_present = False
    for key in ["TestLR","TestSR"]:
        p = cluster.get(key)
        if p and p.exists():
            rows.extend(_rows_from_labeled_file(p))
            any_present = True
    if any_present:
        return _dedup_preserve_order(rows)

    # legacy fallback
    for key in ["TestingData","Test"]:
        p = cluster.get(key)
        if p and p.exists():
            return _dedup_preserve_order(_rows_from_labeled_file(p))

    # final fallback: small stratified sample from learning
    G = [(t,l) for (t,l) in learning_rows if l=="grammatical"]
    U = [(t,l) for (t,l) in learning_rows if l=="ungrammatical"]
    g_take = max(1, int(0.2*len(G))) if G else 0
    u_take = max(1, int(0.2*len(U))) if U else 0
    random.seed(0)
    out = []
    if g_take: out += random.sample(G, min(len(G), g_take))
    if u_take: out += random.sample(U, min(len(U), u_take))
    return _dedup_preserve_order(out)

# ---- ordered pairing from SA ∪ LA (NO heuristics, just position) ----
def _build_pairs_from_SA_LA_ordered(cluster, max_pairs=1000):
    """
    Build TestingPairs from TestSA ∪ TestLA ONLY by ORDER:
      collect all grammatical (in file order), collect all ungrammatical (in file order),
      then pair i-th G with i-th U, truncate to ≤ max_pairs.
    """
    rows = []
    for key in ["TestSA","TestLA"]:
        p = cluster.get(key)
        if p and p.exists():
            rows.extend(_rows_from_labeled_file(p))
    if not rows:
        return []

    # Preserve order exactly as read; DO NOT dedup here (order pairing).
    G = [t for (t,l) in rows if l == "grammatical"]
    U = [t for (t,l) in rows if l == "ungrammatical"]
    n = min(len(G), len(U), max_pairs)
    return [(G[i], U[i]) for i in range(n)]

# ---------- writers ----------
def _write_labeled_txt(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for tokens, lab in rows:
            w.writerow([tokens, lab])

def _write_pairs_tsv(path: Path, pairs_rows):
    """Overwrite pairs TSV with header 'grammatical<TAB>ungrammatical'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        # add header
        w.writerow(["grammatical", "ungrammatical"])
        for g, u in pairs_rows:
            w.writerow([_space_normalize(g), _space_normalize(u)])

# ---------- driver ----------
def _convert_one(cluster, did, out_root: Path):
    learning = _collect_learning_rows(cluster)
    if not learning:
        print(f"⚠️  Skip {did}: no LearningData (Train/Dev/Valid or fallbacks)"); return

    testing = _collect_testing_rows_LR_SR(cluster, learning)
    pairs   = _build_pairs_from_SA_LA_ordered(cluster, max_pairs=1000)

    out_dir = out_root / did
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_labeled_txt(out_dir / f"{did}LearningData.txt", _dedup_preserve_order(learning))
    _write_labeled_txt(out_dir / f"{did}TestingData.txt",  _dedup_preserve_order(testing))
    if pairs:
        _write_pairs_tsv(out_dir / f"{did}TestingPairs.tsv", pairs)
    print(f"✅ Wrote: {did}/{{LearningData,TestingData,TestingPairs}}")

def convert(root: Path, out_root: Path, filt: str = None):
    flat_ids = _discover_ids_flat(root)
    mode = "flat" if flat_ids else "subdir"
    if mode == "flat":
        ids = flat_ids
        if filt: ids = [i for i in ids if filt.lower() in i.lower()]
        print(f"📁 Flat layout: {len(ids)} dataset id(s) to convert")
        for did in ids:
            _convert_one(_match_in_dir(root, did), did, out_root)
    else:
        ids = _discover_ids_subdirs(root)
        if filt: ids = [i for i in ids if filt.lower() in i.lower()]
        print(f"📂 Subdirectory layout: {len(ids)} dataset folder(s) to convert")
        for did in ids:
            _convert_one(_match_in_subdir(root / did), did, out_root)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mlregtest",
                    help="Root with flat files or <ID>/ subfolders")
    ap.add_argument("--out", default="data/converted_mlregtest",
                    help="Output root")
    ap.add_argument("--filter", default=None, help="Substring filter on dataset ID")
    args = ap.parse_args()

    root = Path(args.root); out_root = Path(args.out)
    if not root.exists():
        print(f"❌ Not found: {root}"); sys.exit(1)
    convert(root, out_root, filt=args.filter)

if __name__ == "__main__":
    main()
