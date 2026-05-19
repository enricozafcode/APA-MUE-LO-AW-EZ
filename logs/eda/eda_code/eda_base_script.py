
from __future__ import annotations
import sys, os, json, ast, random
from pathlib import Path
from collections import Counter

# ── locate project root ──────────────────────────────────────────────────────
_here = Path(__file__).resolve()
_root = None
for _cand in [_here.parent] + list(_here.parents):
    if (_cand / "src").exists() and (_cand / "data").exists():
        _root = _cand
        break
if _root is None:
    _root = _here.parents[1]

DATA = _root / "data"

def _load_csv(path):
    import csv
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def _sep():
    print("=" * 60)

# ─── 1. FILE INVENTORY ───────────────────────────────────────────────────────
_sep()
print("SECTION 1: FILE INVENTORY")
_sep()
for fname in ["train.csv", "taxonomy.csv", "train_soundscapes_labels.csv",
              "sample_submission.csv"]:
    p = DATA / fname
    if p.exists():
        lines = sum(1 for _ in open(p, encoding="utf-8")) - 1  # minus header
        print(f"  {fname}: {lines} rows")
    else:
        print(f"  {fname}: NOT FOUND")

audio_root = DATA / "train_audio"
sc_root    = DATA / "train_soundscapes"
test_root  = DATA / "test_soundscapes"
n_audio_files = sum(1 for _ in audio_root.rglob("*.ogg")) if audio_root.exists() else 0
n_sc_files    = sum(1 for _ in sc_root.rglob("*.ogg"))    if sc_root.exists()    else 0
n_test_files  = sum(1 for _ in test_root.rglob("*.ogg"))  if test_root.exists()  else 0
print(f"  train_audio/*.ogg: {n_audio_files} files")
print(f"  train_soundscapes/*.ogg: {n_sc_files} files")
print(f"  test_soundscapes/*.ogg: {n_test_files} files")

# ─── 2. TRAIN.CSV OVERVIEW ───────────────────────────────────────────────────
_sep()
print("SECTION 2: TRAIN.CSV OVERVIEW")
_sep()
train = _load_csv(DATA / "train.csv")
print(f"  Total rows: {len(train)}")
print(f"  Columns: {list(train[0].keys()) if train else []}")

# Missing values
for col in train[0].keys():
    n_missing = sum(1 for r in train if not r[col].strip())
    if n_missing:
        print(f"  Missing in '{col}': {n_missing} ({n_missing/len(train)*100:.1f}%)")

# ─── 3. SPECIES / CLASS DISTRIBUTION ─────────────────────────────────────────
_sep()
print("SECTION 3: SPECIES & CLASS DISTRIBUTION")
_sep()
species_counts = Counter(r["primary_label"] for r in train)
class_counts   = Counter(r.get("class_name", "unknown") for r in train)

print(f"  Unique species (primary_label): {len(species_counts)}")
print(f"  Taxonomic classes: {dict(class_counts)}")

counts_sorted = sorted(species_counts.values(), reverse=True)
print(f"  Max recordings per species: {counts_sorted[0]}")
print(f"  Min recordings per species: {counts_sorted[-1]}")
print(f"  Median recordings per species: {sorted(counts_sorted)[len(counts_sorted)//2]}")

# Species with very few samples (< 10)
scarce = [sp for sp, c in species_counts.items() if c < 10]
print(f"  Species with < 10 recordings: {len(scarce)}")

# Top 10 most frequent
top10 = species_counts.most_common(10)
print(f"  Top 10 species: {top10}")

# ─── 4. RATING & QUALITY ─────────────────────────────────────────────────────
_sep()
print("SECTION 4: RECORDING QUALITY")
_sep()
ratings = [float(r["rating"]) for r in train if r.get("rating","").strip()]
if ratings:
    zero_rated = sum(1 for x in ratings if x == 0)
    print(f"  Rating range: {min(ratings):.1f} to {max(ratings):.1f}")
    print(f"  Zero-rated (unrated, mostly iNat): {zero_rated} ({zero_rated/len(ratings)*100:.1f}%)")
    collections = Counter(r.get("collection", "?") for r in train)
    print(f"  Collection split: {dict(collections)}")
