# ── AUTO-INJECTED bootstrap (portable — do not edit; works on any machine) ────
import ast
import random
from pathlib import Path
from collections import Counter

_here = Path(__file__).resolve()
PROJECT_ROOT = None
for _cand in [_here.parent] + list(_here.parents):
    if (_cand / "src").is_dir() and (_cand / "data").is_dir():
        PROJECT_ROOT = _cand
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _here.parents[1]

DATA = PROJECT_ROOT / "data"
TRAIN_CSV = DATA / "train.csv"
TAXONOMY_CSV = DATA / "taxonomy.csv"
TRAIN_SOUNDSCAPES_LABELS_CSV = DATA / "train_soundscapes_labels.csv"
SAMPLE_SUBMISSION_CSV = DATA / "sample_submission.csv"
TRAIN_AUDIO_DIR = DATA / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = DATA / "test_soundscapes"

def _load_csv(path):
    """Return list[dict] — NOT a pandas DataFrame."""
    import csv
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def _sep(title=""):
    print("=" * 60)
    if title:
        print(title)
        print("=" * 60)

def _parse_secondary_labels_row(row):
    raw = row.get("secondary_labels", "[]").strip()
    try:
        slist = ast.literal_eval(raw)
        return slist if isinstance(slist, list) else []
    except Exception:
        return []

def _soundscape_species(sc_rows):
    out = set()
    for r in sc_rows:
        for sp in str(r.get("primary_label", "")).split(";"):
            sp = sp.strip()
            if sp:
                out.add(sp)
    return out

def _train_species_counts(train_rows):
    return Counter(r["primary_label"] for r in train_rows if r.get("primary_label"))

def _sample_ogg_files(root, n=20, max_collect=8000):
    if not root.is_dir():
        return []
    found = []
    for p in root.rglob("*.ogg"):
        found.append(p)
        if len(found) >= max_collect:
            break
    if not found:
        return []
    return random.sample(found, min(n, len(found)))

print(f"\n{'='*80}\nSECTION 8 — FOCAL COVERAGE GAP\n{'='*80}")

_sep("SECTION 8 — FOCAL COVERAGE GAP")

sub = _load_csv(SAMPLE_SUBMISSION_CSV)
sub_species = [c for c in sub[0].keys() if c != 'row_id']
species_counts = _train_species_counts(_load_csv(TRAIN_CSV))

# Calculate the number of species with at least one recording in the sample submission
focal_species_count = sum(1 for s in sub_species if s in species_counts)

# Calculate the total number of species in the training data
total_species_count = len(species_counts)

# Calculate the focal coverage gap
if total_species_count > 0:
    focal_coverage_gap = (total_species_count - focal_species_count) / total_species_count * 100
else:
    focal_coverage_gap = None

print(f"Focal species count: {focal_species_count}")
print(f"Total species count: {total_species_count}")
if focal_coverage_gap is not None:
    print(f"Focal coverage gap: {focal_coverage_gap:.2f}%")
else:
    print("No species in the training data.")