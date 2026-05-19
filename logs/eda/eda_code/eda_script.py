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

print("SECTION 14 — FOLLOW-UP")

# Load necessary CSVs
train = _load_csv(TRAIN_CSV)
taxonomy = _load_csv(TAXONOMY_CSV)
train_soundscapes_labels = _load_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)

# Check for missing or empty files
if not train:
    print("train.csv is empty or missing")
if not taxonomy:
    print("taxonomy.csv is empty or missing")
if not train_soundscapes_labels:
    print("train_soundscapes_labels.csv is empty or missing")

# Follow-up analysis on the data
# Example: Check for any discrepancies between train and taxonomy CSVs
discrepancies = []
for row in train:
    primary_label = row['primary_label']
    if primary_label not in [t['primary_label'] for t in taxonomy]:
        discrepancies.append(primary_label)

if discrepancies:
    print("Discrepancies found in train.csv compared to taxonomy.csv:")
    for d in discrepancies:
        print(d)
else:
    print("No discrepancies found between train.csv and taxonomy.csv")

# Example: Check for any missing primary_labels in train_soundscapes_labels
missing_primary_labels = []
for row in train_soundscapes_labels:
    primary_label = row['primary_label']
    if primary_label not in [t['primary_label'] for t in train]:
        missing_primary_labels.append(primary_label)

if missing_primary_labels:
    print("Missing primary_labels found in train_soundscapes_labels.csv:")
    for m in missing_primary_labels:
        print(m)
else:
    print("No missing primary_labels found in train_soundscapes_labels.csv")