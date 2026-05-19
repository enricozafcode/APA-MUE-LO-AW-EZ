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

print("SECTION 6 — TAXONOMY")

# Load taxonomy CSV
taxonomy = _load_csv(TAXONOMY_CSV)

# Extract unique species from train.csv
train = _load_csv(TRAIN_CSV)
unique_species_train = set(row['primary_label'] for row in train if row['primary_label'])

# Extract unique species from taxonomy.csv
unique_species_taxonomy = set(row['primary_label'] for row in taxonomy)

# Find common and unique species
common_species = unique_species_train.intersection(unique_species_taxonomy)
unique_to_train = unique_species_train - common_species
unique_to_taxonomy = unique_species_taxonomy - common_species

# Count occurrences of each class in train.csv
class_counts = Counter(row['class_name'] for row in train if row['primary_label'])

# Print results
print(f"Common species between train and taxonomy: {len(common_species)}")
print(f"Unique to train: {len(unique_to_train)}")
print(f"Unique to taxonomy: {len(unique_to_taxonomy)}")
print("Class distribution in train.csv:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")