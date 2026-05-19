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

# Load the train CSV file again in case it was modified
train = _load_csv(TRAIN_CSV)

# Initialize counters for valid and invalid latitude/longitude values
valid_lat_lon_count = 0
invalid_lat_lon_count = 0

# Check if 'latitude' and 'longitude' columns exist in the train data
if 'latitude' in train[0] and 'longitude' in train[0]:
    # Iterate through each row in the train data
    for row in train:
        lat = row['latitude']
        lon = row['longitude']

        # Check if both latitude and longitude are present and within valid ranges
        if lat is not None and lon is not None and -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180:
            valid_lat_lon_count += 1
        else:
            invalid_lat_lon_count += 1

# Print the results
print(f"Valid latitude/longitude values: {valid_lat_lon_count}")
print(f"Invalid latitude/longitude values: {invalid_lat_lon_count}")

# Calculate and print the percentage of valid latitude/longitude values
if len(train) > 0:
    percentage_valid = (valid_lat_lon_count / len(train)) * 100
    print(f"Percentage of valid latitude/longitude values: {percentage_valid:.2f}%")
else:
    print("No data available to calculate percentage.")