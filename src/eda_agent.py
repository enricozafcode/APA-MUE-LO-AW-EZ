"""
EDA phase for the BirdCLEF autonomous agent.

Pipeline:
  1. Run hardcoded ``EDA_BASE_SCRIPT`` (sections 1–4) for a safe, working start
  2. Up to 10 LLM exploration iterations (5 fix attempts each), feeding cumulative stdout back
  3. Save combined output → ``eda_raw_output.txt``
  4. LLM summary → ``eda_summary.txt``; optional ``eda_brief.txt``

If the base script fails, fall back to the full hardcoded ``EDA_SCRIPT``.

Each exploration iteration gets a portable path bootstrap auto-prepended (``src/`` + ``data/``).
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# EDA SCRIPT (executed in subprocess via CodeExecutor)
# ─────────────────────────────────────────────────────────────────────────────

EDA_SCRIPT = '''
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

# ─── 5. MULTI-LABEL (SECONDARY LABELS) ───────────────────────────────────────
_sep()
print("SECTION 5: MULTI-LABEL STRUCTURE")
_sep()
has_secondary = 0
total_secondary = 0
for r in train:
    raw = r.get("secondary_labels", "[]").strip()
    try:
        slist = ast.literal_eval(raw)
    except Exception:
        slist = []
    if slist:
        has_secondary += 1
        total_secondary += len(slist)
print(f"  Recordings with secondary labels: {has_secondary} ({has_secondary/len(train)*100:.1f}%)")
print(f"  Total secondary label entries: {total_secondary}")
print(f"  Avg secondary labels (when present): {total_secondary/has_secondary:.2f}" if has_secondary else "")

# ─── 6. TAXONOMY MAPPING ─────────────────────────────────────────────────────
_sep()
print("SECTION 6: TAXONOMY / SPECIES MAPPING")
_sep()
tax_path = DATA / "taxonomy.csv"
if tax_path.exists():
    taxonomy = _load_csv(tax_path)
    print(f"  Taxonomy rows: {len(taxonomy)}")
    print(f"  Taxonomy columns: {list(taxonomy[0].keys()) if taxonomy else []}")
    tax_cols = list(taxonomy[0].keys()) if taxonomy else []
    if "order" in tax_cols:
        orders = Counter(r.get("order", "?") for r in taxonomy)
        print(f"  Unique orders: {len(orders)}")
    else:
        print("  'order' column not in taxonomy - skipped")
    if "family" in tax_cols:
        families = Counter(r.get("family", "?") for r in taxonomy)
        print(f"  Unique families: {len(families)}")
    else:
        print("  'family' column not in taxonomy - skipped")
    classes_in_tax = Counter(r.get("class_name", "?") for r in taxonomy)
    print(f"  Taxonomic class breakdown: {dict(classes_in_tax)}")
    # Check coverage: are all primary_labels in taxonomy?
    tax_species = {r.get("species_code", r.get("primary_label","")) for r in taxonomy}
    train_species = set(species_counts.keys())
    not_in_tax = train_species - tax_species
    if not_in_tax:
        print(f"  Train species NOT in taxonomy: {len(not_in_tax)} -> {list(not_in_tax)[:5]}")
    else:
        print(f"  All train species found in taxonomy.")
else:
    print("  taxonomy.csv not found")

# ─── 7. TRAIN SOUNDSCAPES ─────────────────────────────────────────────────────
_sep()
print("SECTION 7: TRAIN SOUNDSCAPES")
_sep()
sc_labels_path = DATA / "train_soundscapes_labels.csv"
if sc_labels_path.exists():
    sc_labels = _load_csv(sc_labels_path)
    print(f"  Soundscape label rows: {len(sc_labels)}")
    print(f"  Columns: {list(sc_labels[0].keys()) if sc_labels else []}")

    # Species in soundscapes
    sc_species = set()
    for r in sc_labels:
        raw = r.get("primary_label", "")
        for sp in str(raw).split(";"):
            sp = sp.strip()
            if sp:
                sc_species.add(sp)
    print(f"  Unique species in soundscapes: {len(sc_species)}")

    # Overlap with train_audio
    train_audio_species = set(species_counts.keys())
    only_in_sc   = sc_species - train_audio_species
    only_in_audio = train_audio_species - sc_species
    both = sc_species & train_audio_species
    print(f"  Species in BOTH train_audio and soundscapes: {len(both)}")
    print(f"  Species ONLY in soundscapes (no audio clip): {len(only_in_sc)} -> {list(only_in_sc)[:10]}")
    print(f"  Species ONLY in train_audio (not in soundscapes): {len(only_in_audio)}")
else:
    print("  train_soundscapes_labels.csv not found")

# ─── 8. FOCAL COVERAGE GAP ───────────────────────────────────────────────────
_sep()
print("SECTION 8: FOCAL COVERAGE GAP (submission species vs train_audio)")
_sep()
sub_path = DATA / "sample_submission.csv"
if sub_path.exists():
    sub = _load_csv(sub_path)
    # Submission columns = row_id + one col per species
    sub_species = [c for c in sub[0].keys() if c != "row_id"]
    print(f"  Species required in submission: {len(sub_species)}")
    print(f"  Species in train_audio: {len(species_counts)}")
    missing_from_train = set(sub_species) - set(species_counts.keys())
    print(f"  Submission species with NO train_audio examples: {len(missing_from_train)}")
    print(f"  -> {sorted(missing_from_train)[:15]} ...")
else:
    print("  sample_submission.csv not found")

# ─── 9. AUDIO SAMPLE CHARACTERISTICS ─────────────────────────────────────────
_sep()
print("SECTION 9: AUDIO SAMPLE CHARACTERISTICS")
_sep()
try:
    import librosa
    audio_files = list(audio_root.rglob("*.ogg")) if audio_root.exists() else []
    sample_files = random.sample(audio_files, min(30, len(audio_files)))
    durations, sample_rates = [], []
    for f in sample_files:
        try:
            info = librosa.core.audio.__audioread_load if False else None
            dur = librosa.get_duration(path=str(f))
            sr  = librosa.get_samplerate(str(f))
            durations.append(dur)
            sample_rates.append(sr)
        except Exception:
            pass
    if durations:
        print(f"  Sample size: {len(durations)} files")
        print(f"  Duration range: {min(durations):.1f}s to {max(durations):.1f}s")
        print(f"  Median duration: {sorted(durations)[len(durations)//2]:.1f}s")
        print(f"  Unique sample rates: {sorted(set(sample_rates))}")
except ImportError:
    print("  librosa not available — skipping audio sample analysis")

_sep()
print("EDA COMPLETE")
_sep()
'''

# First four sections only — always run before LLM exploration (portable paths via __file__)
EDA_BASE_SCRIPT = EDA_SCRIPT.split("# ─── 5. MULTI-LABEL")[0].rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Portable path bootstrap (auto-prepended to every LLM-generated EDA script)
# Same discovery as EDA_SCRIPT: walk up from __file__ until src/ + data/ exist.
# ─────────────────────────────────────────────────────────────────────────────

EDA_CODEGEN_BOOTSTRAP = '''\
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
'''

# Names the LLM must use (never absolute paths or guessed Desktop/data/…)
EDA_PATH_CONSTANTS_DOC = """\
DATA LOADING: `train = _load_csv(TRAIN_CSV)` → list of dicts (NOT pandas — no .empty, .dropna, row['col'] on DataFrame).

Paths: `TRAIN_CSV`, `TAXONOMY_CSV`, `TRAIN_SOUNDSCAPES_LABELS_CSV`, `SAMPLE_SUBMISSION_CSV`,
`TRAIN_AUDIO_DIR`, `TRAIN_SOUNDSCAPES_DIR`, `TEST_SOUNDSCAPES_DIR`, `DATA`, `PROJECT_ROOT`.

Helpers (already defined — use these): `_load_csv`, `_sep`, `_parse_secondary_labels_row`,
`_soundscape_species`, `_train_species_counts`, `_sample_ogg_files`.

Imports already available: ast, random, Counter, Path."""

_BOOTSTRAP_STRIP_MARKERS = (
    "PROJECT_ROOT",
    "DATA =",
    "Path(__file__)",
    "_here = Path",
    "def _load_csv",
    "def _sep",
    "def _parse_secondary",
    "def _soundscape_species",
    "def _train_species_counts",
    "def _sample_ogg_files",
    "from pathlib",
    "import pathlib",
    "import ast",
    "from collections import Counter",
    "AUTO-INJECTED",
    "REQUIRED bootstrap",
    "TRAIN_CSV =",
    "TAXONOMY_CSV =",
    "TRAIN_AUDIO_DIR =",
    "TRAIN_SOUNDSCAPES_DIR =",
    "TEST_SOUNDSCAPES_DIR =",
)

_FORBIDDEN_IN_ANALYSIS = ("import pandas", "pd.", "as pd", "DataFrame", ".empty", ".dropna(")


def discover_project_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (or this file) until ``src/`` and ``data/`` exist."""
    here = Path(start or __file__).resolve()
    for cand in [here.parent] + list(here.parents):
        if (cand / "src").is_dir() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


def _csv_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return None


def _csv_columns(path: Path, max_cols: int = 12) -> list[str]:
    if not path.is_file():
        return []
    try:
        import csv

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
        return cols[:max_cols] + (["…"] if len(cols) > max_cols else [])
    except OSError:
        return []


def _count_ogg_files(root: Path, *, max_scan: int = 50_000) -> int | None:
    """Count ``*.ogg`` under ``root``; return None if scan cap exceeded."""
    if not root.is_dir():
        return None
    n = 0
    for _ in root.rglob("*.ogg"):
        n += 1
        if n >= max_scan:
            return None
    return n


def gather_eda_preflight(root: Path | None = None) -> dict:
    """
    Lightweight filesystem inventory for LLM prompts (no heavy audio loading).
    """
    project_root = discover_project_root(root)
    data = project_root / "data"
    csv_names = [
        "train.csv",
        "taxonomy.csv",
        "train_soundscapes_labels.csv",
        "sample_submission.csv",
    ]
    csv_files: dict[str, dict] = {}
    for name in csv_names:
        p = data / name
        csv_files[name] = {
            "rel_path": f"data/{name}",
            "exists": p.is_file(),
            "rows": _csv_row_count(p),
            "columns": _csv_columns(p),
        }

    train_audio = data / "train_audio"
    train_soundscapes = data / "train_soundscapes"
    test_soundscapes = data / "test_soundscapes"
    n_species_dirs = (
        sum(1 for p in train_audio.iterdir() if p.is_dir()) if train_audio.is_dir() else None
    )

    return {
        "project_root": str(project_root),
        "data_dir": str(data),
        "csv_files": csv_files,
        "dirs": {
            "train_audio": {
                "rel_path": "data/train_audio",
                "exists": train_audio.is_dir(),
                "species_subdirs": n_species_dirs,
                "ogg_count": _count_ogg_files(train_audio),
            },
            "train_soundscapes": {
                "rel_path": "data/train_soundscapes",
                "exists": train_soundscapes.is_dir(),
                "ogg_count": _count_ogg_files(train_soundscapes),
            },
            "test_soundscapes": {
                "rel_path": "data/test_soundscapes",
                "exists": test_soundscapes.is_dir(),
                "ogg_count": _count_ogg_files(test_soundscapes),
            },
        },
    }


def format_eda_preflight_block(preflight: dict) -> str:
    """Human-readable dataset inventory for codegen (relative paths only — portable)."""
    _csv_var = {
        "train.csv": "TRAIN_CSV",
        "taxonomy.csv": "TAXONOMY_CSV",
        "train_soundscapes_labels.csv": "TRAIN_SOUNDSCAPES_LABELS_CSV",
        "sample_submission.csv": "SAMPLE_SUBMISSION_CSV",
    }
    _dir_var = {
        "train_audio": "TRAIN_AUDIO_DIR",
        "train_soundscapes": "TRAIN_SOUNDSCAPES_DIR",
        "test_soundscapes": "TEST_SOUNDSCAPES_DIR",
    }
    lines = [
        "## DATA LAYOUT (same on every machine — under project root)",
        "Project root = folder containing `src/` and `data/`. Discovered at runtime from __file__.",
        "Use ONLY the injected constants below — never absolute paths (/Users/..., Desktop/data, etc.).",
        "",
        "## INJECTED CONSTANTS (prepended automatically — do not redefine)",
        EDA_PATH_CONSTANTS_DOC,
        "",
        "## FILES ON THIS MACHINE (for your awareness)",
    ]
    for name, info in preflight["csv_files"].items():
        status = "FOUND" if info["exists"] else "MISSING"
        rows = info["rows"]
        row_s = f"{rows} rows" if rows is not None else "n/a"
        cols = ", ".join(info["columns"]) if info["columns"] else "n/a"
        var = _csv_var.get(name, "DATA")
        lines.append(f"- {info['rel_path']} [{status}] — {row_s}; load via `_load_csv({var})`")
        lines.append(f"    columns: {cols}")
    lines.append("")
    for label, info in preflight["dirs"].items():
        status = "FOUND" if info["exists"] else "MISSING"
        var = _dir_var.get(label, "DATA")
        lines.append(f"- {info['rel_path']}/ [{status}] — use `{var}` in code")
        if label == "train_audio" and info.get("species_subdirs") is not None:
            lines.append(f"    species subdirectories: {info['species_subdirs']}")
        ogg = info.get("ogg_count")
        if ogg is not None:
            lines.append(f"    *.ogg files (scanned): {ogg}")
        elif info["exists"]:
            lines.append("    *.ogg count: large tree (scan capped)")
    lines.append("")
    lines.append(
        "If a CSV shows 0 rows or is MISSING, print that fact and skip ratios "
        "(never divide by len(rows) when rows is empty)."
    )
    return "\n".join(lines)


def _strip_llm_path_setup(code: str) -> str:
    """Remove echoed bootstrap / path setup; keep analysis-only lines for the LLM body."""
    # If the model pasted multiple bootstrap blocks, keep only text after the last _sep helper.
    marker = "def _sample_ogg_files"
    if marker in code:
        code = code.split(marker, 1)[-1]
        if ")" in code:
            code = code.split(")", 1)[-1]

    lines = code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if any(m in line for m in _BOOTSTRAP_STRIP_MARKERS):
            i += 1
            continue
        if s.startswith("import ") and s.split()[1].split(".")[0] in (
            "csv",
            "collections",
            "ast",
            "random",
            "os",
            "sys",
            "json",
            "pathlib",
        ):
            i += 1
            continue
        if ("open(" in line or "Path(" in line) and (
            "/Users/" in line or "/home/" in line or "Desktop/data" in line
        ):
            i += 1
            if i < len(lines) and lines[i].strip() in ("pass", "..."):
                i += 1
            continue
        if s in ("pass", "..."):
            i += 1
            continue
        break
    body = "\n".join(lines[i:]).strip()
    # Drop pandas-style lines the model often adds
    cleaned = []
    for line in body.splitlines():
        if any(f in line for f in _FORBIDDEN_IN_ANALYSIS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def prepare_eda_script_for_execution(llm_code: str) -> str:
    """
    Prepend the portable bootstrap to LLM analysis code (exactly once).
    Paths always resolve via src/+data/ discovery — works on any computer.
    """
    body = _strip_llm_path_setup(llm_code)
    return EDA_CODEGEN_BOOTSTRAP.rstrip() + ("\n\n" + body if body else "")


def _run_prepared_snippet(executor, script_path: Path, analysis_code: str) -> tuple[str | None, object]:
    """Execute bootstrap + analysis snippet; return (stdout, result)."""
    script_path.write_text(prepare_eda_script_for_execution(analysis_code), encoding="utf-8")
    result, _ = _run_eda_script_once(executor, script_path)
    if _eda_script_run_ok(result):
        return (result.stdout or "").strip(), result
    return None, result


def save_eda_preflight(logs_dir: Path, preflight: dict) -> None:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "eda_preflight.json").write_text(
        json.dumps(preflight, indent=2),
        encoding="utf-8",
    )
    (logs_dir / "eda_preflight.txt").write_text(
        format_eda_preflight_block(preflight) + "\n",
        encoding="utf-8",
    )


def resolve_eda_codegen_options(config: dict | None) -> dict:
    """Codegen flags from ``eda`` / ``meta_agent.eda`` config."""
    cfg = config or {}
    eda_cfg = cfg.get("eda") or {}
    meta_eda = (cfg.get("meta_agent") or {}).get("eda") or {}
    use_iterative = bool(
        eda_cfg.get("use_iterative_eda", meta_eda.get("use_iterative_eda", True))
    )
    return {
        "use_iterative_eda": use_iterative,
        "exploration_iterations": int(
            eda_cfg.get("exploration_iterations")
            or meta_eda.get("exploration_iterations")
            or DEFAULT_EDA_EXPLORATION_ITERATIONS
        ),
        "fixes_per_iteration": int(
            eda_cfg.get("fixes_per_iteration")
            or meta_eda.get("fixes_per_iteration")
            or DEFAULT_EDA_FIXES_PER_ITERATION
        ),
        "staged_codegen": bool(
            eda_cfg.get("staged_codegen", meta_eda.get("staged_codegen", False))
        ),
    }


# Guided exploration plan: theme + working reference + hardcoded fallback (from EDA_SCRIPT)
EDA_EXPLORATION_PLAN: list[dict[str, str]] = [
    {
        "title": "SECTION 5 — MULTI-LABEL",
        "theme": "Multi-label secondary_labels using _parse_secondary_labels_row on each train row.",
        "reference": (
            "train = _load_csv(TRAIN_CSV)\n"
            "has_secondary = sum(1 for r in train if _parse_secondary_labels_row(r))\n"
            "Use len(train) for ratios; skip if train is empty."
        ),
        "fallback": '''
_sep("SECTION 5: MULTI-LABEL STRUCTURE")
train = _load_csv(TRAIN_CSV)
if not train:
    print("  train.csv empty — skipped")
else:
    has_secondary = 0
    total_secondary = 0
    for r in train:
        slist = _parse_secondary_labels_row(r)
        if slist:
            has_secondary += 1
            total_secondary += len(slist)
    print(f"  Recordings with secondary labels: {has_secondary} ({has_secondary/len(train)*100:.1f}%)")
    print(f"  Total secondary label entries: {total_secondary}")
    if has_secondary:
        print(f"  Avg secondary labels (when present): {total_secondary/has_secondary:.2f}")
''',
    },
    {
        "title": "SECTION 6 — TAXONOMY",
        "theme": "Taxonomy mapping via TAXONOMY_CSV and train species coverage.",
        "reference": "taxonomy = _load_csv(TAXONOMY_CSV); use Counter on columns order/family/class_name if present.",
        "fallback": '''
_sep("SECTION 6: TAXONOMY / SPECIES MAPPING")
if not TAXONOMY_CSV.is_file():
    print("  taxonomy.csv not found")
else:
    taxonomy = _load_csv(TAXONOMY_CSV)
    train = _load_csv(TRAIN_CSV)
    species_counts = _train_species_counts(train)
    print(f"  Taxonomy rows: {len(taxonomy)}")
    print(f"  Taxonomy columns: {list(taxonomy[0].keys()) if taxonomy else []}")
    tax_cols = list(taxonomy[0].keys()) if taxonomy else []
    if "order" in tax_cols:
        print(f"  Unique orders: {len(Counter(r.get('order', '?') for r in taxonomy))}")
    if "family" in tax_cols:
        print(f"  Unique families: {len(Counter(r.get('family', '?') for r in taxonomy))}")
    print(f"  Taxonomic class breakdown: {dict(Counter(r.get('class_name', '?') for r in taxonomy))}")
    tax_species = {r.get("species_code", r.get("primary_label", "")) for r in taxonomy}
    not_in_tax = set(species_counts.keys()) - tax_species
    if not_in_tax:
        print(f"  Train species NOT in taxonomy: {len(not_in_tax)} -> {list(not_in_tax)[:5]}")
    else:
        print("  All train species found in taxonomy.")
''',
    },
    {
        "title": "SECTION 7 — SOUNDSCAPES",
        "theme": "Soundscape labels vs focal train species; use _soundscape_species and _train_species_counts.",
        "reference": (
            "sc_labels = _load_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)\n"
            "sc_species = _soundscape_species(sc_labels)\n"
            "train_species = set(_train_species_counts(_load_csv(TRAIN_CSV)).keys())"
        ),
        "fallback": '''
_sep("SECTION 7: TRAIN SOUNDSCAPES")
if not TRAIN_SOUNDSCAPES_LABELS_CSV.is_file():
    print("  train_soundscapes_labels.csv not found")
else:
    sc_labels = _load_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)
    train = _load_csv(TRAIN_CSV)
    species_counts = _train_species_counts(train)
    sc_species = _soundscape_species(sc_labels)
    print(f"  Soundscape label rows: {len(sc_labels)}")
    print(f"  Unique species in soundscapes: {len(sc_species)}")
    train_audio_species = set(species_counts.keys())
    only_in_sc = sc_species - train_audio_species
    only_in_audio = train_audio_species - sc_species
    both = sc_species & train_audio_species
    print(f"  Species in BOTH train_audio and soundscapes: {len(both)}")
    print(f"  Species ONLY in soundscapes: {len(only_in_sc)} -> {list(only_in_sc)[:10]}")
    print(f"  Species ONLY in train_audio: {len(only_in_audio)}")
''',
    },
    {
        "title": "SECTION 8 — FOCAL COVERAGE GAP",
        "theme": "Compare SAMPLE_SUBMISSION_CSV species columns to train focal species.",
        "reference": (
            "sub = _load_csv(SAMPLE_SUBMISSION_CSV)\n"
            "sub_species = [c for c in sub[0].keys() if c != 'row_id']\n"
            "species_counts = _train_species_counts(_load_csv(TRAIN_CSV))"
        ),
        "fallback": '''
_sep("SECTION 8: FOCAL COVERAGE GAP (submission species vs train_audio)")
if not SAMPLE_SUBMISSION_CSV.is_file():
    print("  sample_submission.csv not found")
else:
    sub = _load_csv(SAMPLE_SUBMISSION_CSV)
    train = _load_csv(TRAIN_CSV)
    species_counts = _train_species_counts(train)
    sub_species = [c for c in sub[0].keys() if c != "row_id"] if sub else []
    print(f"  Species required in submission: {len(sub_species)}")
    print(f"  Species in train_audio: {len(species_counts)}")
    missing = set(sub_species) - set(species_counts.keys())
    print(f"  Submission species with NO train_audio examples: {len(missing)}")
    print(f"  -> {sorted(missing)[:15]} ...")
''',
    },
    {
        "title": "SECTION 9 — AUDIO SAMPLES",
        "theme": "Librosa on up to 20 paths from _sample_ogg_files(TRAIN_AUDIO_DIR) — NOT os.listdir.",
        "reference": (
            "files = _sample_ogg_files(TRAIN_AUDIO_DIR, n=20)\n"
            "for f in files: librosa.get_duration(path=str(f))"
        ),
        "fallback": '''
_sep("SECTION 9: AUDIO SAMPLE CHARACTERISTICS")
try:
    import librosa
    sample_files = _sample_ogg_files(TRAIN_AUDIO_DIR, n=20)
    durations, sample_rates = [], []
    for f in sample_files:
        try:
            durations.append(librosa.get_duration(path=str(f)))
            sample_rates.append(librosa.get_samplerate(str(f)))
        except Exception:
            pass
    if durations:
        print(f"  Sample size: {len(durations)} files")
        print(f"  Duration range: {min(durations):.1f}s to {max(durations):.1f}s")
        print(f"  Median duration: {sorted(durations)[len(durations)//2]:.1f}s")
        print(f"  Unique sample rates: {sorted(set(sample_rates))}")
    else:
        print("  No audio samples collected")
except ImportError:
    print("  librosa not available — skipped")
''',
    },
    {
        "title": "SECTION 10 — LONG-TAIL",
        "theme": "Rare species counts from _train_species_counts.",
        "reference": "counts = _train_species_counts(_load_csv(TRAIN_CSV)); list rarest via counts.most_common()",
        "fallback": '''
_sep("SECTION 10: LONG-TAIL SPECIES")
train = _load_csv(TRAIN_CSV)
counts = _train_species_counts(train)
if not counts:
    print("  no train species")
else:
    scarce5 = sum(1 for c in counts.values() if c < 5)
    scarce10 = sum(1 for c in counts.values() if c < 10)
    print(f"  Species with <5 recordings: {scarce5}")
    print(f"  Species with <10 recordings: {scarce10}")
    rarest = sorted(counts.items(), key=lambda x: x[1])[:15]
    print(f"  15 rarest: {rarest}")
''',
    },
    {
        "title": "SECTION 11 — COLLECTION × RATING",
        "theme": "Per-collection rating stats from train rows (no pandas).",
        "reference": "Loop train rows; bucket by row.get('collection'); parse float rating.",
        "fallback": '''
_sep("SECTION 11: COLLECTION × RATING")
train = _load_csv(TRAIN_CSV)
by_coll = Counter(r.get("collection", "?") for r in train)
print(f"  Collection counts: {dict(by_coll)}")
for coll in by_coll:
    ratings = []
    for r in train:
        if r.get("collection", "?") != coll:
            continue
        raw = r.get("rating", "").strip()
        if raw:
            try:
                ratings.append(float(raw))
            except ValueError:
                pass
    if ratings:
        z = sum(1 for x in ratings if x == 0)
        print(f"  {coll}: n={len(ratings)} zero-rated={z} ({z/len(ratings)*100:.1f}%)")
''',
    },
    {
        "title": "SECTION 12 — SOUNDSCAPE DENSITY",
        "theme": "Top filenames by label-row count in soundscape labels.",
        "reference": "Counter(r['filename'] for r in sc_labels).most_common(5)",
        "fallback": '''
_sep("SECTION 12: SOUNDSCAPE LABEL DENSITY")
sc_labels = _load_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)
if not sc_labels:
    print("  no soundscape labels")
else:
    by_file = Counter(r.get("filename", "?") for r in sc_labels)
    print(f"  Top 5 busiest soundscape files: {by_file.most_common(5)}")
''',
    },
    {
        "title": "SECTION 13 — LAT/LON",
        "theme": "Latitude/longitude coverage in train if columns exist.",
        "reference": "Check train[0].keys() for latitude/longitude before parsing floats.",
        "fallback": '''
_sep("SECTION 13: GEOGRAPHIC COVERAGE")
train = _load_csv(TRAIN_CSV)
if not train or "latitude" not in train[0] or "longitude" not in train[0]:
    print("  latitude/longitude columns missing — skipped")
else:
    valid = 0
    lats, lons = [], []
    for r in train:
        la, lo = r.get("latitude", "").strip(), r.get("longitude", "").strip()
        if la and lo:
            try:
                lats.append(float(la))
                lons.append(float(lo))
                valid += 1
            except ValueError:
                pass
    print(f"  Rows with valid lat/lon: {valid} ({valid/len(train)*100:.1f}%)")
    if lats:
        print(f"  Lat range: {min(lats):.2f} to {max(lats):.2f}")
        print(f"  Lon range: {min(lons):.2f} to {max(lons):.2f}")
''',
    },
    {
        "title": "SECTION 14 — FOLLOW-UP",
        "theme": "One new quantitative check suggested by cumulative findings (data only).",
        "reference": "Pick one gap from cumulative findings; use helpers above.",
        "fallback": '''
_sep("SECTION 14: FOLLOW-UP CHECK")
train = _load_csv(TRAIN_CSV)
counts = _train_species_counts(train)
print(f"  Quick recap: {len(train)} clips, {len(counts)} species, median clips/species: {sorted(counts.values())[len(counts)//2] if counts else 0}")
''',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM CODE GENERATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_EDA_CODEGEN_SYSTEM = """\
You are an autonomous ML research agent preparing to train a BirdCLEF+ 2026 model.
Before writing any model code you must explore the dataset yourself.

A portable path bootstrap is prepended automatically when your code runs (finds project
root via src/ + data/ next to the script). You write ONLY the analysis portion.

Use the injected constants TRAIN_CSV, TAXONOMY_CSV, TRAIN_AUDIO_DIR, etc.
Never use absolute paths (/Users/..., /home/..., Desktop/data, or open("data/...") without constants).
Return ONLY a ```python``` code block — analysis code only, no path setup."""

_EDA_CODEGEN_USER_TEMPLATE = """\
You are about to start training a multi-label bird sound classifier for BirdCLEF+ 2026.
Before writing any model, explore the dataset to understand it.

{eda_preflight}

## YOUR TASK
{task_body}

Return ONLY analysis code (print sections). Do NOT include:
- Path discovery, PROJECT_ROOT, or DATA definitions
- def _load_csv / def _sep (already injected)
- Absolute filesystem paths

You MAY use: collections, random, ast, Counter — plus librosa in try/except ImportError.
Do NOT use pandas, matplotlib, or any plotting library.
Guard all divisions when len(rows) == 0.
Do NOT mention model architecture. Only print to stdout."""

_EDA_CODEGEN_TASK_FULL = """\
Write analysis code that:
1. Loads each CSV via TRAIN_CSV, TAXONOMY_CSV, TRAIN_SOUNDSCAPES_LABELS_CSV, SAMPLE_SUBMISSION_CSV
2. Prints row counts, columns, missing values
3. Analyses species/class distribution in train data (unique species, min/max/median per species, scarce species)
4. Checks multi-label structure (secondary_labels fraction and counts)
5. Checks recording quality (rating distribution, XC vs iNat split)
6. Counts .ogg files under TRAIN_AUDIO_DIR and TRAIN_SOUNDSCAPES_DIR
7. Compares SAMPLE_SUBMISSION_CSV species vs train species (zero focal examples)
8. Samples up to 20 random .ogg files from TRAIN_AUDIO_DIR with librosa (duration, sample rate)
9. Identifies species present ONLY in soundscapes (not in train_audio)"""

_EDA_CODEGEN_TASK_STAGE1 = """\
STAGE 1 analysis only (must run quickly):
1. File inventory: for each CSV constant, print exists, row count, columns via _load_csv(TRAIN_CSV) etc.
2. Species count and min/max/median recordings per species from TRAIN_CSV (skip ratios if empty)

Do not sample librosa audio yet."""

_EDA_CODEGEN_TASK_STAGE2 = """\
STAGE 1 succeeded. Write the FULL analysis (stage 1 plus new sections):
- Multi-label stats, rating/collection split, soundscape vs focal overlap
- SAMPLE_SUBMISSION_CSV coverage gap, librosa on up to 20 files from TRAIN_AUDIO_DIR
- End with print("EDA COMPLETE")

Return the complete analysis body (stage 1 + new parts). Path bootstrap is still auto-injected."""

_EDA_CODEGEN_FIX_USER = """\
Your previous EDA analysis code failed when executed. Fix it and return corrected analysis code only.

## EXECUTION ERROR (attempt {attempt}/{max_attempts})
Exit code: {exit_code}
{timed_out_note}

### STDERR
{stderr}

### STDOUT (tail)
{stdout_tail}

## YOUR PREVIOUS CODE (analysis only — bootstrap re-injected on run)
```python
{previous_code}
```

## ORIGINAL TASK
{original_task}

Use ONLY: TRAIN_CSV, TAXONOMY_CSV, TRAIN_AUDIO_DIR, DATA, _load_csv, _sep — no absolute paths.
Guard divisions when CSVs are empty.

Return ONLY corrected analysis code in a ```python``` block (no path bootstrap). \
Print section headers; no files or plots."""

_EDA_ITERATION_SYSTEM = """\
You extend a BirdCLEF+ 2026 EDA (base sections 1–4 already done).

Bootstrap is prepended automatically: paths, _load_csv (returns list[dict]), helpers.

CRITICAL: _load_csv is NOT pandas. Forbidden: pandas, pd, .empty, .dropna, DataFrame.
Use: _parse_secondary_labels_row, _soundscape_species, _train_species_counts, _sample_ogg_files.
For audio use _sample_ogg_files(TRAIN_AUDIO_DIR) — never os.listdir on train_audio (species subfolders).

Return ONLY a ```python``` analysis block (no bootstrap, no path setup)."""

_EDA_ITERATION_USER_TEMPLATE = """\
## ITERATION {iteration}/{total_iterations} — {title}
## THEME
{theme}

## WORKING REFERENCE PATTERN (adapt this — do not use pandas)
```python
{reference}
```

## ALREADY COVERED
{completed_sections}

## CUMULATIVE FINDINGS SO FAR
{cumulative_findings}

{eda_preflight}

## RULES
- Start with _sep("{title}") or print("{title}").
- Use helpers from bootstrap; reload CSVs with _load_csv(TRAIN_CSV) when needed.
- ast/random/Counter already imported; librosa only in try/except.
- Guard len()==0 before divisions. Keep runtime short.

Write ONLY this section's analysis code."""

_EDA_ITERATION_FIX_USER = """\
Iteration {iteration}/{total_iterations} ({title}) — fix analysis code only.

## ERROR (fix {attempt}/{max_attempts})
Exit code: {exit_code}
{timed_out_note}

### STDERR
{stderr}

### STDOUT (tail)
{stdout_tail}

## YOUR CODE
```python
{previous_code}
```

## THEME
{theme}

## REFERENCE
```python
{reference}
```

_load_csv → list[dict]. NO pandas/pd/.empty/.dropna. Audio: _sample_ogg_files(TRAIN_AUDIO_DIR).
Return ONLY corrected ```python``` analysis."""


DEFAULT_EDA_CODEGEN_ATTEMPTS = 5
DEFAULT_EDA_EXPLORATION_ITERATIONS = 10
DEFAULT_EDA_FIXES_PER_ITERATION = 5
EDA_EXPLORATION_LOG_FILE = "eda_exploration_log.txt"
# Per subprocess run of eda_script.py (hardcoded script is usually a few minutes)
DEFAULT_EDA_SCRIPT_TIMEOUT_SECONDS = 600
# LLM calls (codegen, fix, summary, brief)
DEFAULT_EDA_LLM_TIMEOUT_SECONDS = 600
# Wall-clock cap for all codegen attempts combined (then fallback); ~5× script timeout + buffer
DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS = 3600

_EDA_TIMEOUT_FIX_HINT = (
    "\n\nThe script TIMED OUT. Rewrite it to finish within a few minutes: "
    "count CSV rows with line iteration (no pandas); cap librosa to at most 20 audio files; "
    "avoid rglob over huge trees more than once; print progress every 5000 rows if needed."
)


def resolve_eda_timeouts(config: dict) -> tuple[int, int, int]:
    """
    Return (script_timeout_seconds, llm_timeout_seconds, max_codegen_wall_seconds).
    """
    eda_cfg = config.get("eda") or {}
    meta_eda = (config.get("meta_agent") or {}).get("eda") or {}
    exec_cfg = config.get("execution") or {}

    script_timeout = int(
        eda_cfg.get("script_timeout_seconds")
        or meta_eda.get("script_timeout_seconds")
        or eda_cfg.get("timeout_seconds")  # legacy single timeout
        or exec_cfg.get("eda_script_timeout_seconds")
        or DEFAULT_EDA_SCRIPT_TIMEOUT_SECONDS
    )
    llm_timeout = int(
        eda_cfg.get("llm_timeout_seconds")
        or meta_eda.get("llm_timeout_seconds")
        or exec_cfg.get("eda_llm_timeout_seconds")
        or DEFAULT_EDA_LLM_TIMEOUT_SECONDS
    )
    max_wall = int(
        eda_cfg.get("max_codegen_wall_seconds")
        or meta_eda.get("max_codegen_wall_seconds")
        or DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS
    )
    return max(60, script_timeout), max(30, llm_timeout), max(script_timeout, max_wall)


def build_eda_clients(config: dict) -> tuple:
    """
    Build :class:`CodeExecutor` and :class:`LLMClient` with EDA-specific timeouts.
    """
    script_timeout, llm_timeout, max_wall = resolve_eda_timeouts(config)
    py_exe = config.get("execution", {}).get("python_executable", "python3")
    eda_cfg = config.get("eda") or {}
    meta_eda = (config.get("meta_agent") or {}).get("eda") or {}
    provider = str(
        eda_cfg.get("provider")
        or meta_eda.get("provider")
        or config.get("llm", {}).get("provider", "ollama")
    )
    model = str(
        eda_cfg.get("model")
        or meta_eda.get("model")
        or config.get("llm", {}).get("model", "qwen2.5-coder:7b")
    )
    print(
        f"  EDA timeouts: script={script_timeout}s per run  |  "
        f"LLM={llm_timeout}s  |  codegen wall={max_wall}s"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from code_executor import CodeExecutor
    from llm_client import LLMClient

    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=script_timeout)
    llm = LLMClient(provider=provider, model=model, timeout_seconds=llm_timeout)
    return executor, llm, max_wall


# ─────────────────────────────────────────────────────────────────────────────
# LLM SUMMARISATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_EDA_SUMMARY_SYSTEM_PROMPT = """\
You are an expert ML researcher summarizing exploratory data analysis for \
BirdCLEF+ 2026 multi-label bird-sound classification.

Write a factual DATA summary only. Do NOT give model-building advice.

FORBIDDEN in your summary (do not mention):
- Output dimension, num_classes, number of logits, layer width, or head size
- "The model must have N outputs" or similar architecture rules
- Key constraints for model design / hyperparameter recipes for specific architectures"""

_EDA_SUMMARY_USER_TEMPLATE = """\
Below is raw stdout from an automated EDA script on the BirdCLEF+ 2026 dataset.
Summarize into a structured report using the sections below.
Use only facts supported by the numbers in the output — do not speculate.

## RAW EDA OUTPUT
{eda_output}

## REQUIRED SECTIONS

### 1. DATASET STRUCTURE
- Files present, row counts, key columns.
- Train audio clip count and species count in train.csv.
- Soundscape file and label counts.

### 2. CLASS DISTRIBUTION & IMBALANCE
- Total species, min/max/median recordings per species.
- Severity of long-tail imbalance.
- Taxonomic class breakdown (birds vs insects etc.) if available.

### 3. MULTI-LABEL STRUCTURE
- Fraction of recordings with secondary labels.
- Typical number of labels per recording if shown.

### 4. FOCAL COVERAGE GAP
- How many evaluation-list species have ZERO focal train_audio clips (count only).
- Brief note if many species appear only in soundscapes vs focal audio.

### 5. DATA QUALITY SIGNALS
- Rating distribution (e.g. zero-rated / iNat vs verified).
- Recording source bias (e.g. XC vs iNat) if shown.

### 6. AUDIO CHARACTERISTICS
- Typical clip duration and sample rate from samples.
- Whether clips are mostly ~5 seconds or highly variable.

### 7. DATA & TRAINING IMPLICATIONS (data-only)
- Bullet list of the most important **dataset** facts (imbalance, noise, label sparsity, \
rare species, soundscape vs focal coverage, augmentation relevance).
- Stay on data and labels — no architecture, no output sizes, no loss-function prescriptions.

Start directly with '### 1. DATASET STRUCTURE'."""


_EDA_BRIEF_SYSTEM_PROMPT = """\
You compress an EDA summary into exactly two short sentences for a research agent.

Rules:
- Exactly 2 sentences, plain text (no markdown headers).
- Each sentence is a compact semicolon-separated list of the most important data facts.
- Use ONLY facts present in the summary — do not invent numbers.
- Do NOT mention output dimension, num_classes, layer size, logits, or model architecture."""

_EDA_BRIEF_USER_TEMPLATE = """\
From the EDA summary below, write exactly TWO sentences.

Sentence 1: dataset scale, imbalance, multi-label prevalence.
Sentence 2: audio/window characteristics; focal vs soundscape coverage gaps.

Do not mention model architecture or output dimensions.

## EDA SUMMARY
{eda_summary}
"""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _extract_python_block(text: str) -> str:
    """Extract code from a ```python ... ``` block, or return text as-is."""
    import re
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


EDA_SUMMARY_FILE = "eda_summary.txt"
EDA_BRIEF_FILE = "eda_brief.txt"
EDA_RAW_FILE = "eda_raw_output.txt"
# Legacy name kept for callers not yet migrated
EDA_INSIGHTS_FILE = "eda_insights.txt"


def _eda_paths(logs_dir: Path) -> dict[str, Path]:
    logs_dir = Path(logs_dir)
    return {
        "raw": logs_dir / EDA_RAW_FILE,
        "summary": logs_dir / EDA_SUMMARY_FILE,
        "brief": logs_dir / EDA_BRIEF_FILE,
        "legacy_insights": logs_dir / EDA_INSIGHTS_FILE,
    }


def _write_text_report(path: Path, title: str, body: str) -> None:
    header = (
        f"# BirdCLEF+ 2026 — {title}\n"
        f"# Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n\n"
    )
    path.write_text(header + body.strip() + "\n", encoding="utf-8")


def _llm_response_failed(response: str | None) -> bool:
    return not response or str(response).startswith("Error")


def _eda_script_run_ok(result) -> bool:
    """Success = clean exit and non-empty stdout."""
    return bool(result.success and (result.stdout or "").strip())


def _run_eda_script_once(executor, script_path: Path) -> tuple:
    """Run script; return (ExecutionResult, elapsed_sec)."""
    t0 = time.time()
    result = executor.run_file(script_path)
    return result, time.time() - t0


def _build_codegen_user_message(preflight: dict, task_body: str) -> str:
    return _EDA_CODEGEN_USER_TEMPLATE.format(
        eda_preflight=format_eda_preflight_block(preflight),
        task_body=task_body.strip(),
    )


def _cumulative_tail(text: str, max_chars: int = 10_000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text or "(no output yet)"
    return "…\n" + text[-max_chars:]


def _run_eda_base_script(executor, script_path: Path, code_dir: Path) -> str | None:
    """Execute hardcoded sections 1–4; returns stdout or None on failure."""
    base_path = code_dir / "eda_base_script.py"
    base_path.write_text(EDA_BASE_SCRIPT, encoding="utf-8")
    script_path.write_text(EDA_BASE_SCRIPT, encoding="utf-8")
    print("  [EDA] Running hardcoded base script (sections 1–4) …")
    result, elapsed = _run_eda_script_once(executor, base_path)
    print(
        f"  [EDA base] Finished in {elapsed:.1f}s  "
        f"(exit={result.return_code}, stdout={len(result.stdout or '')} chars)"
    )
    if _eda_script_run_ok(result):
        return (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stderr:
        print(f"  [EDA base] stderr tail: …{stderr[-400:]}")
    return None


def _run_one_exploration_iteration(
    executor,
    script_path: Path,
    code_dir: Path,
    llm,
    *,
    iteration: int,
    total_iterations: int,
    plan: dict[str, str],
    preflight: dict,
    cumulative: str,
    completed_sections: list[str],
    temperature: float,
    fixes_per_iteration: int,
    wall_start: float,
    max_wall_seconds: int,
) -> tuple[str | None, str | None]:
    """
    One exploration theme with up to ``fixes_per_iteration`` LLM fix attempts.
    On failure, runs the hardcoded fallback snippet for this section if available.
    Returns (stdout, analysis_code) or (None, last_code).
    """
    fixes_per_iteration = max(1, int(fixes_per_iteration))
    script_cap = getattr(executor, "timeout_seconds", DEFAULT_EDA_SCRIPT_TIMEOUT_SECONDS)
    title = plan.get("title", f"iteration {iteration}")
    theme = plan.get("theme", "")
    reference = plan.get("reference", "").strip()
    fallback = plan.get("fallback", "").strip()
    user_msg = _EDA_ITERATION_USER_TEMPLATE.format(
        iteration=iteration,
        total_iterations=total_iterations,
        title=title,
        theme=theme,
        reference=reference or "# use bootstrap helpers",
        completed_sections=(
            "\n".join(f"- {s}" for s in completed_sections) if completed_sections else "- (base only)"
        ),
        cumulative_findings=_cumulative_tail(cumulative),
        eda_preflight=format_eda_preflight_block(preflight),
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _EDA_ITERATION_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    code: str | None = None
    label = f" iter-{iteration}"

    for attempt in range(1, fixes_per_iteration + 1):
        if time.time() - wall_start > max_wall_seconds:
            print(f"  [EDA{label}] Wall-clock budget exceeded — stopping iteration.")
            return None, code

        print(f"  [EDA{label}] Fix attempt {attempt}/{fixes_per_iteration} — asking LLM …")
        response = llm.generate_from_messages(messages=messages, temperature=temperature)
        if _llm_response_failed(response):
            print(f"  [EDA{label}] LLM error: {str(response)[:200]}")
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid response. Return ONLY a ```python``` analysis block.",
                }
            )
            continue

        extracted = _extract_python_block(response)
        if not extracted:
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": "Return ONLY exploration analysis inside a ```python``` fence.",
                }
            )
            continue

        code = prepare_eda_script_for_execution(extracted)
        iter_script = code_dir / f"eda_iter_{iteration:02d}.py"
        iter_script.write_text(code, encoding="utf-8")
        script_path.write_text(code, encoding="utf-8")

        result, elapsed = _run_eda_script_once(executor, iter_script)
        print(
            f"  [EDA{label}] Run finished in {elapsed:.1f}s  "
            f"(exit={result.return_code}, stdout={len(result.stdout or '')} chars)"
        )
        if _eda_script_run_ok(result):
            stdout = (result.stdout or "").strip()
            (code_dir / f"eda_iter_{iteration:02d}.txt").write_text(stdout + "\n", encoding="utf-8")
            print(f"  [EDA{label}] Success.")
            return stdout, code

        stderr = (result.stderr or "").strip() or "(empty)"
        stdout_tail = (result.stdout or "")[-2500:] or "(empty)"
        print(f"  [EDA{label}] Failed — feeding stderr back …")
        if len(stderr) > 400:
            print(f"    stderr tail: …{stderr[-400:]}")

        timed_out_note = (
            f"TIMED OUT after {script_cap}s — simplify (cap librosa at 20 files, no huge rglob).\n"
            if result.timed_out
            else ""
        )
        fix_user = _EDA_ITERATION_FIX_USER.format(
            iteration=iteration,
            total_iterations=total_iterations,
            title=title,
            attempt=attempt,
            max_attempts=fixes_per_iteration,
            exit_code=result.return_code,
            timed_out_note=timed_out_note,
            stderr=stderr[:4000],
            stdout_tail=stdout_tail,
            previous_code=_strip_llm_path_setup(code or "")[:12000],
            theme=theme,
            reference=reference or "# use bootstrap helpers",
        )
        if result.timed_out:
            fix_user += _EDA_TIMEOUT_FIX_HINT
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": fix_user})

    if fallback:
        print(f"  [EDA{label}] LLM failed — running hardcoded fallback for {title} …")
        iter_script = code_dir / f"eda_iter_{iteration:02d}.py"
        stdout, result = _run_prepared_snippet(executor, iter_script, fallback)
        if stdout:
            iter_script.write_text(prepare_eda_script_for_execution(fallback), encoding="utf-8")
            (code_dir / f"eda_iter_{iteration:02d}.txt").write_text(stdout + "\n", encoding="utf-8")
            print(f"  [EDA{label}] Fallback succeeded.")
            return stdout, fallback
        stderr = (result.stderr or "").strip()
        if stderr:
            print(f"    fallback stderr: …{stderr[-300:]}")

    print(f"  [EDA{label}] All attempts failed — skipping {title}.")
    return None, code


def _run_iterative_eda_exploration(
    executor,
    script_path: Path,
    code_dir: Path,
    llm,
    *,
    temperature: float,
    max_wall_seconds: int,
    logs_dir: Path | None,
    config: dict | None,
) -> str | None:
    """
    Hardcoded base → N guided LLM explorations (each with fix retries) → combined stdout.
    Returns None if base script fails (caller should run full EDA_SCRIPT fallback).
    """
    opts = resolve_eda_codegen_options(config)
    n_iter = max(1, opts["exploration_iterations"])
    fixes = max(1, opts["fixes_per_iteration"])
    plans = EDA_EXPLORATION_PLAN[:n_iter]
    if len(plans) < n_iter:
        print(
            f"  [EDA] Warning: only {len(plans)} plans defined; running {len(plans)} iterations."
        )
        n_iter = len(plans)

    preflight = gather_eda_preflight()
    if logs_dir is not None:
        save_eda_preflight(logs_dir, preflight)

    train_rows = preflight["csv_files"]["train.csv"]["rows"]
    print(
        f"  [EDA] Iterative exploration: {n_iter} iteration(s), "
        f"{fixes} fix(es) each  |  train.csv rows={train_rows}"
    )

    wall_start = time.time()
    base_out = _run_eda_base_script(executor, script_path, code_dir)
    if base_out is None:
        print("  [EDA] Hardcoded base failed — will use full EDA_SCRIPT fallback.")
        return None

    cumulative = (
        "=" * 60 + "\n"
        "HARDCODED BASE (sections 1–4)\n"
        "=" * 60 + "\n"
        + base_out
    )
    completed: list[str] = [
        "SECTION 1: file inventory",
        "SECTION 2: train.csv overview",
        "SECTION 3: species & class distribution",
        "SECTION 4: recording quality",
    ]
    n_ok = 0

    if logs_dir is not None:
        log_path = Path(logs_dir) / EDA_EXPLORATION_LOG_FILE
        log_path.write_text(cumulative + "\n", encoding="utf-8")

    for i, plan in enumerate(plans, start=1):
        if time.time() - wall_start > max_wall_seconds:
            print(f"  [EDA] Wall-clock budget exceeded after {n_ok} exploration(s).")
            break

        title = plan.get("title", f"iteration {i}")
        print(f"\n  [EDA] Exploration {i}/{n_iter} — {title} …")
        section_out, _ = _run_one_exploration_iteration(
            executor,
            script_path,
            code_dir,
            llm,
            iteration=i,
            total_iterations=n_iter,
            plan=plan,
            preflight=preflight,
            cumulative=cumulative,
            completed_sections=completed,
            temperature=temperature,
            fixes_per_iteration=fixes,
            wall_start=wall_start,
            max_wall_seconds=max_wall_seconds,
        )
        if section_out:
            header = f"\n{'=' * 60}\nEXPLORATION {i}/{n_iter}: {title}\n{'=' * 60}\n"
            cumulative += header + section_out
            completed.append(title)
            n_ok += 1
            if logs_dir is not None:
                (Path(logs_dir) / EDA_EXPLORATION_LOG_FILE).write_text(
                    cumulative + "\n", encoding="utf-8"
                )
        else:
            completed.append(f"(skipped) {title}")

    print(
        f"\n  [EDA] Iterative exploration done: base OK, {n_ok}/{n_iter} LLM section(s) succeeded."
    )
    cumulative += f"\n{'=' * 60}\nEDA EXPLORATION COMPLETE\n{'=' * 60}\n"
    snapshot = code_dir / "eda_base_script.py"
    for j in range(n_iter, 0, -1):
        candidate = code_dir / f"eda_iter_{j:02d}.py"
        if candidate.is_file():
            snapshot = candidate
            break
    (code_dir / "eda_script_llm.py").write_text(
        snapshot.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return cumulative.strip()


def _run_llm_eda_codegen_with_retries(
    executor,
    script_path: Path,
    code_dir: Path,
    llm,
    *,
    temperature: float,
    max_attempts: int = DEFAULT_EDA_CODEGEN_ATTEMPTS,
    max_wall_seconds: int = DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS,
    logs_dir: Path | None = None,
    config: dict | None = None,
) -> str | None:
    """
    Ask the coding agent to write EDA Python, execute, and retry with stderr up to
    ``max_attempts`` times. Returns stdout on success, or None if all attempts fail.
    """
    max_attempts = max(1, int(max_attempts))
    max_wall_seconds = max(60, int(max_wall_seconds))
    script_cap = getattr(executor, "timeout_seconds", DEFAULT_EDA_SCRIPT_TIMEOUT_SECONDS)
    codegen_opts = resolve_eda_codegen_options(config)
    staged = codegen_opts["staged_codegen"]

    preflight = gather_eda_preflight()
    if logs_dir is not None:
        save_eda_preflight(logs_dir, preflight)
    train_rows = preflight["csv_files"]["train.csv"]["rows"]
    print(f"  EDA preflight: data/train.csv rows={train_rows} (paths via portable bootstrap)")

    print(
        f"  EDA codegen agent: up to {max_attempts} attempt(s)  |  "
        f"{script_cap}s timeout per run  |  {max_wall_seconds}s total wall clock"
        + ("  |  staged (inventory → full)" if staged else "")
    )
    wall_start = time.time()

    def _run_loop(
        messages: list[dict[str, str]],
        *,
        attempt_budget: int,
        label: str,
        original_task: str,
    ) -> tuple[str | None, str | None]:
        """Returns (stdout, last_code) on success."""
        nonlocal wall_start
        code: str | None = None
        for attempt in range(1, attempt_budget + 1):
            elapsed_wall = time.time() - wall_start
            if elapsed_wall > max_wall_seconds:
                print(
                    f"  [EDA codegen{label}] Wall-clock budget exceeded "
                    f"({elapsed_wall:.0f}s > {max_wall_seconds}s)."
                )
                return None, code

            tag = f"{label} " if label else ""
            print(f"  [EDA codegen{label}] Attempt {attempt}/{attempt_budget} — asking LLM …")
            response = llm.generate_from_messages(
                messages=messages,
                temperature=temperature,
            )
            if _llm_response_failed(response):
                print(f"  [EDA codegen{label}] LLM error: {str(response)[:200]}")
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last response was invalid. Return ONLY a ```python``` code block.",
                    }
                )
                continue

            extracted = _extract_python_block(response)
            if not extracted:
                print(f"  [EDA codegen{label}] No ```python``` block in response.")
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return ONLY a complete EDA script inside a ```python``` fence.",
                    }
                )
                continue

            code = prepare_eda_script_for_execution(extracted)
            script_path.write_text(code, encoding="utf-8")
            if attempt == 1 and not label:
                (code_dir / "eda_script_llm.py").write_text(code, encoding="utf-8")

            print(f"  [EDA codegen{label}] Running eda_script.py (attempt {attempt}) …")
            result, elapsed = _run_eda_script_once(executor, script_path)
            print(
                f"  [EDA codegen{label}] Finished in {elapsed:.1f}s  "
                f"(exit={result.return_code}, stdout={len(result.stdout or '')} chars)"
            )

            if _eda_script_run_ok(result):
                print(f"  [EDA codegen{label}] Success on attempt {attempt}.")
                return (result.stdout or "").strip(), code

            if result.timed_out:
                print(
                    f"  [EDA codegen{label}] Attempt {attempt} timed out after {script_cap}s."
                )

            stderr = (result.stderr or "").strip() or "(empty)"
            stdout_tail = (result.stdout or "")[-2500:] or "(empty)"
            print(f"  [EDA codegen{label}] Failed — feeding error back to agent …")
            if len(stderr) > 400:
                print(f"    stderr tail: …{stderr[-400:]}")

            timed_out_note = (
                f"TIMED OUT after {script_cap}s — simplify the script so it finishes quickly.\n"
                if result.timed_out
                else ""
            )
            fix_user = _EDA_CODEGEN_FIX_USER.format(
                attempt=attempt,
                max_attempts=attempt_budget,
                exit_code=result.return_code,
                timed_out_note=timed_out_note,
                stderr=stderr[:4000],
                stdout_tail=stdout_tail,
                previous_code=_strip_llm_path_setup(code or "")[:12000],
                original_task=original_task,
            )
            if result.timed_out:
                fix_user += _EDA_TIMEOUT_FIX_HINT
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({"role": "user", "content": fix_user})

        return None, code

    if staged:
        stage1_attempts = max(1, min(2, max_attempts // 2))
        stage1_user = _build_codegen_user_message(preflight, _EDA_CODEGEN_TASK_STAGE1)
        messages_s1: list[dict[str, str]] = [
            {"role": "system", "content": _EDA_CODEGEN_SYSTEM},
            {"role": "user", "content": stage1_user},
        ]
        print("  [EDA codegen] Stage 1 — CSV inventory and train.csv overview …")
        stdout_s1, code_s1 = _run_loop(
            messages_s1,
            attempt_budget=stage1_attempts,
            label=" stage-1",
            original_task=stage1_user,
        )
        if stdout_s1 is None:
            print("  [EDA codegen] Stage 1 failed — trying single-shot full script …")
            staged = False
        else:
            stage2_attempts = max(1, max_attempts - stage1_attempts)
            stage2_user = _build_codegen_user_message(preflight, _EDA_CODEGEN_TASK_STAGE2)
            stage2_user += (
                "\n\n## STAGE 1 STDOUT (tail)\n"
                + stdout_s1[-3000:]
                + "\n\n## STAGE 1 ANALYSIS CODE (extend this)\n```python\n"
                + _strip_llm_path_setup(code_s1 or "")[:8000]
                + "\n```"
            )
            messages_s2: list[dict[str, str]] = [
                {"role": "system", "content": _EDA_CODEGEN_SYSTEM},
                {"role": "user", "content": stage1_user},
                {"role": "assistant", "content": f"```python\n{code_s1}\n```"},
                {"role": "user", "content": stage2_user},
            ]
            print("  [EDA codegen] Stage 2 — extend to full EDA …")
            stdout_s2, _ = _run_loop(
                messages_s2,
                attempt_budget=stage2_attempts,
                label=" stage-2",
                original_task=stage2_user,
            )
            if stdout_s2 is not None:
                return stdout_s2
            print("  [EDA codegen] Stage 2 failed — will fall back to hardcoded EDA_SCRIPT.")

    full_user = _build_codegen_user_message(preflight, _EDA_CODEGEN_TASK_FULL)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _EDA_CODEGEN_SYSTEM},
        {"role": "user", "content": full_user},
    ]
    stdout, _ = _run_loop(
        messages,
        attempt_budget=max_attempts,
        label="",
        original_task=full_user,
    )
    if stdout is None:
        print(f"  [EDA codegen] All {max_attempts} attempt(s) failed.")
    return stdout


def _run_hardcoded_eda_fallback(executor, script_path: Path) -> str:
    """Run built-in EDA_SCRIPT after LLM codegen exhausted."""
    print("  Using hardcoded fallback EDA script …")
    script_path.write_text(EDA_SCRIPT, encoding="utf-8")
    result, elapsed = _run_eda_script_once(executor, script_path)
    print(f"  Fallback script finished in {elapsed:.1f}s  (exit={result.return_code})")
    raw_output = result.stdout or ""
    stderr = result.stderr or ""
    if not raw_output.strip():
        raw_output = f"[EDA script produced no stdout]\nSTDERR:\n{stderr[:2000]}"
    return raw_output


def _run_eda_script(
    executor,
    script_path: Path,
    *,
    use_llm_codegen: bool,
    llm,
    temperature: float,
    max_codegen_attempts: int = DEFAULT_EDA_CODEGEN_ATTEMPTS,
    max_codegen_wall_seconds: int = DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS,
    logs_dir: Path | None = None,
    config: dict | None = None,
) -> str:
    """Execute EDA script; LLM codegen with retries, then hardcoded fallback."""
    code_dir = script_path.parent
    code_dir.mkdir(parents=True, exist_ok=True)

    if use_llm_codegen:
        opts = resolve_eda_codegen_options(config)
        if opts.get("use_iterative_eda", True):
            raw = _run_iterative_eda_exploration(
                executor,
                script_path,
                code_dir,
                llm,
                temperature=temperature,
                max_wall_seconds=max_codegen_wall_seconds,
                logs_dir=logs_dir,
                config=config,
            )
        elif opts.get("staged_codegen"):
            raw = _run_llm_eda_codegen_with_retries(
                executor,
                script_path,
                code_dir,
                llm,
                temperature=temperature,
                max_attempts=max_codegen_attempts,
                max_wall_seconds=max_codegen_wall_seconds,
                logs_dir=logs_dir,
                config=config,
            )
        else:
            raw = _run_llm_eda_codegen_with_retries(
                executor,
                script_path,
                code_dir,
                llm,
                temperature=temperature,
                max_attempts=max_codegen_attempts,
                max_wall_seconds=max_codegen_wall_seconds,
                logs_dir=logs_dir,
                config=config,
            )
        if raw:
            return raw
        return _run_hardcoded_eda_fallback(executor, script_path)

    print("  LLM codegen disabled — hardcoded EDA script only")
    return _run_hardcoded_eda_fallback(executor, script_path)


def _summarise_eda_raw(llm, raw_output: str, *, temperature: float) -> str:
    """LLM structured summary from raw stdout (data-only sections)."""
    print("  Asking LLM to summarise EDA findings …")
    truncated = raw_output[:8000]
    user_msg = _EDA_SUMMARY_USER_TEMPLATE.format(eda_output=truncated)
    summary = llm.generate_from_messages(
        messages=[
            {"role": "system", "content": _EDA_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )
    if not summary or summary.startswith("Error"):
        print("  LLM summarisation failed — using truncated raw output as fallback.")
        return truncated
    return summary.strip()


def generate_eda_brief(
    llm,
    summary_text: str,
    logs_dir: Path,
    *,
    temperature: float = 0.2,
    force_rebuild: bool = False,
) -> str:
    """
    Distil ``eda_summary.txt`` into exactly two short list-style sentences → ``eda_brief.txt``.
    Intended for injection into CNN/Perch researcher system prompts (meta-agent step 2).
    """
    paths = _eda_paths(logs_dir)
    if paths["brief"].exists() and not force_rebuild:
        print("  EDA brief already exists — loading from cache.")
        return paths["brief"].read_text(encoding="utf-8")

    body = summary_text.strip()
    if not body and paths["summary"].exists():
        body = paths["summary"].read_text(encoding="utf-8")

    print("  Asking LLM to write 2-sentence EDA brief from summary …")
    brief = llm.generate_from_messages(
        messages=[
            {"role": "system", "content": _EDA_BRIEF_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _EDA_BRIEF_USER_TEMPLATE.format(
                    eda_summary=body[:6000],
                ),
            },
        ],
        temperature=temperature,
    )
    if not brief or brief.startswith("Error"):
        print("  LLM brief generation failed — using first two summary lines as fallback.")
        lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
        brief = ". ".join(lines[:2]) if lines else "Dataset EDA summary unavailable."

    _write_text_report(paths["brief"], "EDA Brief (2 sentences)", brief)
    print(f"  EDA brief saved → {paths['brief']}")
    return paths["brief"].read_text(encoding="utf-8")


def run_eda_phase(
    executor,
    llm,
    logs_dir: Path,
    temperature: float = 0.4,
    *,
    use_llm_codegen: bool = True,
    max_codegen_attempts: int = DEFAULT_EDA_CODEGEN_ATTEMPTS,
    max_codegen_wall_seconds: int = DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS,
    force_rebuild: bool = False,
    write_brief: bool = False,
    config: dict | None = None,
) -> str:
    """
    Run EDA: script → ``eda_raw_output.txt`` → ``eda_summary.txt``.

    Set ``write_brief=True`` to also call :func:`generate_eda_brief` (meta pipeline step 2).
    Returns the full summary text.
    """
    print("\n" + "=" * 60)
    print("  PHASE EDA: DATA EXPLORATION")
    print("=" * 60)

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    paths = _eda_paths(logs_dir)

    if paths["summary"].exists() and not force_rebuild:
        print(f"  EDA summary already exists → {paths['summary'].name}")
        summary_text = paths["summary"].read_text(encoding="utf-8")
        if write_brief:
            generate_eda_brief(llm, summary_text, logs_dir, temperature=temperature)
        print("=" * 60)
        return summary_text

    code_dir = logs_dir / "eda_code"
    script_path = code_dir / "eda_script.py"
    raw_output = _run_eda_script(
        executor,
        script_path,
        use_llm_codegen=use_llm_codegen,
        llm=llm,
        temperature=temperature,
        max_codegen_attempts=max_codegen_attempts,
        max_codegen_wall_seconds=max_codegen_wall_seconds,
        logs_dir=logs_dir,
        config=config,
    )
    paths["raw"].write_text(raw_output, encoding="utf-8")
    print(f"  Raw EDA output saved → {paths['raw'].name}")

    summary_body = _summarise_eda_raw(llm, raw_output, temperature=temperature)
    _write_text_report(paths["summary"], "EDA Summary", summary_body)
    # Legacy alias for older tooling
    paths["legacy_insights"].write_text(paths["summary"].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  EDA summary saved → {paths['summary']}")

    summary_text = paths["summary"].read_text(encoding="utf-8")
    if write_brief:
        generate_eda_brief(llm, summary_body, logs_dir, temperature=temperature)

    print("=" * 60)
    return summary_text


def load_eda_summary(logs_dir: Path) -> str:
    """Return ``eda_summary.txt`` if present."""
    p = Path(logs_dir) / EDA_SUMMARY_FILE
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_eda_brief(logs_dir: Path) -> str:
    """Return ``eda_brief.txt`` if present."""
    p = Path(logs_dir) / EDA_BRIEF_FILE
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_eda_insights(logs_dir: Path) -> str:
    """Legacy: prefer ``eda_summary.txt``, else ``eda_insights.txt``."""
    summary = load_eda_summary(logs_dir)
    if summary:
        return summary
    p = Path(logs_dir) / EDA_INSIGHTS_FILE
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> None:
    """CLI entry: ``python src/eda_agent.py --config configs/agent_config.json``"""
    import argparse

    parser = argparse.ArgumentParser(description="BirdCLEF EDA phase")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to agent_config.json (default: project configs/agent_config.json)",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory for eda_summary.txt (default: logs/eda)",
    )
    parser.add_argument(
        "--no-llm-codegen",
        action="store_true",
        help="Skip LLM EDA script generation; use hardcoded EDA_SCRIPT only",
    )
    parser.add_argument(
        "--write-brief",
        action="store_true",
        help="Also generate eda_brief.txt from the summary (2 sentences)",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore cached summary/brief")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else root / "configs" / "agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    logs_dir = Path(args.logs_dir) if args.logs_dir else root / "logs" / "eda"
    eda_cfg = config.get("eda") or {}
    meta_eda = (config.get("meta_agent") or {}).get("eda") or {}
    temperature = float(
        eda_cfg.get("temperature", config.get("llm", {}).get("temperature", 0.4))
    )
    use_llm_codegen = not args.no_llm_codegen and bool(
        eda_cfg.get("use_llm_codegen", meta_eda.get("use_llm_codegen", True))
    )
    max_codegen_attempts = int(
        eda_cfg.get(
            "max_codegen_attempts",
            meta_eda.get("max_codegen_attempts", DEFAULT_EDA_CODEGEN_ATTEMPTS),
        )
    )
    executor, llm, max_codegen_wall = build_eda_clients(config)

    run_eda_phase(
        executor,
        llm,
        logs_dir,
        temperature=temperature,
        use_llm_codegen=use_llm_codegen,
        max_codegen_attempts=max_codegen_attempts,
        max_codegen_wall_seconds=max_codegen_wall,
        force_rebuild=args.force_rebuild,
        write_brief=args.write_brief,
        config=config,
    )


if __name__ == "__main__":
    main()