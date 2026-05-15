"""
EDA Phase for the autonomous BirdCLEF agent.

Runs a hardcoded exploratory analysis over the raw data files, captures
the numerical output, and asks the LLM to distill it into a structured
eda_insights.txt that is then appended to every subsequent system prompt.
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

# ─── 8. SUBMISSION GAP ───────────────────────────────────────────────────────
_sep()
print("SECTION 8: SUBMISSION GAP (train vs submission species)")
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


# ─────────────────────────────────────────────────────────────────────────────
# LLM SUMMARISATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_EDA_SYSTEM_PROMPT = """\
You are an expert ML researcher summarizing exploratory data analysis results \
for a BirdCLEF+ 2026 multi-label bird-sound classification task. \
Your summary will be read by an autonomous training agent before it designs \
any model. Be concise, precise, and focus on actionable implications."""

_EDA_USER_TEMPLATE = """\
Below is the raw output from an automated EDA script run over the BirdCLEF+ 2026 dataset.
Summarize it into a structured INSIGHTS FILE with the sections listed below.
Write only factual statements grounded in the numbers shown — do not speculate.

## RAW EDA OUTPUT
{eda_output}

## REQUIRED SECTIONS IN YOUR SUMMARY

### 1. DATASET STRUCTURE
- Files present, row counts, key columns.
- Train audio clips count and species count.
- Soundscape files count.

### 2. CLASS DISTRIBUTION & IMBALANCE
- Total species, min/max/median recordings per species.
- Whether a long-tail imbalance exists and how severe.
- Taxonomic class breakdown (birds vs insects etc.).

### 3. MULTI-LABEL COMPLEXITY
- Fraction of recordings with secondary labels.
- Implications for loss function choice.

### 4. SUBMISSION GAP
- How many species are required in the submission but have ZERO training audio clips.
- What strategy this implies (e.g. rely on soundscapes, augmentation, or zero-shot).

### 5. DATA QUALITY SIGNALS
- Rating distribution (zero-rated recordings = iNat, unverified quality).
- Recording source bias (XC vs iNat split).

### 6. AUDIO CHARACTERISTICS
- Typical clip duration and sample rate.
- Whether the 5-second clip assumption from the competition rules holds.

### 7. KEY CONSTRAINTS FOR MODEL DESIGN
Bullet list of the most important facts the training agent MUST keep in mind \
(class imbalance handling, submission gap species, multi-label output head size, etc.).

Write the summary now. Start directly with '### 1. DATASET STRUCTURE'."""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_eda_phase(
    executor,
    llm,
    logs_dir: Path,
    temperature: float = 0.4,
) -> str:
    """
    Run the autonomous EDA phase.

    Executes the hardcoded EDA script, feeds the output to the LLM for
    summarisation, writes logs/eda_insights.txt, and returns the insight text
    so callers can inject it into subsequent system prompts.
    """
    print("\n" + "=" * 60)
    print("  PHASE EDA: AUTONOMOUS DATA EXPLORATION")
    print("=" * 60)

    insights_path = logs_dir / "eda_insights.txt"

    # ── skip if already done ──────────────────────────────────────────────────
    if insights_path.exists():
        print("  EDA insights already exist — loading from cache.")
        return insights_path.read_text(encoding="utf-8")

    # ── write the EDA script to a temp file and execute ───────────────────────
    code_dir = logs_dir / "eda_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script_path = code_dir / "eda_script.py"
    script_path.write_text(EDA_SCRIPT, encoding="utf-8")

    print("  Running EDA script …")
    t0 = time.time()
    result = executor.run_file(script_path)
    elapsed = time.time() - t0
    print(f"  EDA script finished in {elapsed:.1f}s  (exit={result.return_code})")

    raw_output = result.stdout or ""
    stderr     = result.stderr or ""

    if not raw_output.strip():
        raw_output = f"[EDA script produced no stdout]\nSTDERR:\n{stderr[:2000]}"

    # Save raw output for debugging
    raw_path = logs_dir / "eda_raw_output.txt"
    raw_path.write_text(raw_output, encoding="utf-8")
    print(f"  Raw EDA output saved → {raw_path.name}")

    # ── ask LLM to summarise into structured insights ─────────────────────────
    print("  Asking LLM to summarise EDA findings …")
    truncated_output = raw_output[:8000]  # stay within context
    user_msg = _EDA_USER_TEMPLATE.format(eda_output=truncated_output)

    insights = llm.generate_from_messages(
        messages=[
            {"role": "system", "content": _EDA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=temperature,
    )

    if not insights or insights.startswith("Error"):
        # Fallback: use the raw output directly as insights
        print("  LLM summarisation failed — using raw output as fallback.")
        insights = raw_output

    # ── write insights file ───────────────────────────────────────────────────
    header = (
        "# BirdCLEF+ 2026 — Agent EDA Insights\n"
        f"# Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n"
        "# This file was written autonomously by the agent before any model training.\n\n"
    )
    full_text = header + insights.strip()
    insights_path.write_text(full_text, encoding="utf-8")
    print(f"  EDA insights saved → {insights_path}")
    print("=" * 60)

    return full_text


def load_eda_insights(logs_dir: Path) -> str:
    """Return the EDA insights text if available, else empty string."""
    p = logs_dir / "eda_insights.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> None:
    import argparse
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "agent_config.json"))
    args   = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    sys.path.insert(0, str(ROOT / "src"))
    if __package__:
        from .code_executor import CodeExecutor
        from .llm_client import LLMClient
    else:
        from code_executor import CodeExecutor  # type: ignore
        from llm_client import LLMClient        # type: ignore

    provider    = config.get("llm", {}).get("provider", "ollama")
    model       = config.get("llm", {}).get("model", "llama3.2:3b")
    temperature = float(config.get("llm", {}).get("temperature", 0.4))
    py_exe      = config.get("execution", {}).get("python_executable", sys.executable)
    timeout     = int(config.get("execution", {}).get("timeout_seconds", 1800))

    logs_dir = ROOT / "logs" / "eda"
    logs_dir.mkdir(parents=True, exist_ok=True)

    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    llm      = LLMClient(provider=provider, model=model)

    run_eda_phase(executor, llm, logs_dir, temperature=temperature)


if __name__ == "__main__":
    main()
