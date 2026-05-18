"""
EDA phase for the BirdCLEF autonomous agent.

Pipeline:
  1. LLM coding agent writes EDA script (up to 5 attempts with stderr feedback), else ``EDA_SCRIPT``
  2. Save ``eda_raw_output.txt``
  3. LLM structured summary → ``eda_summary.txt`` (data facts only; no model architecture)
  4. (Step 2 integration) ``generate_eda_brief()`` → ``eda_brief.txt`` from summary only

The hardcoded ``EDA_SCRIPT`` is the reliable fallback when LLM codegen or execution fails.
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


# ─────────────────────────────────────────────────────────────────────────────
# LLM CODE GENERATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_EDA_CODEGEN_SYSTEM = """\
You are an autonomous ML research agent preparing to train a BirdCLEF+ 2026 model.
Before writing any model code you must explore the dataset yourself.
Write a self-contained Python EDA script that analyses the available data files
and prints structured findings to stdout.
Return ONLY a ```python``` code block — nothing else."""

_EDA_CODEGEN_USER = """\
You are about to start training a multi-label bird sound classifier for BirdCLEF+ 2026.
Before writing any model, explore the dataset to understand it.

Available data files (all under the `data/` folder relative to project root):
- data/train.csv               — metadata for every training recording (species label, filename, rating, lat/lon, collection, secondary_labels, ...)
- data/taxonomy.csv            — species taxonomy mapping (primary_label, scientific_name, common_name, class_name, ...)
- data/train_soundscapes_labels.csv — labeled segments from soundscape recordings (filename, start, end, primary_label)
- data/sample_submission.csv   — species columns used for evaluation / labeling alignment
- data/train_audio/            — directory of .ogg audio clips, one file per row in train.csv
- data/train_soundscapes/      — longer soundscape .ogg recordings

Write a Python script that:
1. Locates the project root by walking up from __file__ until a folder containing both `src/` and `data/` is found
2. Loads and inspects each CSV file (shape, columns, missing values)
3. Analyses the species/class distribution in train.csv (unique species, min/max/median recordings per species, taxonomic class breakdown, species with very few samples)
4. Checks the multi-label structure (fraction of recordings with secondary_labels, average count)
5. Checks recording quality signals (rating distribution, XC vs iNat split)
6. Counts audio files in train_audio/ and train_soundscapes/
7. Compares species in sample_submission.csv vs species with train_audio clips (count with zero focal examples)
8. Samples ~20 random audio files with librosa to report typical duration and sample rate
9. Identifies species present ONLY in soundscapes (not in train_audio)

Print every finding clearly with section headers using `print()`.
Do NOT mention model architecture, output dimensions, layer sizes, or num_classes.
Use only: pathlib, csv, collections, random, ast — plus librosa for audio sampling (wrap in try/except ImportError).
Do NOT use pandas, matplotlib, or any plotting library.
Do NOT produce any files or side effects — only print to stdout."""

_EDA_CODEGEN_FIX_USER = """\
Your previous EDA script failed when executed. Fix the code and return a complete corrected script.

## EXECUTION ERROR (attempt {attempt}/{max_attempts})
Exit code: {exit_code}
{timed_out_note}

### STDERR
{stderr}

### STDOUT (tail)
{stdout_tail}

## YOUR PREVIOUS CODE
```python
{previous_code}
```

## ORIGINAL TASK
{original_task}

Return ONLY a full corrected ```python``` code block (entire script). \
Do NOT mention model architecture or output dimensions. \
Print section headers with print(); no files or plots."""


DEFAULT_EDA_CODEGEN_ATTEMPTS = 5
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


def _run_llm_eda_codegen_with_retries(
    executor,
    script_path: Path,
    code_dir: Path,
    llm,
    *,
    temperature: float,
    max_attempts: int = DEFAULT_EDA_CODEGEN_ATTEMPTS,
    max_wall_seconds: int = DEFAULT_EDA_MAX_CODEGEN_WALL_SECONDS,
) -> str | None:
    """
    Ask the coding agent to write EDA Python, execute, and retry with stderr up to
    ``max_attempts`` times. Returns stdout on success, or None if all attempts fail.
    """
    max_attempts = max(1, int(max_attempts))
    max_wall_seconds = max(60, int(max_wall_seconds))
    script_cap = getattr(executor, "timeout_seconds", DEFAULT_EDA_SCRIPT_TIMEOUT_SECONDS)
    print(
        f"  EDA codegen agent: up to {max_attempts} attempt(s)  |  "
        f"{script_cap}s timeout per run  |  {max_wall_seconds}s total wall clock"
    )
    wall_start = time.time()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _EDA_CODEGEN_SYSTEM},
        {"role": "user", "content": _EDA_CODEGEN_USER},
    ]
    code: str | None = None

    for attempt in range(1, max_attempts + 1):
        elapsed_wall = time.time() - wall_start
        if elapsed_wall > max_wall_seconds:
            print(
                f"  [EDA codegen] Wall-clock budget exceeded ({elapsed_wall:.0f}s > {max_wall_seconds}s) "
                "— stopping retries."
            )
            break

        print(f"  [EDA codegen] Attempt {attempt}/{max_attempts} — asking LLM …")
        response = llm.generate_from_messages(
            messages=messages,
            temperature=temperature,
        )
        if _llm_response_failed(response):
            print(f"  [EDA codegen] LLM error: {str(response)[:200]}")
            messages.append(
                {
                    "role": "user",
                    "content": "Your last response was invalid. Return ONLY a ```python``` code block.",
                }
            )
            continue

        extracted = _extract_python_block(response)
        if not extracted:
            print("  [EDA codegen] No ```python``` block in response.")
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": "Return ONLY a complete EDA script inside a ```python``` fence.",
                }
            )
            continue

        code = extracted
        script_path.write_text(code, encoding="utf-8")
        if attempt == 1:
            (code_dir / "eda_script_llm.py").write_text(code, encoding="utf-8")

        print(f"  [EDA codegen] Running eda_script.py (attempt {attempt}) …")
        result, elapsed = _run_eda_script_once(executor, script_path)
        print(
            f"  [EDA codegen] Finished in {elapsed:.1f}s  "
            f"(exit={result.return_code}, stdout={len(result.stdout or '')} chars)"
        )

        if _eda_script_run_ok(result):
            print(f"  [EDA codegen] Success on attempt {attempt}.")
            return (result.stdout or "").strip()

        if result.timed_out:
            print(
                f"  [EDA codegen] Attempt {attempt} timed out after {script_cap}s "
                "(process killed — will ask agent to simplify or use fallback)."
            )

        stderr = (result.stderr or "").strip() or "(empty)"
        stdout_tail = (result.stdout or "")[-2500:] or "(empty)"
        print(f"  [EDA codegen] Failed — feeding error back to agent …")
        if len(stderr) > 400:
            print(f"    stderr tail: …{stderr[-400:]}")

        timed_out_note = (
            f"TIMED OUT after {script_cap}s — simplify the script so it finishes quickly.\n"
            if result.timed_out
            else ""
        )
        fix_user = _EDA_CODEGEN_FIX_USER.format(
            attempt=attempt,
            max_attempts=max_attempts,
            exit_code=result.return_code,
            timed_out_note=timed_out_note,
            stderr=stderr[:4000],
            stdout_tail=stdout_tail,
            previous_code=code[:12000],
            original_task=_EDA_CODEGEN_USER,
        )
        if result.timed_out:
            fix_user += _EDA_TIMEOUT_FIX_HINT
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": fix_user})

    print(f"  [EDA codegen] All {max_attempts} attempt(s) failed.")
    return None


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
) -> str:
    """Execute EDA script; LLM codegen with retries, then hardcoded fallback."""
    code_dir = script_path.parent
    code_dir.mkdir(parents=True, exist_ok=True)

    if use_llm_codegen:
        raw = _run_llm_eda_codegen_with_retries(
            executor,
            script_path,
            code_dir,
            llm,
            temperature=temperature,
            max_attempts=max_codegen_attempts,
            max_wall_seconds=max_codegen_wall_seconds,
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
    )


if __name__ == "__main__":
    main()