"""
Experiment memory for the Perch agent.

- Full history stays in experiment_memory.jsonl (append-only audit log).
- Researcher sees a lean run list: arch_type, arch_description, and scores only (~30 runs).
- Optional LLM batch summaries (off by default); rule-based digest is optional legacy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from memory import ExperimentMemory

SUMMARY_BATCH_SIZE = 3
CONFIDENCE_FULL_TRIES = 20

_SPEC_KEYS_COMPACT = (
    "arch_type",
    "hidden_dim",
    "proj_dim",
    "n_layers",
    "dropout",
    "activation",
    "normalization",
    "learning_rate",
    "batch_size",
    "optimizer",
    "epochs",
    "patience",
    "perch_weight",
    "strategy",
)

_SUMMARIZER_SYSTEM = """You compress Perch head architecture search logs into short structured insights.
Respond with ONLY one JSON object (no markdown). Keys:
  "batch_insight": one sentence on what this batch of runs showed overall,
  "arch_insights": list of objects, each with:
    - "arch_type": string (required)
    - "insight": string, max 25 words — best hyperparams, success/failure, ranking score
    - "hyperparams_short": string like "hidden512|ln|gelu|drop0.3|lr1e-3|pw0.2"
Merge duplicate arch_types in this batch into one insight each."""


class PerchExperimentMemory(ExperimentMemory):
    """ExperimentMemory with rolling LLM summaries for the Perch researcher."""

    DIGEST_FILE = "memory_digest.json"

    def __init__(
        self,
        logs_dir: Path,
        ranking_metric: str = "macro_average_precision",
        *,
        researcher_history_max_runs: int | None = None,
        summary_batch_size: int = SUMMARY_BATCH_SIZE,
        confidence_full_tries: int = CONFIDENCE_FULL_TRIES,
    ) -> None:
        super().__init__(
            logs_dir,
            ranking_metric=ranking_metric,
            researcher_history_max_runs=researcher_history_max_runs,
        )
        self.digest_path = Path(logs_dir) / self.DIGEST_FILE
        self.summary_batch_size = max(1, int(summary_batch_size))
        self.confidence_full_tries = max(1, int(confidence_full_tries))
        self.use_llm_summaries = False
        self._summarizer_llm = None
        self._digest: dict[str, Any] = self._load_digest()

    def configure_summaries(self, *, use_llm: bool, llm=None) -> None:
        """Rule-based digest is default; LLM compression is optional (slow on R1 models)."""
        self.use_llm_summaries = bool(use_llm)
        self._summarizer_llm = llm if use_llm else None

    def attach_summarizer(self, llm) -> None:
        """Legacy: enables LLM summaries. Prefer configure_summaries(use_llm=..., llm=...)."""
        self.configure_summaries(use_llm=True, llm=llm)

    def seed_from_stage_1a(
        self,
        parent_mem_dir: Path,
        *,
        arch_type: str,
        aug_baseline: str,
        seed_score: float,
        seed_spec: dict,
    ) -> None:
        """Bootstrap refine-memory from a stage-1a winner (compact parent digest + seed champion)."""
        parent_digest_path = Path(parent_mem_dir) / self.DIGEST_FILE
        parent_reg: dict[str, Any] = {}
        parent_insights: list[str] = []
        if parent_digest_path.exists():
            try:
                pd = json.loads(parent_digest_path.read_text(encoding="utf-8"))
                parent_reg = pd.get("arch_registry") or {}
                parent_insights = list(pd.get("global_insights") or [])[-6:]
            except (json.JSONDecodeError, OSError):
                pass

        reg = self._digest.setdefault("arch_registry", {})
        if arch_type in parent_reg:
            reg[arch_type] = dict(parent_reg[arch_type])
        else:
            reg[arch_type] = {
                "n_tries": 0,
                "n_success": 0,
                "best_ranking_value": float(seed_score),
                "hyperparams_short": self._hyperparams_short(seed_spec),
                "insight": f"Stage-1a champion from aug={aug_baseline}",
            }

        gi = self._digest.setdefault("global_insights", [])
        gi.append(
            f"Refine campaign seeded from 1a/{aug_baseline}: {arch_type} "
            f"@ {self.ranking_metric}={float(seed_score):.4f}"
        )
        for line in parent_insights:
            if arch_type in line or aug_baseline in line:
                gi.append(f"[1a context] {line}")
        self._digest["global_insights"] = gi[-12:]

        self._digest["refine_seed"] = {
            "parent_memory_dir": str(parent_mem_dir),
            "aug_baseline": aug_baseline,
            "arch_type": arch_type,
            "seed_score": float(seed_score),
            "seed_spec": self._compact_spec(seed_spec),
            "seed_hyperparams": self._hyperparams_short(seed_spec),
            "champion_source": "stage_1a_seed",
        }
        self._update_digest_best(self._digest)
        self._save_digest()

    def sync_refine_champion(
        self,
        *,
        arch_type: str,
        aug_baseline: str,
        seed_score: float,
        seed_spec: dict,
        parent_memory_dir: str | None = None,
        champion_source: str = "resolved",
    ) -> None:
        """Set refine champion banner (best model found at campaign start)."""
        self._digest["refine_seed"] = {
            "parent_memory_dir": parent_memory_dir,
            "aug_baseline": aug_baseline,
            "arch_type": arch_type,
            "seed_score": float(seed_score),
            "seed_spec": self._compact_spec(seed_spec),
            "seed_hyperparams": self._hyperparams_short(seed_spec),
            "champion_source": champion_source,
        }
        self._update_digest_best(self._digest)
        self._save_digest()

    def log(self, *, spec: dict, metrics: dict | None, code: str = "") -> None:
        super().log(spec=spec, metrics=metrics, code=code)
        self._update_digest_best()
        if self.use_llm_summaries:
            self._catch_up_summaries()

    def _empty_digest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "ranking_metric": self.ranking_metric,
            "summarized_run_count": 0,
            "arch_registry": {},
            "batch_summaries": [],
            "global_insights": [],
            "best_snapshot": None,
        }

    def _load_digest(self) -> dict[str, Any]:
        if not self.digest_path.exists():
            d = self._empty_digest()
            self._update_digest_best(d)
            self._save_digest(d)
            return d
        try:
            d = json.loads(self.digest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            d = self._empty_digest()
        d.setdefault("arch_registry", {})
        d.setdefault("batch_summaries", [])
        d.setdefault("global_insights", [])
        d.setdefault("summarized_run_count", 0)
        return d

    def _save_digest(self, digest: dict | None = None) -> None:
        d = digest if digest is not None else self._digest
        self.digest_path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @staticmethod
    def _compact_spec(spec: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in _SPEC_KEYS_COMPACT:
            if k in spec and spec[k] is not None:
                out[k] = spec[k]
        return out

    @staticmethod
    def _hyperparams_short(spec: dict) -> str:
        parts = []
        if spec.get("hidden_dim") is not None:
            parts.append(f"h{spec['hidden_dim']}")
        if spec.get("proj_dim") is not None:
            parts.append(f"p{spec['proj_dim']}")
        if spec.get("n_layers") is not None:
            parts.append(f"L{spec['n_layers']}")
        norm = spec.get("normalization")
        if norm:
            parts.append(str(norm)[:6])
        act = spec.get("activation")
        if act:
            parts.append(str(act)[:4])
        if spec.get("dropout") is not None:
            parts.append(f"drop{spec['dropout']}")
        if spec.get("learning_rate") is not None:
            parts.append(f"lr{spec['learning_rate']}")
        if spec.get("perch_weight") is not None:
            parts.append(f"pw{spec['perch_weight']}")
        if spec.get("optimizer"):
            parts.append(str(spec["optimizer"])[:5])
        return "|".join(parts) if parts else "default"

    def _run_compact(self, entry: dict, run_index: int) -> dict[str, Any]:
        spec = entry.get("spec") or {}
        return {
            "run": run_index,
            "arch_type": spec.get("arch_type", "?"),
            "hyperparams": self._hyperparams_short(spec),
            "spec": self._compact_spec(spec),
            "success": bool(entry.get("success")),
            "ranking_value": self._ranking_value(entry) if entry.get("success") else None,
            "macro_ap": entry.get("macro_average_precision"),
            "macro_auc": entry.get("macro_roc_auc"),
            "train_loss": entry.get("train_loss"),
            "val_loss": entry.get("val_loss"),
        }

    def _confidence(self, total_tries: int) -> tuple[float, str]:
        w = min(1.0, total_tries / float(self.confidence_full_tries))
        if w >= 0.75:
            tier = "HIGH"
        elif w >= 0.35:
            tier = "MEDIUM"
        else:
            tier = "LOW"
        return w, tier

    def _update_digest_best(self, digest: dict | None = None) -> None:
        d = digest if digest is not None else self._digest
        best = self.best_runs(1)
        total = self.total()
        if not best:
            d["best_snapshot"] = None
        else:
            r = best[0]
            spec = r.get("spec") or {}
            w, tier = self._confidence(total)
            d["best_snapshot"] = {
                "arch_type": spec.get("arch_type"),
                "hyperparams_short": self._hyperparams_short(spec),
                "spec_compact": self._compact_spec(spec),
                "ranking_metric": self.ranking_metric,
                "ranking_value": self._ranking_value(r),
                "macro_ap": r.get("macro_average_precision"),
                "macro_auc": r.get("macro_roc_auc"),
                "total_tries": total,
                "confidence_weight": round(w, 3),
                "confidence_tier": tier,
            }
        if digest is None:
            self._save_digest()

    def _unsummarized_runs(self) -> list[tuple[int, dict]]:
        start = int(self._digest.get("summarized_run_count", 0))
        out: list[tuple[int, dict]] = []
        for i, r in enumerate(self._runs[start:], start=start + 1):
            out.append((i, r))
        return out

    def _catch_up_summaries(self) -> None:
        pending = self._unsummarized_runs()
        while len(pending) >= self.summary_batch_size:
            batch = pending[: self.summary_batch_size]
            self._summarize_batch(batch)
            pending = self._unsummarized_runs()

    def _merge_arch_registry(self, arch_insights: list[dict]) -> None:
        reg = self._digest.setdefault("arch_registry", {})
        for item in arch_insights:
            at = item.get("arch_type")
            if not at:
                continue
            prev = reg.get(at, {"n_tries": 0, "n_success": 0, "best_ranking_value": -1.0})
            rv = item.get("best_ranking_value")
            if rv is None:
                rv = prev.get("best_ranking_value", -1.0)
            try:
                rv_f = float(rv)
            except (TypeError, ValueError):
                rv_f = prev.get("best_ranking_value", -1.0)
            n_try = int(prev.get("n_tries", 0)) + int(item.get("n_tries", 1))
            n_ok = int(prev.get("n_success", 0)) + int(item.get("n_success", 0))
            best_prev = float(prev.get("best_ranking_value", -1.0))
            reg[at] = {
                "n_tries": n_try,
                "n_success": n_ok,
                "best_ranking_value": max(best_prev, rv_f),
                "hyperparams_short": item.get("hyperparams_short") or prev.get("hyperparams_short", ""),
                "insight": item.get("insight") or prev.get("insight", ""),
            }

    def _deterministic_batch_summary(
        self, batch: list[tuple[int, dict]]
    ) -> dict[str, Any]:
        by_arch: dict[str, list[dict]] = {}
        for _idx, r in batch:
            at = (r.get("spec") or {}).get("arch_type", "unknown")
            by_arch.setdefault(at, []).append(r)

        arch_insights: list[dict] = []
        for at, runs in by_arch.items():
            ok = [x for x in runs if x.get("success")]
            scores = [self._ranking_value(x) for x in ok]
            best = max(scores) if scores else -1.0
            best_run = ok[scores.index(best)] if ok else runs[0]
            hp = self._hyperparams_short(best_run.get("spec") or {})
            arch_insights.append({
                "arch_type": at,
                "n_tries": len(runs),
                "n_success": len(ok),
                "best_ranking_value": best if ok else None,
                "hyperparams_short": hp,
                "insight": (
                    f"{len(ok)}/{len(runs)} ok; best {self.ranking_metric}={best:.4f} @ {hp}"
                    if ok
                    else f"0/{len(runs)} failed"
                ),
            })
        return {
            "batch_insight": (
                f"Batch of {len(batch)} runs across {len(by_arch)} arch type(s); "
                f"best {self.ranking_metric}="
                f"{max((self._ranking_value(r) for _, r in batch if r.get('success')), default=-1):.4f}"
            ),
            "arch_insights": arch_insights,
        }

    def _parse_summary_json(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        raw = m.group(1) if m else cleaned
        start = raw.find("{")
        if start < 0:
            return None
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _llm_batch_summary(self, batch: list[tuple[int, dict]]) -> dict[str, Any] | None:
        if self._summarizer_llm is None:
            return None
        compact = [self._run_compact(r, i) for i, r in batch]
        user = (
            f"Summarize these {len(batch)} Perch head experiments. "
            f"Ranking metric: {self.ranking_metric} (higher is better).\n"
            f"Runs JSON:\n{json.dumps(compact, indent=2)}\n\n"
            "If multiple runs share arch_type, merge into one arch_insights entry."
        )
        resp = self._summarizer_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": _SUMMARIZER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            num_predict=450,
            stream_debug=False,
        )
        if resp.startswith("Error communicating"):
            return None
        return self._parse_summary_json(resp)

    def _summarize_batch(self, batch: list[tuple[int, dict]]) -> None:
        run_ids = [i for i, _ in batch]
        summary = None
        if self.use_llm_summaries:
            summary = self._llm_batch_summary(batch)
        if not summary:
            summary = self._deterministic_batch_summary(batch)
            if self.use_llm_summaries:
                print(
                    f"  [Memory] Batch summary rule-based (runs {run_ids[0]}–{run_ids[-1]})."
                )
            else:
                print(
                    f"  [Memory] Batch digest updated (runs {run_ids[0]}–{run_ids[-1]}, no LLM)."
                )
        else:
            print(f"  [Memory] Batch summary LLM OK (runs {run_ids[0]}–{run_ids[-1]}).")

        arch_raw = summary.get("arch_insights") or []
        arch_norm: list[dict] = []
        by_arch: dict[str, list[tuple[int, dict]]] = {}
        for i, r in batch:
            at = (r.get("spec") or {}).get("arch_type", "unknown")
            by_arch.setdefault(at, []).append((i, r))

        for item in arch_raw:
            at = item.get("arch_type")
            if at and at in by_arch:
                runs = [r for _, r in by_arch[at]]
                ok = [x for x in runs if x.get("success")]
                scores = [self._ranking_value(x) for x in ok]
                best = max(scores) if scores else None
                arch_norm.append({
                    "arch_type": at,
                    "n_tries": len(runs),
                    "n_success": len(ok),
                    "best_ranking_value": best,
                    "hyperparams_short": item.get("hyperparams_short")
                    or self._hyperparams_short((ok[scores.index(best)] if ok else runs[0]).get("spec") or {}),
                    "insight": (item.get("insight") or "")[:200],
                })
        for at, runs_t in by_arch.items():
            if any(x.get("arch_type") == at for x in arch_norm):
                continue
            runs = [r for _, r in runs_t]
            arch_norm.extend(self._deterministic_batch_summary(
                [(0, r) for r in runs]
            )["arch_insights"])

        self._merge_arch_registry(arch_norm)
        batch_insight = (summary.get("batch_insight") or "")[:300]
        batches = self._digest.setdefault("batch_summaries", [])
        batches.append({
            "runs": run_ids,
            "batch_insight": batch_insight,
            "arch_insights": arch_norm,
        })
        gi = self._digest.setdefault("global_insights", [])
        if batch_insight:
            gi.append(batch_insight)
        self._digest["global_insights"] = gi[-12:]

        self._digest["summarized_run_count"] = run_ids[-1]
        self._update_digest_best(self._digest)
        self._save_digest()

    _DESC_MAX_CHARS = 220

    def _scores_line(self, entry: dict) -> str:
        if not entry.get("success"):
            return "FAILED"
        from soundscape_evaluator import format_metrics_dict

        m = entry.get("metrics") or {}
        return format_metrics_dict(
            {
                "status": "success",
                "macro_average_precision": entry.get("macro_average_precision")
                or m.get("macro_average_precision"),
                "macro_roc_auc": entry.get("macro_roc_auc") or m.get("macro_roc_auc"),
                "median_per_class_auc": entry.get("median_per_class_auc")
                or m.get("median_per_class_auc"),
                "train_loss": entry.get("train_loss") or m.get("train_loss"),
                "val_loss": entry.get("val_loss") or m.get("val_loss"),
            },
            ranking_metric=self.ranking_metric,
        )

    def _run_researcher_line(self, entry: dict, run_index: int) -> list[str]:
        spec = entry.get("spec") or {}
        arch = spec.get("arch_type", "?")
        desc = (spec.get("arch_description") or "").strip().replace("\n", " ")
        if len(desc) > self._DESC_MAX_CHARS:
            desc = desc[: self._DESC_MAX_CHARS - 1] + "…"
        slot = spec.get("slot")
        slot_s = f" slot={slot}" if slot else ""
        hp = self._hyperparams_short(spec)
        head = f"#{run_index} | {arch}{slot_s} | {self._scores_line(entry)}"
        lines = [head, f"  hyperparams: {hp}"]
        if desc:
            lines.append(f"  description: {desc}")
        else:
            lines.append("  description: (none logged)")
        reason = (entry.get("reasoning") or spec.get("reasoning") or "").strip()
        hyp = (entry.get("hypothesis") or spec.get("hypothesis") or "").strip()
        cap = self._RATIONALE_MAX_CHARS
        if reason:
            r = reason if len(reason) <= cap else reason[: cap - 1] + "…"
            lines.append(f"  reasoning: {r}")
        if hyp:
            h = hyp if len(hyp) <= cap else hyp[: cap - 1] + "…"
            lines.append(f"  hypothesis: {h}")
        return lines

    def champion_context_block(
        self,
        *,
        locked_arch_type: str,
        seed_spec: dict,
        seed_score: float | None,
    ) -> str:
        """Always-visible 1a champion config for stage-1b refine prompts."""
        spec = dict(seed_spec or {})
        desc = (spec.get("arch_description") or "").strip().replace("\n", " ")
        if len(desc) > 280:
            desc = desc[:279] + "…"
        seed = self._digest.get("refine_seed") or {}
        src = seed.get("champion_source", "memory")
        lines = [
            "REFINE CHAMPION (best model at campaign start — LOCKED arch_type; beat this score):",
            f"  arch_type: {locked_arch_type}",
            f"  source: {src}",
        ]
        if seed_score is not None:
            lines.append(
                f"  champion {self.ranking_metric}: {float(seed_score):.5f} "
                f"(floor to beat for bonus tries)"
            )
        lines.append(f"  champion hyperparams: {self._hyperparams_short(spec)}")
        if desc:
            lines.append(f"  champion description: {desc}")
        parent = seed.get("parent_memory_dir")
        if parent:
            lines.append(f"  memory dir: {parent}")
        return "\n".join(lines)

    def _refine_champion_spec_for_prompt(self) -> dict:
        """Full champion spec (file beats compact digest)."""
        for name in ("refine_champion_spec.json", "stage_1a_champion_spec.json"):
            champ_path = self.path.parent / name
            if not champ_path.exists():
                continue
            try:
                payload = json.loads(champ_path.read_text(encoding="utf-8"))
                spec = dict(payload.get("spec") or {})
                if payload.get("locked_arch_type"):
                    spec["arch_type"] = payload["locked_arch_type"]
                if spec:
                    return spec
            except (json.JSONDecodeError, OSError):
                pass
        seed = self._digest.get("refine_seed") or {}
        return dict(seed.get("seed_spec") or {})

    def researcher_context(self, *, max_runs: int | None = None) -> str:
        """Lean history for the planner: arch_type, description, results, prior LLM rationale."""
        self._update_digest_best()

        total = self.total()
        seed = self._digest.get("refine_seed")
        if total == 0 and not seed:
            return "No experiments have been run yet. This is the very first iteration."

        ok = self.successful_runs()
        fails = self.failed_runs()
        best = self.best_runs(1)

        lines: list[str] = []
        if seed:
            champ_spec = self._refine_champion_spec_for_prompt()
            lines.append(
                self.champion_context_block(
                    locked_arch_type=str(
                        seed.get("arch_type", champ_spec.get("arch_type", "?"))
                    ),
                    seed_spec=champ_spec,
                    seed_score=seed.get("seed_score"),
                )
            )
            lines.append("")

        lines.extend([
            "EXPERIMENT LOG (arch_type + hyperparams + description + results; full jsonl on disk)",
            f"Runs: {total} ({len(ok)} ok, {len(fails)} failed) | "
            f"optimize: {self.ranking_metric} (also logged: macro_AP, macro_AUC, "
            f"median_AUC, train_loss, val_loss)",
            "",
        ])

        if best:
            b = best[0]
            bs = (b.get("spec") or {}).get("arch_type", "?")
            lines.append(f"BEST SO FAR: {bs} | {self._format_run_score(b)}")
            lines.append("")

        cap = max_runs if max_runs is not None else self.researcher_history_max_runs
        show = self._runs[-cap:] if cap and total > cap else self._runs
        start_idx = total - len(show) + 1
        cap_note = f" (capped to last {cap})" if cap and total > cap else ""
        lines.append(f"RUNS (chronological, {len(show)} shown{cap_note}):")
        for offset, entry in enumerate(show):
            lines.extend(self._run_researcher_line(entry, start_idx + offset))

        tried = sorted({
            (r.get("spec") or {}).get("arch_type", "?")
            for r in self._runs
            if (r.get("spec") or {}).get("arch_type")
        })
        if tried:
            lines.append("")
            lines.append(f"arch_types seen ({len(tried)}): {', '.join(tried)}")

        return "\n".join(lines)
