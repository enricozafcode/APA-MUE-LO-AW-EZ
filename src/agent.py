from __future__ import annotations

import ast
import json
from pathlib import Path

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from augmentation import build_augmenters


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(active_audio: list[str], active_spec: list[str]) -> str:
    return (
        "You are an expert ML engineer working on BirdCLEF audio bird classification (Track B).\n"
        "Your task is to write complete, runnable Python training scripts. "
        "Output raw Python code only — no markdown, no explanations, no ```python blocks.\n\n"
        "## Augmentation module\n"
        "An augmentation module is available at src/augmentation.py. "
        "Always import and use it for training data — never apply it to validation or test data.\n\n"
        "Usage:\n"
        "    from augmentation import build_augmenters\n"
        "    audio_aug, spec_aug = build_augmenters(config)\n"
        "    # In your training data loader:\n"
        "    audio = audio_aug.apply(audio, sr)          # before mel conversion\n"
        "    spec  = spec_aug.apply(mel_spectrogram)     # after mel conversion\n"
        "    # For mixup (at batch level, if enabled):\n"
        "    spec, label = spec_aug.mixup(s1, l1, s2, l2)\n\n"
        f"## Currently active augmentations\n"
        f"Audio:       {active_audio if active_audio else 'none'}\n"
        f"Spectrogram: {active_spec if active_spec else 'none'}\n\n"
        "## Output requirement\n"
        "At the end of training, print the validation cmAP score in this exact format:\n"
        "    cmAP: <value>   (e.g. cmAP: 0.4521)\n"
        "This is required for the evaluation loop to track progress."
    )


def strip_markdown(raw: str) -> str:
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def generate_with_syntax_fix(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 3,
) -> str:
    """Generate code and retry with the syntax error if the LLM produces invalid Python."""
    current_prompt = user_prompt
    for attempt in range(1, max_retries + 1):
        print(f"  Generation attempt {attempt}/{max_retries}...")
        raw = llm.generate_code(system_prompt=system_prompt, user_prompt=current_prompt, temperature=temperature)
        code = strip_markdown(raw)
        try:
            ast.parse(code)
            print("  Syntax check passed.")
            return code
        except SyntaxError as e:
            print(f"  Syntax error on line {e.lineno}: {e.msg}")
            if attempt < max_retries:
                current_prompt = (
                    f"The code you generated has a syntax error:\n"
                    f"  Line {e.lineno}: {e.msg}\n"
                    f"  Near: {e.text}\n\n"
                    f"Here is the broken code:\n{code}\n\n"
                    "Fix the syntax error and return the complete corrected script. "
                    "Raw Python only — no markdown, no explanations."
                )
    print("  Warning: could not produce valid syntax after all retries. Using last attempt.")
    return code


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "agent_config.json"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("Loading configuration...")
    config = load_config(config_path)

    print(f"Initializing LLM Client with model: {config['llm']['model']}...")
    llm = LLMClient(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
    )

    src_dir = str(Path(__file__).resolve().parent)
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=config["execution"]["timeout_seconds"],
        extra_pythonpath=[src_dir],
    )

    evaluator = Evaluator()

    audio_aug, spec_aug = build_augmenters(config)
    active_audio = audio_aug.active_strategies()
    active_spec = spec_aug.active_strategies()
    print(f"Active audio augmentations:       {active_audio}")
    print(f"Active spectrogram augmentations: {active_spec}")

    system_prompt = build_system_prompt(active_audio, active_spec)

    user_prompt = (
        "Write a BirdCLEF Track B training script that:\n"
        "1. Loads audio files and converts them to mel spectrograms\n"
        "2. Applies augmentation (from augmentation.py) to training data only\n"
        "3. Trains a simple CNN classifier\n"
        "4. Evaluates on validation data and prints: cmAP: <value>\n"
        "Use the currently active augmentation strategies shown in the system prompt."
    )

    scaffold = project_root / "src" / "train_scaffold.py"
    print(f"\n--- EXECUTING TRAINING SCAFFOLD ---")
    result = executor.run_file(scaffold)

    print("\n--- EXECUTION RESULTS ---")
    print(f"Success:     {result.success}")
    print(f"Return Code: {result.return_code}")
    print("\nStandard Output:")
    print(result.stdout if result.stdout else "(None)")
    if result.stderr:
        print("\nStandard Error:")
        print(result.stderr)

    summary = evaluator.build_summary(
        stdout=result.stdout,
        stderr=result.stderr,
        active_audio_strategies=active_audio,
        active_spec_strategies=active_spec,
    )

    print("\n--- EVALUATION ---")
    print(f"Metrics: {summary.metrics}")

    if config["logging"].get("save_metrics_json"):
        metrics_file = logs_dir / "metrics.json"
        metrics_file.write_text(json.dumps(summary.metrics, indent=2), encoding="utf-8")
        print(f"Metrics saved to {metrics_file}")

    print("\n--- NEXT EXPERIMENT SUGGESTION ---")
    print(summary.analysis_prompt)


if __name__ == "__main__":
    main()
