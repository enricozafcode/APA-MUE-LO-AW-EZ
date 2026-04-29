from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int


class CodeExecutor:
    """Runs generated Python code in an isolated subprocess."""

    def __init__(
        self,
        python_executable: str = "python",
        timeout_seconds: int = 1800,
        extra_pythonpath: list[str] | None = None,
    ) -> None:
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.extra_pythonpath = extra_pythonpath or []

    def run_file(self, script_path: Path) -> ExecutionResult:
        env = os.environ.copy()
        if self.extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            added = os.pathsep.join(self.extra_pythonpath)
            env["PYTHONPATH"] = f"{added}{os.pathsep}{existing}" if existing else added

        completed = subprocess.run(
            [self.python_executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        return ExecutionResult(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )
