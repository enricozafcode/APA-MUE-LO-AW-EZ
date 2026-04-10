from __future__ import annotations

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

    def __init__(self, python_executable: str = "python", timeout_seconds: int = 1800) -> None:
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def run_file(self, script_path: Path) -> ExecutionResult:
        completed = subprocess.run(
            [self.python_executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        return ExecutionResult(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )
