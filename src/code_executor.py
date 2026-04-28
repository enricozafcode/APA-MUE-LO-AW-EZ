from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def _decode_output(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data


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
        timeout_seconds: int | None = 1800,
    ) -> None:
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def run_file(self, script_path: Path) -> ExecutionResult:
        try:
            kwargs: dict = {
                "capture_output": True,
                "text": True,
            }
            if self.timeout_seconds is not None:
                kwargs["timeout"] = self.timeout_seconds
            completed = subprocess.run(
                [self.python_executable, str(script_path)],
                **kwargs,
            )
            return ExecutionResult(
                success=completed.returncode == 0,
                stdout=_decode_output(completed.stdout),
                stderr=_decode_output(completed.stderr),
                return_code=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
            msg = f"\nExecution timed out after {self.timeout_seconds} seconds."
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=(stderr + msg).strip(),
                return_code=-1,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Executor failed before script completion: {exc}",
                return_code=-1,
            )
