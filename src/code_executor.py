from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


class CodeExecutor:
    """Runs generated Python code in an isolated subprocess."""

    def __init__(self, python_executable: str = "python", timeout_seconds: int = 1800) -> None:
        self.python_executable = python_executable
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds is not None else 1800

    def run_file(
        self,
        script_path: Path,
        *,
        stream_output: bool = False,
        label: str = "",
    ) -> ExecutionResult:
        cmd = [self.python_executable, "-u", str(script_path)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if stream_output:
            return self._run_streaming(cmd, env=env, label=label)
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                env=env,
            )
            return ExecutionResult(
                success=completed.returncode == 0,
                stdout=completed.stdout,
                stderr=completed.stderr,
                return_code=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nExecution timed out after {self.timeout_seconds} seconds."
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr.strip(),
                return_code=-1,
                timed_out=True,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Executor failed before script completion: {exc}",
                return_code=-1,
            )

    def _run_streaming(self, cmd: list[str], *, env: dict, label: str) -> ExecutionResult:
        prefix = f"  [{label}] " if label else "  "
        start = time.time()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        timed_out = False
        try:
            while True:
                elapsed = time.time() - start
                if self.timeout_seconds > 0 and elapsed > self.timeout_seconds:
                    timed_out = True
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                    break
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    print(f"{prefix}{line}", end="", flush=True)
                elif proc.poll() is not None:
                    break
                else:
                    time.sleep(0.05)
            remainder = proc.stdout.read()
            if remainder:
                lines.append(remainder)
                print(f"{prefix}{remainder}", end="", flush=True)
        except Exception as exc:
            proc.kill()
            return ExecutionResult(
                success=False,
                stdout="".join(lines),
                stderr=f"Executor failed while streaming output: {exc}",
                return_code=-1,
            )

        return_code = proc.poll()
        if return_code is None:
            return_code = proc.wait()
        stdout = "".join(lines)
        if timed_out:
            stderr = (
                f"Execution timed out after {self.timeout_seconds} seconds "
                f"(elapsed {time.time() - start:.1f}s)."
            )
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                return_code=-1,
                timed_out=True,
            )
        return ExecutionResult(
            success=return_code == 0,
            stdout=stdout,
            stderr="",
            return_code=return_code,
        )
