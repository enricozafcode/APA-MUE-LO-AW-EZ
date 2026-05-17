import json
import sys

import requests


def llm_response_failed(response: str | None) -> bool:
    """True when the client gave up (timeout, connection error, empty body)."""
    if not response or not str(response).strip():
        return True
    return str(response).strip().startswith("Error communicating with local LLM")


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Installed Ollama model names (empty if daemon unreachable)."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        return [str(m.get("name", "")) for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def resolve_ollama_model(
    requested: str,
    *,
    base_url: str = "http://localhost:11434",
) -> tuple[str | None, str]:
    """
    Return (model_name, error). Exact match first, then name before ':' tag.
    """
    name = (requested or "").strip()
    if not name:
        return None, "No model name configured."
    installed = list_ollama_models(base_url)
    if not installed:
        return None, (
            "Ollama is not reachable at "
            f"{base_url} (is `ollama serve` running?)."
        )
    if name in installed:
        return name, ""
    base = name.split(":", 1)[0]
    for candidate in installed:
        if candidate.split(":", 1)[0] == base:
            return candidate, ""
    return None, (
        f"Model {name!r} is not installed. Available: {', '.join(sorted(installed))}. "
        f"Run: ollama pull {name}"
    )


class LLMClient:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2:3b",
        timeout_seconds: float = 600,
        *,
        stream_debug: bool = False,
    ):
        self.model_name = model
        self.timeout_seconds = float(timeout_seconds)
        self.stream_debug = bool(stream_debug)
        self._is_ollama = provider.lower() == "ollama"
        if self._is_ollama:
            self.base_url = "http://localhost:11434/api/chat"
        else:
            self.base_url = "http://localhost:1234/v1/chat/completions"

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        format_json: bool = False,
        num_predict: int | None = None,
    ) -> str:
        if self.stream_debug and self._is_ollama:
            return self._generate_ollama_stream(
                messages, temperature, format_json=format_json, num_predict=num_predict
            )
        return self._generate_blocking(
            messages, temperature, format_json=format_json, num_predict=num_predict
        )

    def _ollama_options(
        self,
        temperature: float,
        *,
        format_json: bool = False,
        num_predict: int | None = None,
    ) -> dict:
        opts: dict = {"temperature": temperature}
        if format_json and self._is_ollama:
            opts["format"] = "json"
        if num_predict is not None:
            opts["num_predict"] = int(num_predict)
        return opts

    def _generate_blocking(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        format_json: bool = False,
        num_predict: int | None = None,
    ) -> str:
        try:
            resp = requests.post(
                self.base_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": self._ollama_options(
                        temperature, format_json=format_json, num_predict=num_predict
                    ),
                },
                timeout=self.timeout_seconds,
            )
            if not resp.ok:
                detail = resp.text[:300]
                try:
                    detail = resp.json().get("error", detail)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
                return f"Error communicating with local LLM: {detail}"
            data = resp.json()
            msg = data.get("message") or {}
            thinking = (msg.get("thinking") or "").strip()
            content = (msg.get("content") or "").strip()
            if thinking and self.stream_debug:
                preview = thinking[:200].replace("\n", " ")
                print(
                    f"  [LLM think] {len(thinking)} chars"
                    f"{'' if len(thinking) <= 200 else '…'}: {preview}",
                    flush=True,
                )
            out = content or thinking
            if not out:
                return "Error communicating with local LLM: empty response body"
            return out
        except requests.exceptions.Timeout:
            return (
                f"Error communicating with local LLM: timed out after "
                f"{self.timeout_seconds:.0f}s"
            )
        except Exception as e:
            return f"Error communicating with local LLM: {e}"

    def _generate_ollama_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        format_json: bool = False,
        num_predict: int | None = None,
    ) -> str:
        """Stream tokens to stdout so long R1 calls show live progress."""
        label = self.model_name
        print(f"  [LLM {label}] streaming…", flush=True)
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        in_think = False
        in_answer = False
        try:
            with requests.post(
                self.base_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": self._ollama_options(
                        temperature, format_json=format_json, num_predict=num_predict
                    ),
                },
                timeout=self.timeout_seconds,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    t_chunk = msg.get("thinking") or ""
                    c_chunk = msg.get("content") or ""
                    if t_chunk:
                        if not in_think:
                            in_think = True
                            sys.stdout.write("\n  [think] ")
                            sys.stdout.flush()
                        thinking_parts.append(t_chunk)
                        sys.stdout.write(t_chunk)
                        sys.stdout.flush()
                    if c_chunk:
                        if not in_answer:
                            in_answer = True
                            if in_think:
                                sys.stdout.write("\n  [answer] ")
                            else:
                                sys.stdout.write("\n  [answer] ")
                            sys.stdout.flush()
                        content_parts.append(c_chunk)
                        sys.stdout.write(c_chunk)
                        sys.stdout.flush()
                    if chunk.get("done"):
                        break
            if in_think or in_answer:
                sys.stdout.write("\n")
                sys.stdout.flush()
            content = "".join(content_parts).strip()
            thinking = "".join(thinking_parts).strip()
            if content:
                print(
                    f"  [LLM {label}] done — answer {len(content)} chars"
                    + (f", thinking {len(thinking)} chars" if thinking else ""),
                    flush=True,
                )
                return content
            if thinking:
                print(
                    f"  [LLM {label}] done — no answer field, using thinking ({len(thinking)} chars)",
                    flush=True,
                )
                return thinking
            return "Error communicating with local LLM: empty streamed response"
        except requests.exceptions.Timeout:
            return (
                f"Error communicating with local LLM: timed out after "
                f"{self.timeout_seconds:.0f}s"
            )
        except Exception as e:
            return f"Error communicating with local LLM: {e}"

    def generate_code(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        return self.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
