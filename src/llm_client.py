from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str


class LLMClient:
    """Provider-agnostic local LLM client (starter implementation)."""

    def __init__(self, provider: str = "ollama", model: str = "qwen3-coder") -> None:
        self.provider = provider
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate_experiment(self, prompt: str) -> LLMResponse:
        # TODO: Replace with real provider calls (ollama/openai-compatible).
        # This stub keeps the project runnable before integration.
        content = (
            "EXPERIMENT_PROPOSAL\n"
            "Hypothesis: A compact CNN on mel-spectrograms improves signal robustness.\n"
            "Code: # TODO generate executable training code\n"
        )
        return LLMResponse(content=content)
