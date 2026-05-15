import requests

class LLMClient:
    def __init__(self, provider: str = "ollama", model: str = "llama3.2:3b"):
        self.model_name = model
        if provider.lower() == "ollama":
            self.base_url = "http://localhost:11434/api/chat"
        else:
            self.base_url = "http://localhost:1234/v1/chat/completions"

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        try:
            resp = requests.post(
                self.base_url,
                json={"model": self.model_name, "messages": messages, "stream": False,
                      "options": {"temperature": temperature}},
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
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
