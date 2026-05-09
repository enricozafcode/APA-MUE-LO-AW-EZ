from openai import OpenAI

class LLMClient:
    def __init__(self, provider: str = "ollama", model: str = "gemma4"):
        """Initializes the client pointing to the local LLM server."""
        self.model_name = model
        self.provider = provider
        
        # Determine the base URL depending on the provider from the config
        if provider.lower() == "ollama":
            base_url = "http://localhost:11434/v1"
        else:
            # Fallback if using LM Studio or others
            base_url = "http://localhost:1234/v1" 
            
        self.client = OpenAI(
            base_url=base_url,
            api_key="local-dummy-key",
            timeout=600.0,  # 10 minutes — local models can be slow
        )

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """Sends full chat history to the local LLM and returns the reply."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error communicating with local LLM: {e}"

    def generate_code(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """
        Sends a prompt to the local LLM and returns the response.
        """
        return self.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )