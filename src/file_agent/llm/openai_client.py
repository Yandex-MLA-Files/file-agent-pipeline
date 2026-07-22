from openai import OpenAI


class OpenAILLMClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self.client = client
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.model:
            raise ValueError("model is required")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.choices:
            raise ValueError("LLM returned an empty response")

        text = response.choices[0].message.content
        if not text or not text.strip():
            raise ValueError("LLM returned an empty response")

        return text.strip()
