from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 60


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        auth_scheme: str = "Bearer",
        session: requests.Session | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        require_api_key: bool = True,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.api_key = api_key.strip() if api_key else None
        self.auth_scheme = auth_scheme.strip()
        self.session = session or requests.Session()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.require_api_key = require_api_key

        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.model:
            raise ValueError("model is required")
        if self.require_api_key and not self.api_key:
            raise ValueError("api_key is required")

    def generate(self, prompt: str) -> str:
        response_data = self._request_chat_completion(prompt)
        text = self._extract_text(response_data)

        if not text:
            raise ValueError("LLM returned an empty response")

        return text

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_chat_completion(self, prompt: str) -> dict[str, Any]:
        response = self.session.post(
            self.chat_completions_url,
            headers=self._build_headers(),
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"{self.auth_scheme} {self.api_key}"

        return headers

    def _extract_text(self, response_data: dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()
