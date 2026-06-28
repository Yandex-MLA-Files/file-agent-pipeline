import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


YANDEX_GPT_COMPLETION_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)
DEFAULT_YANDEX_MODEL = "yandexgpt-lite"


class YandexGPTClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        env_file: str | Path | None = ".env",
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv(env_file)

        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.model = os.getenv("YANDEX_MODEL", DEFAULT_YANDEX_MODEL)
        self.session = session or requests.Session()

        if not self.api_key:
            raise ValueError("YANDEX_API_KEY is required to use YandexGPTClient")
        if not self.folder_id:
            raise ValueError("YANDEX_FOLDER_ID is required to use YandexGPTClient")

    def generate(self, prompt: str) -> str:
        response_data = self._request_completion(prompt)
        text = self._extract_text(response_data)

        if not text:
            raise ValueError("YandexGPT returned an empty response")

        return text

    @property
    def model_uri(self) -> str:
        return f"gpt://{self.folder_id}/{self.model}/latest"

    def _request_completion(self, prompt: str) -> dict[str, Any]:
        response = self.session.post(
            YANDEX_GPT_COMPLETION_URL,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "modelUri": self.model_uri,
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.2,
                    "maxTokens": "2000",
                },
                "messages": [
                    {
                        "role": "user",
                        "text": prompt,
                    }
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _extract_text(self, response_data: dict[str, Any]) -> str:
        alternatives = response_data.get("result", {}).get("alternatives", [])
        if not alternatives:
            return ""

        message = alternatives[0].get("message", {})
        return str(message.get("text", "")).strip()
