class FakeLLM:
    def generate(self, prompt: str) -> str:
        preview = prompt[:300]
        return (
            "FakeLLM: prompt was built successfully. "
            "Real YandexGPT is not connected yet.\n\n"
            f"Prompt length: {len(prompt)} characters\n\n"
            f"Prompt preview:\n{preview}"
        )
