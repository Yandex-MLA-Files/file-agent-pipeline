from types import SimpleNamespace

from PIL import Image

from file_agent.vlm.openai_compatible import OpenAICompatibleVLMClient


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def _completion(**message_kwargs):
    defaults = {"content": None, "reasoning": None, "reasoning_content": None}
    message = SimpleNamespace(**{**defaults, **message_kwargs})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(response) -> OpenAICompatibleVLMClient:
    client = OpenAICompatibleVLMClient(base_url="http://localhost:8000/v1", model="test-model")
    client.client = FakeOpenAI(response)
    return client


def test_describe_image_returns_content_when_present():
    client = _client(_completion(content="A red square."))

    description = client.describe_image(Image.new("RGB", (4, 4)), "Describe this.")

    assert description == "A red square."


def test_describe_image_falls_back_to_reasoning_when_content_is_empty():
    # Reasoning-enabled backends (confirmed on the shared Qwen3.5 endpoint) can
    # exhaust max_tokens mid-thought and never write a final content message -
    # the answer is still recoverable from the reasoning field.
    client = _client(_completion(content=None, reasoning="The color is clearly red."))

    description = client.describe_image(Image.new("RGB", (4, 4)), "Describe this.")

    assert description == "The color is clearly red."


def test_describe_image_falls_back_to_reasoning_content_field():
    client = _client(_completion(content="", reasoning_content="Thinking about the image..."))

    description = client.describe_image(Image.new("RGB", (4, 4)), "Describe this.")

    assert description == "Thinking about the image..."


def test_describe_image_returns_empty_string_when_nothing_recorded():
    client = _client(_completion(content=None))

    description = client.describe_image(Image.new("RGB", (4, 4)), "Describe this.")

    assert description == ""


def test_describe_image_uses_configured_max_tokens():
    client = OpenAICompatibleVLMClient(
        base_url="http://localhost:8000/v1", model="test-model", max_tokens=1500
    )
    fake = FakeOpenAI(_completion(content="ok"))
    client.client = fake

    client.describe_image(Image.new("RGB", (4, 4)), "Describe this.")

    assert fake.chat.completions.calls[0]["max_tokens"] == 1500
