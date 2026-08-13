from types import SimpleNamespace

import pytest
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
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _response(content="Visible chart"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


def test_openai_vlm_sends_base64_image_and_qwen_thinking_setting():
    fake_openai = FakeOpenAI(_response("  Revenue increased.  "))
    client = OpenAICompatibleVLMClient(
        base_url="http://unused/v1",
        model="Qwen/Qwen3.5-27B",
        max_tokens=900,
        temperature=0.2,
        enable_thinking=False,
    )
    client.client = fake_openai

    result = client.describe_image(Image.new("RGB", (20, 10)), "Analyze revenue")

    assert result == "Revenue increased."
    request = fake_openai.completions.calls[0]
    assert request["model"] == "Qwen/Qwen3.5-27B"
    assert request["max_tokens"] == 900
    assert request["temperature"] == 0.2
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    content = request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Analyze revenue"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], usage=None),
        _response(None),
        _response("  "),
    ],
)
def test_openai_vlm_rejects_empty_response(response):
    client = OpenAICompatibleVLMClient(
        base_url="http://unused/v1",
        model="test-vlm",
    )
    client.client = FakeOpenAI(response)

    with pytest.raises(ValueError, match="empty response"):
        client.describe_image(Image.new("RGB", (10, 10)), "Analyze")
