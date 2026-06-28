from file_agent.llm.fake import FakeLLM


def test_fake_llm_returns_debug_answer():
    answer = FakeLLM().generate("This is a test prompt")

    assert "FakeLLM" in answer
    assert "Real YandexGPT is not connected yet" in answer
    assert "Prompt length:" in answer
    assert "This is a test prompt" in answer
