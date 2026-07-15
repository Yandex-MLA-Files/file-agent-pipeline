# Local LLM inference

The application does not import vLLM or SGLang directly. `OpenAILLMClient`
uses the official OpenAI Python SDK to call a local OpenAI-compatible HTTP
endpoint. A blank `LOCAL_LLM_API_KEY` is supported for servers without
authentication.

## vLLM

Example:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Project configuration:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_LLM_API_KEY=
```

## SGLang

Example:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Project configuration:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:30000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_LLM_API_KEY=
```

