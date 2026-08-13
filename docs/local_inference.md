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

### Qwen3.5 tool calling and visual analysis

`Qwen/Qwen3.5-27B` can serve text generation, tool calling, and image input from
the same OpenAI-compatible endpoint. Keep the deployment-specific tensor
parallel and memory settings, and enable the relevant capabilities:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

Do not pass `--language-model-only`; that option disables multimodal input.

Use the same endpoint for both project clients:

```env
RAG_MODE=tool_agent
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen3.5-27B
LOCAL_LLM_API_KEY=
LOCAL_LLM_ENABLE_THINKING=false

VLM_BACKEND=openai
VLM_BASE_URL=http://localhost:8000/v1
VLM_MODEL=Qwen/Qwen3.5-27B
VLM_API_KEY=dummy
VLM_MAX_TOKENS=1000
VLM_TEMPERATURE=0.2
VLM_TIMEOUT_SECONDS=120
VLM_ENABLE_THINKING=false
```

The server must support both image requests and automatic tool choice. Verify
the exact deployment with one image request before enabling it for users.

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
