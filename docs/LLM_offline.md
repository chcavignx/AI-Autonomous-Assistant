# Offline Large Language Model (LLM)

The goal is to enable the assistant to generate intelligent responses offline using local language models.
We have integrated a local LLM client in `src/llm/` that is designed to support both local Ollama and OpenAI-style APIs, configured centrally.

## 1. Ollama

### Description
Ollama is a lightweight, extensible framework for building and running language models locally. It provides a simple API and CLI to download, run, and manage models on your device, making it an excellent choice for offline setups on hardware like the Raspberry Pi.

### Installation

1. **Install Ollama**:
You can use the provided script or install it directly using the official curl command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Alternatively, run our provided script:
```bash
./scripts/install/install_ollama.sh
```

2. **Download Models**:
Once Ollama is installed, you need to pull the models you wish to use offline. For instance, `tinyllama` or `llama3.2:1b` are suitable for lightweight setups.
```bash
ollama pull tinyllama
```
Alternatively, you can run the preloading script to load a set of default models:
```bash
./scripts/models/preload_ollama_models.sh
```

## 2. LLM Client Library

The LLM response generation logic is structured as a generic client package in `src/llm/`.

### Configuration Layer

Configuration is managed via `src/utils/config.py` and `config.yaml`. The LLM configuration block includes:
- `api_type`: Configures the LLM provider (e.g., `ollama` or `openai`).
- `url`: The endpoint URL (e.g., `http://localhost:11434/api/generate` for Ollama).
- `model`: The name of the model to use (e.g., `tinyllama`).
- `api_key`: API key if using a cloud provider (can be null for local Ollama).
- `timeout`: Request timeout in seconds.

### Voice Agent Integration

The offline voice agent (`examples/VAD/voice_agent_offline.py`) has been updated to use this client library:
- Added a `ResponsesConfig` configuration block to handle response rules.
- The agent includes a `use_llm` toggle. If enabled, the agent attempts to generate a response via the LLM API.
- Support is provided for formatting payloads and forwarding custom overrides when using Ollama.
- If the LLM generation fails or times out, it gracefully falls back to basic intent matching or a fallback template.

### Testing and Examples

- **Tests**: The `tests/llm/` and `tests/audio/` directories contain unit and integration tests covering Ollama responses, OpenAI completions, headers, timeout settings, request exceptions, and non-200 responses.
- **Example Script**: A standalone example is provided at `examples/llm/llm_example.py` for end-to-end testing of the LLM library manually using custom prompts or overriding configurations.

```bash
python examples/llm/llm_example.py --prompt "What is the capital of France?" --api-type ollama --model tinyllama
```

## 3. HuggingFace Transformers (Alternative)

An alternative integration for a local LLM involves using HuggingFace Transformers directly.

**Changes required for Transformers integration:**
- Update `llm_model_name`, `llm_device` (e.g. `cpu`), and `llm_torch_dtype` in the configuration.
- Load the model and tokenizer manually.
- Note: This method is significantly more resource-intensive on the Raspberry Pi compared to Ollama.

**Manual Verification:**
```bash
pip install torch transformers accelerate
```
