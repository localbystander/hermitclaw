# Multi-LLM Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add support for OpenAI-compatible LLM providers (OpenRouter, local models, custom endpoints) alongside existing OpenAI Responses API support.

**Architecture:** Adapter pattern inside `providers.py`. The public interface (`chat()`, `chat_short()`, `embed()`) stays unchanged. Internally, `chat()` routes to either the Responses API (OpenAI) or Chat Completions API (everything else) based on a `provider` config field. Translation functions convert between the two message formats so `brain.py` needs zero changes.

**Tech Stack:** Python, OpenAI SDK (already installed — used for both Responses and Chat Completions clients), PyYAML

**Design doc:** `docs/plans/2026-02-16-multi-provider-design.md`

---

### Task 1: Add test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_providers.py`
- Modify: `pyproject.toml`

**Step 1: Add pytest dependency**

In `pyproject.toml`, add optional test dependencies:

```toml
[project.optional-dependencies]
test = ["pytest>=8.0"]
```

**Step 2: Create test directory and empty init**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 3: Create test file with first test — tool translation**

```python
"""Tests for multi-provider translation in providers.py."""

from hermitclaw.providers import _translate_tools_for_completions


def test_translate_tools_filters_web_search():
    """web_search_preview should be dropped for Chat Completions providers."""
    tools = [
        {"type": "function", "name": "shell", "description": "Run a command",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"type": "web_search_preview"},
        {"type": "function", "name": "respond", "description": "Talk",
         "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    ]
    result = _translate_tools_for_completions(tools)
    assert len(result) == 2
    assert all(t["type"] == "function" for t in result)
    assert result[0]["function"]["name"] == "shell"
    assert result[1]["function"]["name"] == "respond"


def test_translate_tools_converts_format():
    """Function tools should be converted to Chat Completions format."""
    tools = [
        {"type": "function", "name": "shell", "description": "Run a command",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    ]
    result = _translate_tools_for_completions(tools)
    assert result[0] == {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
```

**Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL — `_translate_tools_for_completions` doesn't exist yet

**Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "feat: add test infrastructure and first provider translation tests"
```

---

### Task 2: Implement tool translation

**Files:**
- Modify: `hermitclaw/providers.py` (add `_translate_tools_for_completions`)

**Step 1: Write `_translate_tools_for_completions` in providers.py**

Add after the `TOOLS` definition (after line 67):

```python
def _translate_tools_for_completions(tools: list[dict]) -> list[dict]:
    """Convert Responses API tool defs to Chat Completions format.

    Drops web_search_preview (unsupported). Wraps function tools in the
    {"type": "function", "function": {...}} structure Chat Completions expects.
    """
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        })
    return result
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 2 PASS

**Step 3: Commit**

```bash
git add hermitclaw/providers.py
git commit -m "feat: add tool translation for Chat Completions format"
```

---

### Task 3: Test and implement input translation

**Files:**
- Modify: `tests/test_providers.py` (add input translation tests)
- Modify: `hermitclaw/providers.py` (add `_translate_input_to_messages`)

**Step 1: Write failing tests for input translation**

Append to `tests/test_providers.py`:

```python
from hermitclaw.providers import _translate_input_to_messages


def test_translate_role_messages_pass_through():
    """Standard role-based messages should pass through unchanged."""
    input_list = [
        {"role": "assistant", "content": "I'm thinking..."},
        {"role": "user", "content": "Hello there"},
    ]
    messages = _translate_input_to_messages(input_list, instructions=None)
    assert messages == input_list


def test_translate_prepends_system_message():
    """Instructions should become a system message at the front."""
    input_list = [{"role": "user", "content": "Hello"}]
    messages = _translate_input_to_messages(input_list, instructions="You are a crab.")
    assert messages[0] == {"role": "system", "content": "You are a crab."}
    assert messages[1] == {"role": "user", "content": "Hello"}


def test_translate_function_call_output():
    """function_call_output items should become tool role messages."""
    input_list = [
        {"role": "user", "content": "Run ls"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "shell", "arguments": '{"command": "ls"}'}}
        ]},
        {"type": "function_call_output", "call_id": "call_123", "output": "file1.txt\nfile2.txt"},
    ]
    messages = _translate_input_to_messages(input_list, instructions=None)
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_123", "content": "file1.txt\nfile2.txt"}


def test_translate_multimodal_content():
    """Responses API image format should be converted to Chat Completions format."""
    input_list = [
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
            {"type": "input_text", "text": "What do you see?"},
        ]},
    ]
    messages = _translate_input_to_messages(input_list, instructions=None)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
    assert content[1] == {"type": "text", "text": "What do you see?"}


def test_translate_skips_responses_api_objects():
    """Items that are Responses API SDK objects (not dicts) should be skipped."""
    class FakeResponseItem:
        type = "message"
    input_list = [
        {"role": "user", "content": "Hello"},
        FakeResponseItem(),  # SDK object from response["output"]
    ]
    messages = _translate_input_to_messages(input_list, instructions=None)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL — `_translate_input_to_messages` doesn't exist

**Step 3: Implement `_translate_input_to_messages`**

Add to `hermitclaw/providers.py`:

```python
def _translate_input_to_messages(input_list: list, instructions: str | None) -> list[dict]:
    """Convert a Responses API input_list to Chat Completions messages.

    Handles:
    - Role-based dicts ({"role": ..., "content": ...}) pass through
    - {"type": "function_call_output", ...} -> {"role": "tool", ...}
    - Multimodal content: input_image -> image_url, input_text -> text
    - Non-dict items (SDK objects from Responses API) are skipped
    - Instructions become a system message at index 0
    """
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for item in input_list:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item.get("output", ""),
            })
        elif "role" in item:
            content = item.get("content")
            if isinstance(content, list):
                content = _translate_multimodal(content)
            messages.append({**item, "content": content})
        # else: skip unknown items

    return messages


def _translate_multimodal(content_parts: list[dict]) -> list[dict]:
    """Convert Responses API multimodal content to Chat Completions format."""
    result = []
    for part in content_parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "input_image":
            result.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
        elif part.get("type") == "input_text":
            result.append({"type": "text", "text": part["text"]})
        else:
            result.append(part)
    return result
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add hermitclaw/providers.py tests/test_providers.py
git commit -m "feat: add input translation from Responses API to Chat Completions format"
```

---

### Task 4: Test and implement response normalization

**Files:**
- Modify: `tests/test_providers.py` (add response normalization tests)
- Modify: `hermitclaw/providers.py` (add `_normalize_completions_response`)

**Step 1: Write failing tests**

Append to `tests/test_providers.py`:

```python
from hermitclaw.providers import _normalize_completions_response


def test_normalize_text_only_response():
    """A text-only response should have text set and empty tool_calls."""
    # Simulate openai ChatCompletion response object
    class FakeContent:
        def __init__(self):
            self.content = "I'm thinking about shells."
            self.tool_calls = None
    class FakeChoice:
        def __init__(self):
            self.message = FakeContent()
    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    result = _normalize_completions_response(FakeResponse())
    assert result["text"] == "I'm thinking about shells."
    assert result["tool_calls"] == []
    assert result["output"] == []


def test_normalize_tool_call_response():
    """Tool calls should be normalized to same format as Responses API."""
    class FakeFunction:
        def __init__(self):
            self.name = "shell"
            self.arguments = '{"command": "ls"}'
    class FakeToolCall:
        def __init__(self):
            self.id = "call_abc"
            self.type = "function"
            self.function = FakeFunction()
    class FakeContent:
        def __init__(self):
            self.content = None
            self.tool_calls = [FakeToolCall()]
    class FakeChoice:
        def __init__(self):
            self.message = FakeContent()
    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    result = _normalize_completions_response(FakeResponse())
    assert result["text"] is None
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["name"] == "shell"
    assert tc["arguments"] == {"command": "ls"}
    assert tc["call_id"] == "call_abc"


def test_normalize_output_for_tool_loop():
    """output list should contain dicts that brain.py can append to input_list.

    When brain.py does input_list += response["output"], the resulting items
    must be translatable by _translate_input_to_messages on the next call.
    """
    class FakeFunction:
        def __init__(self):
            self.name = "shell"
            self.arguments = '{"command": "ls"}'
    class FakeToolCall:
        def __init__(self):
            self.id = "call_abc"
            self.type = "function"
            self.function = FakeFunction()
    class FakeContent:
        def __init__(self):
            self.content = "Let me check."
            self.tool_calls = [FakeToolCall()]
    class FakeChoice:
        def __init__(self):
            self.message = FakeContent()
    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    result = _normalize_completions_response(FakeResponse())
    # output should be a single assistant message with tool_calls
    assert len(result["output"]) == 1
    msg = result["output"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"][0]["id"] == "call_abc"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL — `_normalize_completions_response` doesn't exist

**Step 3: Implement `_normalize_completions_response`**

Add to `hermitclaw/providers.py`:

```python
def _normalize_completions_response(response) -> dict:
    """Normalize a Chat Completions response into the same format as _chat_responses.

    Returns {"text", "tool_calls", "output"} where output contains dicts
    that brain.py can safely append to input_list for follow-up calls.
    """
    message = response.choices[0].message
    text = message.content
    tool_calls = []
    output = []

    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
                "call_id": tc.id,
            })

        # Build a synthetic assistant message for brain.py's input_list
        output.append({
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

    return {"text": text, "tool_calls": tool_calls, "output": output}
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add hermitclaw/providers.py tests/test_providers.py
git commit -m "feat: add response normalization for Chat Completions"
```

---

### Task 5: Update config for multi-provider support

**Files:**
- Modify: `config.yaml`
- Modify: `hermitclaw/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
"""Tests for multi-provider config loading."""

import os
import pytest


def test_default_provider_is_openai(tmp_path, monkeypatch):
    """Config without provider field should default to openai."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: gpt-4.1\napi_key: test-key\n")
    monkeypatch.setattr("hermitclaw.config.CONFIG_PATH", str(config_file))

    from hermitclaw.config import load_config
    cfg = load_config()
    assert cfg["provider"] == "openai"


def test_openrouter_provider_sets_base_url(tmp_path, monkeypatch):
    """OpenRouter provider should auto-set base_url."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: openrouter\nmodel: openai/gpt-4.1\napi_key: or-key\n")
    monkeypatch.setattr("hermitclaw.config.CONFIG_PATH", str(config_file))

    from hermitclaw.config import load_config
    cfg = load_config()
    assert cfg["base_url"] == "https://openrouter.ai/api/v1"


def test_openrouter_api_key_env_var(tmp_path, monkeypatch):
    """OPENROUTER_API_KEY should be used for openrouter provider."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: openrouter\nmodel: openai/gpt-4.1\n")
    monkeypatch.setattr("hermitclaw.config.CONFIG_PATH", str(config_file))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from hermitclaw.config import load_config
    cfg = load_config()
    assert cfg["api_key"] == "or-env-key"


def test_custom_provider_requires_base_url(tmp_path, monkeypatch):
    """Custom provider without base_url should raise ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: custom\nmodel: llama3\napi_key: key\n")
    monkeypatch.setattr("hermitclaw.config.CONFIG_PATH", str(config_file))

    from hermitclaw.config import load_config
    with pytest.raises(ValueError, match="base_url"):
        load_config()


def test_env_var_overrides(tmp_path, monkeypatch):
    """HERMITCLAW_PROVIDER and HERMITCLAW_BASE_URL should override config."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: openai\nmodel: gpt-4.1\napi_key: key\n")
    monkeypatch.setattr("hermitclaw.config.CONFIG_PATH", str(config_file))
    monkeypatch.setenv("HERMITCLAW_PROVIDER", "custom")
    monkeypatch.setenv("HERMITCLAW_BASE_URL", "http://localhost:11434/v1")

    from hermitclaw.config import load_config
    cfg = load_config()
    assert cfg["provider"] == "custom"
    assert cfg["base_url"] == "http://localhost:11434/v1"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — config doesn't handle provider field yet

**Step 3: Update `config.yaml`**

Add `provider` and `base_url` fields:

```yaml
# HermitClaw Configuration
provider: "openai"              # "openai" | "openrouter" | "custom"
model: "gpt-4.1"
api_key: null                   # set here or via OPENAI_API_KEY / OPENROUTER_API_KEY env var
base_url: null                  # auto-set for known providers; required for "custom"

thinking_pace_seconds: 5       # how often it thinks (steady pulse)
max_thoughts_in_context: 4     # rolling window of recent thoughts

# Memory stream settings
reflection_threshold: 50       # accumulated importance before reflecting
memory_retrieval_count: 3      # how many memories to retrieve per query
embedding_model: "text-embedding-3-small"
recency_decay_rate: 0.995      # exponential decay rate for recency scoring

# environment_path is auto-detected from *_box/ directories
# Uncomment to override: environment_path: "./mybox"
```

**Step 4: Update `hermitclaw/config.py`**

Replace the full file:

```python
"""All configuration in one place."""

import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

# Known provider presets: provider_name -> default base_url
PROVIDER_PRESETS = {
    "openai": None,  # uses OpenAI default
    "openrouter": "https://openrouter.ai/api/v1",
}

# Provider-specific API key env vars (checked before OPENAI_API_KEY fallback)
PROVIDER_KEY_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
}


def load_config() -> dict:
    """Load config from config.yaml, with env var overrides."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Provider (default: openai)
    config["provider"] = (
        os.environ.get("HERMITCLAW_PROVIDER")
        or config.get("provider", "openai")
    )
    provider = config["provider"]

    # Base URL: env var > config > provider preset
    config["base_url"] = (
        os.environ.get("HERMITCLAW_BASE_URL")
        or config.get("base_url")
        or PROVIDER_PRESETS.get(provider)
    )

    # API key: provider-specific env var > OPENAI_API_KEY > config
    provider_key_var = PROVIDER_KEY_ENV_VARS.get(provider)
    config["api_key"] = (
        (os.environ.get(provider_key_var) if provider_key_var else None)
        or os.environ.get("OPENAI_API_KEY")
        or config.get("api_key")
    )

    # Model
    config["model"] = os.environ.get("HERMITCLAW_MODEL") or config.get("model", "gpt-4o")

    # Defaults for numeric settings
    config.setdefault("thinking_pace_seconds", 45)
    config.setdefault("max_thoughts_in_context", 20)
    config.setdefault("environment_path", "./environment")
    config.setdefault("reflection_threshold", 50)
    config.setdefault("memory_retrieval_count", 3)
    config.setdefault("embedding_model", "text-embedding-3-small")
    config.setdefault("recency_decay_rate", 0.995)

    # Resolve environment_path relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isabs(config["environment_path"]):
        config["environment_path"] = os.path.join(project_root, config["environment_path"])

    # Validation
    if provider == "custom" and not config.get("base_url"):
        raise ValueError("Provider 'custom' requires base_url in config.yaml or HERMITCLAW_BASE_URL env var")

    return config


# Global config — loaded once, can be updated at runtime
config = load_config()
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add config.yaml hermitclaw/config.py tests/test_config.py
git commit -m "feat: add multi-provider config with presets and env var overrides"
```

---

### Task 6: Wire up Chat Completions path in providers.py

**Files:**
- Modify: `hermitclaw/providers.py`

This is the integration task — connect the translation functions to actual API calls.

**Step 1: Add `_uses_responses_api` helper and `_completions_client`**

Add to `hermitclaw/providers.py`:

```python
def _uses_responses_api() -> bool:
    """True if the configured provider uses the OpenAI Responses API."""
    return config["provider"] == "openai"


def _completions_client() -> openai.OpenAI:
    """Create an OpenAI client configured for Chat Completions (with base_url)."""
    kwargs = {"api_key": config["api_key"]}
    if config.get("base_url"):
        kwargs["base_url"] = config["base_url"]
    return openai.OpenAI(**kwargs)
```

**Step 2: Add `_chat_completions` function**

```python
def _chat_completions(input_list: list, tools: bool = True, instructions: str = None, max_tokens: int = 300) -> dict:
    """Make a Chat Completions API call. Same return format as _chat_responses."""
    messages = _translate_input_to_messages(input_list, instructions)

    kwargs = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        completions_tools = _translate_tools_for_completions(TOOLS)
        if completions_tools:
            kwargs["tools"] = completions_tools

    response = _completions_client().chat.completions.create(**kwargs)
    return _normalize_completions_response(response)
```

**Step 3: Rename existing `chat` internals and update routing**

Rename current `chat()` body to `_chat_responses()`, then make `chat()` route:

```python
def _chat_responses(input_list: list, tools: bool = True, instructions: str = None, max_tokens: int = 300) -> dict:
    """Make one Responses API call (OpenAI only)."""
    kwargs = {
        "model": config["model"],
        "input": input_list,
        "max_output_tokens": max_tokens,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if tools:
        kwargs["tools"] = TOOLS

    response = _client().responses.create(**kwargs)

    text_parts = []
    tool_calls = []
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
        elif item.type == "function_call":
            tool_calls.append({
                "name": item.name,
                "arguments": json.loads(item.arguments),
                "call_id": item.call_id,
            })

    return {
        "text": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls,
        "output": response.output,
    }


def chat(input_list: list, tools: bool = True, instructions: str = None, max_tokens: int = 300) -> dict:
    """Make an LLM call. Routes to Responses API or Chat Completions based on provider config.

    Returns:
    {
        "text": str or None,
        "tool_calls": [{"name": str, "arguments": dict, "call_id": str}],
        "output": list,   # raw output (for appending back to input on follow-up calls)
    }
    """
    if _uses_responses_api():
        return _chat_responses(input_list, tools, instructions, max_tokens)
    return _chat_completions(input_list, tools, instructions, max_tokens)
```

**Step 4: Verify existing behavior is preserved**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (translation tests still pass, no behavior change for openai provider)

**Step 5: Commit**

```bash
git add hermitclaw/providers.py
git commit -m "feat: wire up Chat Completions path with provider routing in chat()"
```

---

### Task 7: Update embed() for multi-provider support

**Files:**
- Modify: `hermitclaw/providers.py`

**Step 1: Update `embed()` to use the configured provider's client**

Replace the current `embed()`:

```python
def embed(text: str) -> list[float]:
    """Get an embedding vector for a text string.

    Uses the configured provider's embeddings endpoint. Falls back to OpenAI
    if the provider doesn't support embeddings (requires OPENAI_API_KEY).
    """
    from hermitclaw.config import config as cfg
    model = cfg.get("embedding_model", "text-embedding-3-small")

    # Try configured provider first
    try:
        client = _completions_client() if not _uses_responses_api() else _client()
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception:
        if _uses_responses_api():
            raise  # OpenAI is already the provider, don't retry
        # Fall back to OpenAI for embeddings
        fallback_key = os.environ.get("OPENAI_API_KEY")
        if not fallback_key:
            raise
        fallback = openai.OpenAI(api_key=fallback_key)
        response = fallback.embeddings.create(model=model, input=text)
        return response.data[0].embedding
```

Add `import os` at the top of providers.py if not already there.

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add hermitclaw/providers.py
git commit -m "feat: multi-provider embeddings with OpenAI fallback"
```

---

### Task 8: Final integration and manual smoke test

**Files:**
- No file changes — verification only

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Verify OpenAI path still works (default config)**

Start the app with current config (provider: openai):

```bash
python hermitclaw/main.py
```

Verify the crab starts thinking. Stop with Ctrl+C.

**Step 3: Verify config validation for custom provider**

Test that missing base_url raises an error:

```bash
HERMITCLAW_PROVIDER=custom python -c "from hermitclaw.config import load_config; load_config()"
```

Expected: `ValueError: Provider 'custom' requires base_url`

**Step 4: Verify OpenRouter config loads correctly**

```bash
HERMITCLAW_PROVIDER=openrouter OPENROUTER_API_KEY=test python -c "
from hermitclaw.config import load_config
cfg = load_config()
print(f'provider={cfg[\"provider\"]}, base_url={cfg[\"base_url\"]}, key={cfg[\"api_key\"][:4]}...')
"
```

Expected: `provider=openrouter, base_url=https://openrouter.ai/api/v1, key=test...`

**Step 5: Commit any remaining changes and tag**

```bash
git add -A
git status  # verify no unexpected files
git commit -m "feat: complete multi-LLM provider support" --allow-empty
```
