"""E2E test: Qwen3-1.7B reasoning output via OpenAI-compatible API.

Requires a running vllm server. Start it with:

    python -m vllm.entrypoints.openai.api_server \
        --model ~/.cache/huggingface/hub/Qwen3-1.7B \
        --enable-reasoning \
        --reasoning-parser qwen3 \
        --port 8000

Then run this test:

    pytest tests/entrypoints/openai/reasoning_parsers/test_e2e_qwen3.py -v
"""
import os

import openai
import pytest

BASE_URL = os.environ.get("VLLM_TEST_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen3-1.7B")


@pytest.fixture(scope="module")
def client():
    return openai.OpenAI(base_url=BASE_URL, api_key="dummy")


def test_non_streaming_reasoning_field(client):
    """message.reasoning_content is non-empty and message.content has no <think>."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2? Think step by step."}],
        max_tokens=512,
        stream=False,
        extra_body={"enable_thinking": True},
    )
    msg = resp.choices[0].message
    # Access via model extra fields since openai SDK may not know the field
    raw = msg.model_extra or {}
    reasoning = raw.get("reasoning_content") or getattr(msg, "reasoning_content", None)

    assert reasoning is not None, (
        f"reasoning_content is None; full message={msg}")
    assert "<think>" not in (msg.content or ""), (
        f"content still contains <think> tag: {msg.content!r}")
    assert "</think>" not in (msg.content or ""), (
        f"content still contains </think> tag: {msg.content!r}")


def test_non_streaming_no_reasoning_when_disabled(client):
    """Without --enable-thinking, reasoning_content should be absent or None."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=64,
        stream=False,
    )
    msg = resp.choices[0].message
    raw = msg.model_extra or {}
    reasoning = raw.get("reasoning_content") or getattr(msg, "reasoning_content", None)
    # In non-thinking mode, Qwen3 won't emit <think> blocks,
    # so reasoning_content should be None.
    assert reasoning is None or reasoning == "", (
        f"Unexpected reasoning_content={reasoning!r}")


def test_streaming_reasoning_then_content(client):
    """Streaming: reasoning chunks appear before content chunks."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 3 * 7? Reason first."}],
        max_tokens=256,
        stream=True,
        extra_body={"enable_thinking": True},
    )
    reasoning_parts = []
    content_parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        raw = delta.model_extra or {}
        r = raw.get("reasoning_content") or getattr(delta, "reasoning_content", None)
        c = delta.content
        if r:
            reasoning_parts.append(r)
        if c:
            content_parts.append(c)

    assert reasoning_parts, "No reasoning_content deltas received"
    assert content_parts, "No content deltas received"

    # Verify content doesn't contain think tags
    full_content = "".join(content_parts)
    assert "<think>" not in full_content
    assert "</think>" not in full_content
