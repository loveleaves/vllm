"""Unit tests for DeepSeekR1ReasoningParser (also covers qwen3 alias).

Uses facebook/opt-125m tokenizer with manually injected <think>/</think>
tokens so no GPU is needed.
"""
from typing import List, Sequence
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.reasoning_parsers import (
    DeepSeekR1ReasoningParser, ReasoningParserManager)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPT_MODEL = "facebook/opt-125m"
THINK_START = "<think>"
THINK_END = "</think>"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(OPT_MODEL)
    tok.add_tokens([THINK_START, THINK_END], special_tokens=True)
    return tok


@pytest.fixture(scope="module")
def parser(tokenizer):
    return DeepSeekR1ReasoningParser(tokenizer)


def _make_request() -> ChatCompletionRequest:
    mock = MagicMock(spec=ChatCompletionRequest)
    return mock


def _tokenize(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _simulate_streaming(
    parser: DeepSeekR1ReasoningParser,
    tokenizer,
    full_text: str,
):
    """Simulate token-by-token streaming; return list of DeltaMessage."""
    token_ids = _tokenize(tokenizer, full_text)
    deltas: List[DeltaMessage] = []
    prev_text = ""
    prev_ids: List[int] = []
    for i, tid in enumerate(token_ids):
        curr_ids = prev_ids + [tid]
        curr_text = tokenizer.decode(curr_ids, skip_special_tokens=False)
        delta_text = tokenizer.decode([tid], skip_special_tokens=False)
        msg = parser.extract_reasoning_content_streaming(
            previous_text=prev_text,
            current_text=curr_text,
            delta_text=delta_text,
            previous_token_ids=prev_ids,
            current_token_ids=curr_ids,
            delta_token_ids=[tid],
        )
        if msg is not None:
            deltas.append(msg)
        prev_text = curr_text
        prev_ids = curr_ids
    return deltas


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


def test_non_streaming_full_think_block(parser):
    req = _make_request()
    reasoning, content = parser.extract_reasoning_content(
        "<think>step1 step2</think>final answer", req)
    assert reasoning == "step1 step2"
    assert content == "final answer"


def test_non_streaming_no_think_block(parser):
    req = _make_request()
    reasoning, content = parser.extract_reasoning_content(
        "just a plain answer", req)
    assert reasoning is None
    assert content == "just a plain answer"


def test_non_streaming_unclosed_think(parser):
    """Generation cut short before </think>."""
    req = _make_request()
    reasoning, content = parser.extract_reasoning_content(
        "<think>incomplete reasoning", req)
    assert reasoning == "incomplete reasoning"
    assert content is None


def test_non_streaming_empty_think_block(parser):
    req = _make_request()
    reasoning, content = parser.extract_reasoning_content(
        "<think></think>answer", req)
    assert reasoning is None  # empty string treated as None
    assert content == "answer"


def test_non_streaming_multiline_reasoning(parser):
    req = _make_request()
    block = "line one\nline two\nline three"
    reasoning, content = parser.extract_reasoning_content(
        f"<think>{block}</think>conclusion", req)
    assert reasoning == block
    assert content == "conclusion"


def test_non_streaming_only_think_no_content(parser):
    """Reasoning present but no content after closing tag."""
    req = _make_request()
    reasoning, content = parser.extract_reasoning_content(
        "<think>reasoning only</think>", req)
    assert reasoning == "reasoning only"
    assert content is None  # empty string → None


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def test_streaming_full_think_block(parser, tokenizer):
    deltas = _simulate_streaming(
        parser, tokenizer,
        "<think>think content</think>answer text")
    reasoning_parts = [d.reasoning_content for d in deltas
                       if d.reasoning_content is not None]
    content_parts = [d.content for d in deltas
                     if d.content is not None]
    assert "".join(reasoning_parts)  # non-empty reasoning
    assert "".join(content_parts)    # non-empty content
    # Content must not contain the think tags
    full_content = "".join(content_parts)
    assert THINK_START not in full_content
    assert THINK_END not in full_content


def test_streaming_no_think_block(parser, tokenizer):
    deltas = _simulate_streaming(parser, tokenizer, "plain answer")
    reasoning_parts = [d.reasoning_content for d in deltas
                       if d.reasoning_content is not None]
    content_parts = [d.content for d in deltas if d.content is not None]
    assert not reasoning_parts  # no reasoning tokens
    assert "".join(content_parts)  # some content


def test_streaming_reasoning_before_content(parser, tokenizer):
    """All reasoning_content deltas must appear before any content deltas."""
    deltas = _simulate_streaming(
        parser, tokenizer,
        "<think>think</think>answer")
    seen_content = False
    for d in deltas:
        if d.content is not None:
            seen_content = True
        if d.reasoning_content is not None:
            assert not seen_content, (
                "reasoning_content delta appeared after content delta")


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


def test_parser_registered_under_both_names():
    assert "deepseek_r1" in ReasoningParserManager.reasoning_parsers
    assert "qwen3" in ReasoningParserManager.reasoning_parsers
    assert (ReasoningParserManager.reasoning_parsers["deepseek_r1"]
            is ReasoningParserManager.reasoning_parsers["qwen3"])


def test_missing_think_tokens_raises(tmp_path):
    """A tokenizer without <think> tokens must cause RuntimeError."""
    tok = AutoTokenizer.from_pretrained(OPT_MODEL)
    # Do NOT add think tokens
    with pytest.raises(RuntimeError, match="single tokens"):
        DeepSeekR1ReasoningParser(tok)
