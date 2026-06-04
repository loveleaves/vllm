import re
from typing import Optional, Sequence, Tuple

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                               DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

# Registered under both names: DeepSeek-R1 and Qwen3 use identical tags.
@ReasoningParserManager.register_module(["deepseek_r1", "qwen3"])
class DeepSeekR1ReasoningParser(ReasoningParser):
    """Reasoning parser for models that use <think>…</think> blocks.

    Validated with DeepSeek-R1 and Qwen3 series.
    """

    THINK_START = "<think>"
    THINK_END = "</think>"

    def __init__(self, tokenizer: AnyTokenizer) -> None:
        super().__init__(tokenizer)
        vocab = self.vocab
        start_id = vocab.get(self.THINK_START)
        end_id = vocab.get(self.THINK_END)
        if start_id is None or end_id is None:
            raise RuntimeError(
                f"Tokenizer does not have '{self.THINK_START}' or "
                f"'{self.THINK_END}' as single tokens. "
                "Cannot use DeepSeekR1ReasoningParser with this tokenizer.")
        self.think_start_token_id: int = start_id
        self.think_end_token_id: int = end_id

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> Tuple[Optional[str], Optional[str]]:
        start = model_output.find(self.THINK_START)
        end = model_output.find(self.THINK_END)

        if start == -1:
            # No think block at all.
            return None, model_output

        if end == -1:
            # Think block opened but not closed (generation cut short).
            reasoning = model_output[start + len(self.THINK_START):]
            return reasoning or None, None

        reasoning = model_output[start + len(self.THINK_START):end]
        content = model_output[end + len(self.THINK_END):]
        return reasoning or None, content or None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional[DeltaMessage]:
        start_id = self.think_start_token_id
        end_id = self.think_end_token_id

        delta_has_start = start_id in delta_token_ids
        delta_has_end = end_id in delta_token_ids
        prev_has_start = start_id in previous_token_ids
        prev_has_end = end_id in previous_token_ids

        # --- Control-token suppression ---
        # Delta is *only* the <think> or </think> token → skip chunk.
        if delta_has_start and not delta_has_end:
            # Just entered reasoning; suppress the opening tag token.
            return None
        if delta_has_end and not prev_has_end:
            # Just closed reasoning; check whether there is trailing content
            # after </think> in the same delta.
            after = delta_text.split(self.THINK_END, 1)
            trailing = after[1] if len(after) > 1 else ""
            if trailing:
                return DeltaMessage(content=trailing)
            return None  # pure closing tag, suppress

        # --- Route based on state ---
        if prev_has_start and not prev_has_end:
            # Still inside the think block.
            return DeltaMessage(reasoning_content=delta_text)

        if prev_has_end:
            # Past the think block → normal content.
            return DeltaMessage(content=delta_text)

        # No think block started yet → normal content.
        return DeltaMessage(content=delta_text)
