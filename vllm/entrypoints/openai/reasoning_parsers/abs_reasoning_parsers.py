from functools import cached_property
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                               DeltaMessage)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class ReasoningParser:
    """Base class for reasoning content parsers.

    Subclasses extract <think>…</think> content from model output and split it
    from the final answer, both for non-streaming and streaming responses.
    """

    def __init__(self, tokenizer: AnyTokenizer) -> None:
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> Dict[str, int]:
        return self.model_tokenizer.get_vocab()

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming: return (reasoning_content, content).

        reasoning_content is None when no <think> block is present.
        content is None when the output is entirely reasoning (no answer yet).
        """
        raise NotImplementedError

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional[DeltaMessage]:
        """Streaming: return a DeltaMessage routed to reasoning or content.

        Returns None to signal that the delta is a control token and should
        be suppressed (no chunk sent to the client).
        """
        raise NotImplementedError


class ReasoningParserManager:
    reasoning_parsers: Dict[str, Type[ReasoningParser]] = {}

    @classmethod
    def get_reasoning_parser(cls, name: str) -> Type[ReasoningParser]:
        parser = cls.reasoning_parsers.get(name)
        if parser is None:
            raise KeyError(
                f"Reasoning parser '{name}' is not registered. "
                f"Available: {list(cls.reasoning_parsers.keys())}")
        return parser

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = True,
        module: Optional[Type] = None,
    ) -> Union[Type, Callable]:
        """Decorator or direct call to register a ReasoningParser subclass."""

        def _register(parser_cls: Type) -> Type:
            names = [name] if isinstance(name, str) else (name or [])
            if not names:
                names = [parser_cls.__name__]
            for n in names:
                if n in cls.reasoning_parsers and not force:
                    raise KeyError(
                        f"Reasoning parser '{n}' already registered.")
                cls.reasoning_parsers[n] = parser_cls
            return parser_cls

        if module is not None:
            return _register(module)
        return _register
