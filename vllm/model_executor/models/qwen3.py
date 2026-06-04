"""Inference-only Qwen3 model compatible with HuggingFace weights.

Qwen3 differs from Qwen2 in two ways:
  1. No bias on QKV projection (attention_bias=False).
  2. Per-head RMSNorm on Q and K before rotary embeddings (QK-norm).

All other components are reused from the Qwen2 implementation.
"""
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import (Qwen2ForCausalLM, Qwen2MLP,
                                              Qwen2Model)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors


class Qwen3Attention(nn.Module):
    """Qwen3 multi-head attention with QK-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 131072,
        rope_theta: float = 1000000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling=None,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # No bias in Qwen3 attention projections
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        # QK-norm: one RMSNorm per head dimension for both Q and K
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Flatten to 2D for CUDA-safe RMSNorm (ops.rms_norm expects 2D),
        # then reshape back. Use reshape (not view) since norm output may
        # be non-contiguous.
        t = q.shape[0]
        q = self.q_norm(q.reshape(t * self.num_heads, self.head_dim)).reshape(
            t, self.q_size)
        k = self.k_norm(k.reshape(t * self.num_kv_heads, self.head_dim)).reshape(
            t, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer block (Qwen2 block + QK-norm in attention)."""

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            rms_norm_eps=config.rms_norm_eps,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3Model(Qwen2Model):
    """Qwen3 model body: uses Qwen3DecoderLayer instead of Qwen2DecoderLayer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Deliberately bypass Qwen2Model.__init__ to avoid building
        # Qwen2DecoderLayer (which would register duplicate Attention names).
        nn.Module.__init__(self)
        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding)
        from vllm.model_executor.models.utils import (
            PPMissingLayer, make_empty_intermediate_tensors_factory,
            make_layers)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    """Qwen3 causal language model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Bypass Qwen2ForCausalLM.__init__ to use Qwen3Model
        nn.Module.__init__(self)
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.sampler import get_sampler
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            from vllm.model_executor.models.utils import PPMissingLayer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
