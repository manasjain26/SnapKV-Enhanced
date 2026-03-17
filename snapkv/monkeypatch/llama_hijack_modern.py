"""
SnapKV hijack for modern transformers (>= 4.43) where LlamaAttention is unified.
LlamaFlashAttention2 no longer exists; attention backend is selected via config._attn_implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import warnings
import math

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from snapkv.monkeypatch.snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)


def llama_attention_forward_modern(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Modern LlamaAttention.forward with SnapKV KV cache compression.
    Compatible with transformers >= 4.43.
    """
    # [SnapKV] register kv_cluster
    init_snapkv(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    bsz, num_heads, q_len, head_dim = query_states.shape

    # Resolve the attention interface once
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if self.config._attn_implementation == "eager":
        from transformers.models.llama.modeling_llama import eager_attention_forward
        attention_interface = eager_attention_forward
    elif self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        from transformers.models.llama.modeling_llama import eager_attention_forward
        attention_interface = eager_attention_forward
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, None
        )
        if attention_interface is None:
            from transformers.models.llama.modeling_llama import eager_attention_forward
            attention_interface = eager_attention_forward

    if past_key_value is not None:
        past_seen = past_key_value.get_seq_length(self.layer_idx)

        if past_seen == 0:
            # === PREFILL PHASE: Apply SnapKV compression ===

            # [SnapKV] Track actual input length so prepare_inputs_for_generation
            # knows all tokens have been processed (cache length != input length
            # after compression).
            if not hasattr(self, 'kv_seq_len'):
                self.kv_seq_len = 0
            self.kv_seq_len = q_len

            # Expand KV to num_attention_heads ONLY for SnapKV compression
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

            # Compress using SnapKV (operates on expanded heads)
            key_compressed_exp, value_compressed_exp = self.kv_cluster.update_kv(
                key_states_expanded, query_states, value_states_expanded,
                attention_mask, self.num_key_value_groups
            )

            # Do full attention for the prefill output
            # NOTE: attention_interface internally calls repeat_kv,
            # so we pass the ORIGINAL (unexpanded) key/value states
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            # Store COMPRESSED KV in cache (reduced back to num_kv_heads)
            if self.num_key_value_groups > 1:
                k_for_cache = key_compressed_exp[:, ::self.num_key_value_groups, :, :]
                v_for_cache = value_compressed_exp[:, ::self.num_key_value_groups, :, :]
            else:
                k_for_cache = key_compressed_exp
                v_for_cache = value_compressed_exp

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_value.update(k_for_cache, v_for_cache, self.layer_idx, cache_kwargs)

        else:
            # === DECODE PHASE: Normal attention with cached KV ===
            self.kv_seq_len += q_len

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            # attention_interface handles repeat_kv internally
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
    else:
        # No cache — normal attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def prepare_inputs_for_generation_llama_modern(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    """
    Modern prepare_inputs_for_generation for Llama with SnapKV support.
    Uses kv_seq_len (actual tokens processed) instead of cache.seen_tokens
    (compressed cache length) to correctly identify which tokens are new.
    """
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            # Use the actual tokens processed (kv_seq_len), NOT cache.seen_tokens
            # which reflects the compressed cache size after SnapKV.
            past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = getattr(past_key_values, 'get_max_length', lambda: None)()
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    cache_position = kwargs.get("cache_position", None)
    if cache_position is None:
        past_seen = self.model.layers[0].self_attn.kv_seq_len if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + input_ids.shape[1], device=input_ids.device
        )

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }
    )
    return model_inputs
