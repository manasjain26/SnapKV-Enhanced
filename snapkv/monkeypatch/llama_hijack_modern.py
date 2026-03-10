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

    # [SnapKV] Expand to num_attention_heads before compression
    key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
    value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

    bsz, num_heads, q_len, head_dim = query_states.shape

    if past_key_value is not None:
        # Determine if this is the prefill phase (first forward pass)
        past_seen = past_key_value.get_seq_length(self.layer_idx)

        if past_seen == 0:
            # === PREFILL PHASE: Apply SnapKV compression ===
            self._snapkv_seq_len = q_len

            # Compress KV cache using SnapKV
            key_states_compressed, value_states_compressed = self.kv_cluster.update_kv(
                key_states_expanded, query_states, value_states_expanded,
                attention_mask, self.num_key_value_groups
            )

            # For the current forward pass, use compressed KV for attention
            # But we need to do full attention for the prefill to get correct output
            # So we compute attention with full KV, then store compressed KV in cache

            # --- Full attention for this prefill step ---
            # Use the attention_interface from the original code path
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            attention_interface: Callable = None
            if self.config._attn_implementation == "eager":
                from transformers.models.llama.modeling_llama import eager_attention_forward
                attention_interface = eager_attention_forward
            else:
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    from transformers.models.llama.modeling_llama import eager_attention_forward
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states_expanded,
                value_states_expanded,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            # --- Store COMPRESSED KV in cache ---
            # We need to convert back to num_key_value_heads if using GQA
            if self.num_key_value_groups > 1:
                # Take every num_key_value_groups-th head from compressed
                kv_heads = key_states.shape[1]  # num_key_value_heads
                # compressed is [bsz, num_attention_heads, compressed_len, head_dim]
                # We need [bsz, num_key_value_heads, compressed_len, head_dim]
                # For GQA, all heads in a group see the same KV, so just take the first of each group
                compressed_len = key_states_compressed.shape[2]
                k_for_cache = key_states_compressed[:, ::self.num_key_value_groups, :, :]
                v_for_cache = value_states_compressed[:, ::self.num_key_value_groups, :, :]
            else:
                k_for_cache = key_states_compressed
                v_for_cache = value_states_compressed

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_value.update(k_for_cache, v_for_cache, self.layer_idx, cache_kwargs)

        else:
            # === DECODE PHASE: Normal attention with cached KV ===
            self._snapkv_seq_len = past_seen + q_len

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            attention_interface: Callable = None
            if self.config._attn_implementation == "eager":
                from transformers.models.llama.modeling_llama import eager_attention_forward
                attention_interface = eager_attention_forward
            else:
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    from transformers.models.llama.modeling_llama import eager_attention_forward
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states_expanded,
                value_states_expanded,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
    else:
        # No cache - just do normal attention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.llama.modeling_llama import eager_attention_forward

        if self.config._attn_implementation == "eager":
            attention_interface = eager_attention_forward
        elif self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states_expanded,
            value_states_expanded,
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
    Resets SnapKV seq length tracking when starting a new generation.
    """
    if past_key_values is None:
        # Reset SnapKV state for new generation
        for layer in self.model.layers:
            if hasattr(layer.self_attn, '_snapkv_seq_len'):
                layer.self_attn._snapkv_seq_len = 0

    # Call the original prepare_inputs_for_generation
    # We need to replicate the essential logic here
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = getattr(past_key_values, 'get_max_length', lambda: None)()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
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

    # Handle cache_position for modern transformers
    cache_position = kwargs.get("cache_position", None)
    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
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
