
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool',
                 num_obs_windows = 1, protect_spikes = False, spike_reserve_ratio = 0.1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        # === Enhancement parameters ===
        self.num_obs_windows = num_obs_windows          # Improvement 2: multi-window
        self.protect_spikes = protect_spikes            # Improvement 3: spike protection
        self.spike_reserve_ratio = spike_reserve_ratio  # fraction of budget reserved for spikes

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool',
              num_obs_windows = 1, protect_spikes = False, spike_reserve_ratio = 0.1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.num_obs_windows = num_obs_windows
        self.protect_spikes = protect_spikes
        self.spike_reserve_ratio = spike_reserve_ratio

    # =========================================================================
    # Improvement 2: Multi-Window Observation
    # Addresses the "lost-in-the-middle" problem by using observation windows
    # at multiple positions (beginning, middle, end) instead of just the end.
    # =========================================================================
    def _compute_multi_window_attention(self, query_states, key_states, head_dim):
        """
        Compute attention scores from multiple observation windows placed at
        evenly-spaced positions throughout the sequence.

        Each window only sees prefix positions causally reachable by its queries.
        To prevent early windows from being drowned out (they see fewer positions),
        we normalize each window's summed attention so all windows contribute on
        a comparable scale before taking the element-wise max.

        Args:
            query_states: [B, H, q_len, D] full query states
            key_states:   [B, H, q_len, D] full key states (before window split)
            head_dim:     int, head dimension for scaling

        Returns:
            attn_weights_sum: [B, H, prefix_len] aggregated attention to prefix
        """
        bsz, num_heads, q_len, _ = query_states.shape
        prefix_len = q_len - self.window_size

        window_positions = []
        for i in range(self.num_obs_windows):
            if self.num_obs_windows == 1:
                start = q_len - self.window_size
            else:
                start = int(i * (q_len - self.window_size) / max(1, self.num_obs_windows - 1))
                start = min(start, q_len - self.window_size)
            window_positions.append(start)
        seen = set()
        window_positions = [s for s in window_positions if not (s in seen or seen.add(s))]

        all_attn_sums = []
        for start in window_positions:
            end = start + self.window_size
            window_q = query_states[:, :, start:end, :]

            prefix_keys = key_states[:, :, :prefix_len, :]
            attn = torch.matmul(window_q, prefix_keys.transpose(2, 3)) / math.sqrt(head_dim)

            if start < prefix_len:
                query_positions = torch.arange(start, end, device=query_states.device)
                key_positions = torch.arange(prefix_len, device=query_states.device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)

            attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_sum = attn.sum(dim=-2)  # [B, H, prefix_len]

            # Normalize: scale so each window's scores are comparable regardless
            # of how many prefix positions it can see.  Without this, early windows
            # (which can only attend to a small fraction of the prefix due to
            # causal masking) have near-zero scores and are effectively ignored.
            norm = attn_sum.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_sum = attn_sum / norm * self.window_size  # scale to window_size

            all_attn_sums.append(attn_sum)

        aggregated = torch.stack(all_attn_sums, dim=0).max(dim=0).values
        return aggregated

    # =========================================================================
    # Improvement 3: Critical Token (Spike) Protection
    # Detects attention spikes — positions whose attention is significantly
    # above the local average — and guarantees their inclusion, even if
    # pooling would otherwise smooth them away.
    # =========================================================================
    def _identify_critical_spikes(self, attn_weights_sum):
        """
        Find positions with attention spikes (significantly above local average).
        These positions likely represent isolated critical tokens (entities, numbers).

        Args:
            attn_weights_sum: [B, H, prefix_len] summed attention scores

        Returns:
            spike_mask: [B, H, prefix_len] boolean mask of spike positions
        """
        wide_kernel = min(self.kernel_size * 3, attn_weights_sum.shape[-1])
        if wide_kernel % 2 == 0:
            wide_kernel += 1  # ensure odd kernel

        # Compute local average
        local_avg = F.avg_pool1d(
            attn_weights_sum,
            kernel_size=wide_kernel,
            padding=wide_kernel // 2,
            stride=1
        )
        # Compute local std via E[X^2] - E[X]^2
        local_sq_avg = F.avg_pool1d(
            attn_weights_sum ** 2,
            kernel_size=wide_kernel,
            padding=wide_kernel // 2,
            stride=1
        )
        local_std = (local_sq_avg - local_avg ** 2).clamp(min=1e-10).sqrt()

        # A spike = position where attention > local_avg + 2 * local_std
        spike_mask = attn_weights_sum > (local_avg + 2.0 * local_std)

        return spike_mask

    # =========================================================================
    # Improvement 1: Weighted Pooling
    # Blends avg-pool (captures neighborhoods) and max-pool (captures peaks),
    # weighted by attention consistency across the observation window.
    # =========================================================================
    def _weighted_pooling(self, attn_weights_sum, attn_weights_prefix):
        """
        Combines avg and max pooling, weighted by per-position consistency
        (inverse variance) of attention across the observation window queries.

        Args:
            attn_weights_sum:    [B, H, prefix_len] summed attention
            attn_weights_prefix: [B, H, W, prefix_len] raw attention from obs window to prefix

        Returns:
            attn_cache: [B, H, prefix_len] pooled attention scores
        """
        # Avg pool: captures clustered important regions
        avg_pooled = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
        # Max pool: captures isolated peaks
        max_pooled = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)

        # Consistency: low variance across obs queries = more reliable signal
        attn_var = attn_weights_prefix.var(dim=-2)  # [B, H, prefix_len]
        consistency = 1.0 / (1.0 + attn_var)
        # Normalize so consistency scores average to 1.0 (no net scaling)
        consistency = consistency / (consistency.mean(dim=-1, keepdim=True) + 1e-8)

        # Blend: 50/50 avg+max, modulated by consistency
        alpha = 0.5
        attn_cache = (alpha * avg_pooled + (1 - alpha) * max_pooled) * consistency

        return attn_cache

    def _aggregate_gqa_scores(self, scores, num_key_value_groups):
        """Aggregate attention scores across query heads that share the same KV head.

        For GQA models (e.g. 32 query heads, 8 KV heads, groups=4), each set of
        4 query heads shares one KV head. We mean-aggregate their scores so the
        index selection reflects all query heads in the group.

        Args:
            scores: [B, H_expanded, prefix_len]
            num_key_value_groups: number of query heads per KV head

        Returns:
            aggregated: [B, H_kv, prefix_len]
        """
        if num_key_value_groups <= 1:
            return scores
        bsz, num_expanded, seq_len = scores.shape
        num_kv_heads = num_expanded // num_key_value_groups
        scores = scores.view(bsz, num_kv_heads, num_key_value_groups, seq_len)
        return scores.mean(dim=2)  # [B, H_kv, prefix_len]

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            budget = self.max_capacity_prompt - self.window_size
            prefix_len = q_len - self.window_size

            # =================================================================
            # Step 1: Compute attention scores
            # =================================================================
            if self.num_obs_windows > 1:
                attn_weights_sum = self._compute_multi_window_attention(
                    query_states, key_states, head_dim
                )
                attn_weights_prefix = None
                if self.pooling == 'weighted':
                    end_q = query_states[:, :, -self.window_size:, :]
                    prefix_k = key_states[:, :, :prefix_len, :]
                    attn_raw = torch.matmul(end_q, prefix_k.transpose(2, 3)) / math.sqrt(head_dim)
                    attn_raw = nn.functional.softmax(attn_raw, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_weights_prefix = attn_raw
            else:
                attn_weights = torch.matmul(
                    query_states[..., -self.window_size:, :],
                    key_states.transpose(2, 3)
                ) / math.sqrt(head_dim)

                mask = torch.full((self.window_size, self.window_size),
                                  torch.finfo(attn_weights.dtype).min,
                                  device=attn_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(attn_weights.device)
                causal_mask = mask[None, None, :, :]

                attn_weights[:, :, -self.window_size:, -self.window_size:] += causal_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                                     dtype=torch.float32).to(query_states.dtype)

                attn_weights_prefix = attn_weights[:, :, -self.window_size:, :-self.window_size]
                attn_weights_sum = attn_weights_prefix.sum(dim=-2)  # [B, H, prefix_len]

            # =================================================================
            # Step 1b: Aggregate across GQA groups so all query heads sharing
            # the same KV head vote together for which positions to keep.
            # =================================================================
            attn_weights_sum_agg = self._aggregate_gqa_scores(attn_weights_sum, num_key_value_groups)
            if attn_weights_prefix is not None and num_key_value_groups > 1:
                b, h_exp, w, pl = attn_weights_prefix.shape
                h_kv = h_exp // num_key_value_groups
                attn_weights_prefix_agg = attn_weights_prefix.view(
                    b, h_kv, num_key_value_groups, w, pl
                ).mean(dim=2)
            else:
                attn_weights_prefix_agg = attn_weights_prefix

            # =================================================================
            # Step 2: Apply pooling (Improvement 1: weighted pooling option)
            # =================================================================
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum_agg, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum_agg, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'weighted':
                if attn_weights_prefix_agg is not None:
                    attn_cache = self._weighted_pooling(attn_weights_sum_agg, attn_weights_prefix_agg)
                else:
                    avg_pooled = F.avg_pool1d(attn_weights_sum_agg, kernel_size=self.kernel_size,
                                              padding=self.kernel_size // 2, stride=1)
                    max_pooled = F.max_pool1d(attn_weights_sum_agg, kernel_size=self.kernel_size,
                                              padding=self.kernel_size // 2, stride=1)
                    attn_cache = 0.5 * avg_pooled + 0.5 * max_pooled
            else:
                raise ValueError('Pooling method not supported')

            # =================================================================
            # Step 3: Select indices (Improvement 3: spike protection)
            # Operates on KV-head-level scores after GQA aggregation.
            # =================================================================
            if self.protect_spikes:
                spike_mask = self._identify_critical_spikes(attn_weights_sum_agg)
                num_spikes_per_head = spike_mask.sum(dim=-1)  # [B, H_kv]
                max_spike_budget = int(budget * self.spike_reserve_ratio)

                spike_budget = min(max_spike_budget, int(num_spikes_per_head.max().item()))
                spike_budget = max(spike_budget, 0)

                if spike_budget > 0:
                    spike_scores = attn_weights_sum_agg.clone()
                    spike_scores[~spike_mask] = -float('inf')

                    actual_spike_budget = min(spike_budget, spike_scores.shape[-1])
                    _, spike_indices = spike_scores.topk(actual_spike_budget, dim=-1)

                    attn_cache_modified = attn_cache.clone()
                    attn_cache_modified.scatter_(2, spike_indices, float('-inf'))

                    main_budget = budget - actual_spike_budget
                    _, main_indices = attn_cache_modified.topk(main_budget, dim=-1)

                    indices = torch.cat([spike_indices, main_indices], dim=-1)
                    indices = indices.sort(dim=-1).values
                else:
                    indices = attn_cache.topk(budget, dim=-1).indices
            else:
                indices = attn_cache.topk(budget, dim=-1).indices

            # =================================================================
            # Step 4: Gather selected KV pairs and concatenate with obs window.
            # indices is [B, H_kv, budget] — expand back to expanded heads so
            # all query heads in a GQA group use the same selected positions.
            # =================================================================
            if num_key_value_groups > 1:
                # [B, H_kv, budget] → [B, H_kv, 1, budget] → [B, H_kv, G, budget] → [B, H_exp, budget]
                indices = indices.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1)
                indices = indices.reshape(bsz, num_heads, -1)

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states

def init_snapkv(self):
    if not hasattr(self.config, 'window_size'):
        self.config.window_size = 32
    if not hasattr(self.config, 'max_capacity_prompt'):
        self.config.max_capacity_prompt = 2048
    if not hasattr(self.config, 'kernel_size'):
        self.config.kernel_size = 5
    if not hasattr(self.config, 'pooling'):
        self.config.pooling = 'avgpool'
    if not hasattr(self.config, 'num_obs_windows'):
        self.config.num_obs_windows = 1
    if not hasattr(self.config, 'protect_spikes'):
        self.config.protect_spikes = False
    if not hasattr(self.config, 'spike_reserve_ratio'):
        self.config.spike_reserve_ratio = 0.1

    # Only create if missing, or if config changed since last creation.
    _cfg_key = (
        self.config.window_size, self.config.max_capacity_prompt,
        self.config.kernel_size, self.config.pooling,
        self.config.num_obs_windows, self.config.protect_spikes,
        self.config.spike_reserve_ratio,
    )
    if not hasattr(self, "kv_cluster") or getattr(self, "_kv_cluster_cfg", None) != _cfg_key:
        self.kv_cluster = SnapKVCluster(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            num_obs_windows=self.config.num_obs_windows,
            protect_spikes=self.config.protect_spikes,
            spike_reserve_ratio=self.config.spike_reserve_ratio,
        )
        self._kv_cluster_cfg = _cfg_key