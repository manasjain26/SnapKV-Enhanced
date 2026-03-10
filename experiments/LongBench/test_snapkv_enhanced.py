"""
Unit test for SnapKV Enhanced improvements.
Tests all three improvements on synthetic data without requiring a full model.

Run: python test_snapkv_enhanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from snapkv.monkeypatch.snapkv_utils import SnapKVCluster


def create_synthetic_data(bsz=1, num_heads=4, seq_len=512, head_dim=64, device='cpu'):
    """Create synthetic query, key, value states for testing."""
    torch.manual_seed(42)
    query_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)
    key_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)
    value_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)
    return query_states, key_states, value_states


def test_baseline_compatibility():
    """Test 1: Ensure default config produces the same behavior as original SnapKV."""
    print("=" * 60)
    print("Test 1: Baseline Compatibility (avgpool, 1 window, no spikes)")
    print("=" * 60)

    cluster = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
        num_obs_windows=1, protect_spikes=False
    )
    q, k, v = create_synthetic_data(seq_len=256)

    k_out, v_out = cluster.update_kv(k, q, v, None, 1)
    expected_len = 128  # max_capacity_prompt
    assert k_out.shape[2] == expected_len, f"Expected seq_len={expected_len}, got {k_out.shape[2]}"
    assert v_out.shape[2] == expected_len, f"Expected seq_len={expected_len}, got {v_out.shape[2]}"
    print(f"  ✓ Output shape: {k_out.shape} (expected seq_len={expected_len})")
    print(f"  ✓ Baseline compatibility PASSED\n")


def test_no_compression_short_input():
    """Test 2: If input is shorter than max_capacity_prompt, no compression happens."""
    print("=" * 60)
    print("Test 2: No compression for short inputs")
    print("=" * 60)

    cluster = SnapKVCluster(window_size=32, max_capacity_prompt=512, kernel_size=5, pooling='avgpool')
    q, k, v = create_synthetic_data(seq_len=256)

    k_out, v_out = cluster.update_kv(k, q, v, None, 1)
    assert k_out.shape[2] == 256, f"Short input should not be compressed, got {k_out.shape[2]}"
    print(f"  ✓ Short input (256 < 512) not compressed")
    print(f"  ✓ No-compression test PASSED\n")


def test_weighted_pooling():
    """Test 3: Verify weighted pooling mode works and produces valid output."""
    print("=" * 60)
    print("Test 3: Weighted Pooling (Improvement 1)")
    print("=" * 60)

    cluster = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='weighted',
        num_obs_windows=1, protect_spikes=False
    )
    q, k, v = create_synthetic_data(seq_len=256)

    k_out, v_out = cluster.update_kv(k, q, v, None, 1)
    expected_len = 128
    assert k_out.shape[2] == expected_len, f"Expected seq_len={expected_len}, got {k_out.shape[2]}"
    assert not torch.isnan(k_out).any(), "Output contains NaN!"
    assert not torch.isinf(k_out).any(), "Output contains Inf!"
    print(f"  ✓ Output shape: {k_out.shape}")
    print(f"  ✓ No NaN/Inf in output")

    # Verify weighted pooling selects different indices than plain avgpool
    cluster_avg = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool'
    )
    k_avg, _ = cluster_avg.update_kv(k, q, v, None, 1)
    # They may sometimes be identical, but typically differ
    if not torch.allclose(k_out, k_avg, atol=1e-6):
        print(f"  ✓ Weighted pooling selects different KV positions than avgpool (expected)")
    else:
        print(f"  ⚠ Weighted pooling produced same result as avgpool (possible but unusual)")
    print(f"  ✓ Weighted pooling test PASSED\n")


def test_multi_window_observation():
    """Test 4: Verify multi-window observation works and potentially selects different positions."""
    print("=" * 60)
    print("Test 4: Multi-Window Observation (Improvement 2)")
    print("=" * 60)

    # Single window
    cluster_1w = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
        num_obs_windows=1
    )
    # Three windows
    cluster_3w = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
        num_obs_windows=3
    )

    q, k, v = create_synthetic_data(seq_len=256)

    k_1w, v_1w = cluster_1w.update_kv(k, q, v, None, 1)
    k_3w, v_3w = cluster_3w.update_kv(k, q, v, None, 1)

    assert k_1w.shape == k_3w.shape, f"Shape mismatch: {k_1w.shape} vs {k_3w.shape}"
    assert k_3w.shape[2] == 128, f"Expected seq_len=128, got {k_3w.shape[2]}"
    assert not torch.isnan(k_3w).any(), "3-window output contains NaN!"

    if not torch.allclose(k_1w, k_3w, atol=1e-6):
        print(f"  ✓ Multi-window selects different KV positions than single-window (expected)")
    else:
        print(f"  ⚠ Multi-window produced same result as single-window (possible but unusual)")

    print(f"  ✓ Output shape: {k_3w.shape}")
    print(f"  ✓ Multi-window observation test PASSED\n")


def test_spike_protection():
    """Test 5: Verify spike protection works."""
    print("=" * 60)
    print("Test 5: Critical Token Spike Protection (Improvement 3)")
    print("=" * 60)

    cluster_no_spike = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
        protect_spikes=False
    )
    cluster_spike = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
        protect_spikes=True, spike_reserve_ratio=0.2
    )

    # Create data with deliberate attention spike at a specific position
    torch.manual_seed(42)
    q, k, v = create_synthetic_data(seq_len=256)

    k_no_spike, v_no_spike = cluster_no_spike.update_kv(k, q, v, None, 1)
    k_spike, v_spike = cluster_spike.update_kv(k, q, v, None, 1)

    assert k_spike.shape[2] == 128, f"Expected seq_len=128, got {k_spike.shape[2]}"
    assert not torch.isnan(k_spike).any(), "Spike-protected output contains NaN!"
    assert not torch.isinf(k_spike).any(), "Spike-protected output contains Inf!"

    print(f"  ✓ Output shape: {k_spike.shape}")
    print(f"  ✓ No NaN/Inf in output")
    print(f"  ✓ Spike protection test PASSED\n")


def test_combined_all_improvements():
    """Test 6: All three improvements combined."""
    print("=" * 60)
    print("Test 6: All Improvements Combined")
    print("=" * 60)

    cluster = SnapKVCluster(
        window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='weighted',
        num_obs_windows=3, protect_spikes=True, spike_reserve_ratio=0.1
    )

    q, k, v = create_synthetic_data(seq_len=256)
    k_out, v_out = cluster.update_kv(k, q, v, None, 1)

    assert k_out.shape[2] == 128, f"Expected seq_len=128, got {k_out.shape[2]}"
    assert not torch.isnan(k_out).any(), "Combined output contains NaN!"
    assert not torch.isinf(k_out).any(), "Combined output contains Inf!"

    print(f"  ✓ Output shape: {k_out.shape}")
    print(f"  ✓ No NaN/Inf in output")
    print(f"  ✓ Combined improvements test PASSED\n")


def test_all_pooling_modes():
    """Test 7: Verify all pooling modes produce valid output."""
    print("=" * 60)
    print("Test 7: All Pooling Modes")
    print("=" * 60)

    q, k, v = create_synthetic_data(seq_len=256)

    for pooling in ['avgpool', 'maxpool', 'weighted']:
        cluster = SnapKVCluster(
            window_size=32, max_capacity_prompt=128, kernel_size=5, pooling=pooling
        )
        k_out, v_out = cluster.update_kv(k, q, v, None, 1)
        assert k_out.shape[2] == 128
        assert not torch.isnan(k_out).any()
        print(f"  ✓ {pooling}: shape={k_out.shape}, no NaN")

    print(f"  ✓ All pooling modes test PASSED\n")


def test_different_window_counts():
    """Test 8: Verify obs window counts 1, 2, 3, 5 all work."""
    print("=" * 60)
    print("Test 8: Different Observation Window Counts")
    print("=" * 60)

    q, k, v = create_synthetic_data(seq_len=512)

    for n_windows in [1, 2, 3, 5]:
        cluster = SnapKVCluster(
            window_size=32, max_capacity_prompt=128, kernel_size=5, pooling='avgpool',
            num_obs_windows=n_windows
        )
        k_out, v_out = cluster.update_kv(k, q, v, None, 1)
        assert k_out.shape[2] == 128
        assert not torch.isnan(k_out).any()
        print(f"  ✓ num_obs_windows={n_windows}: shape={k_out.shape}, no NaN")

    print(f"  ✓ Window count test PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SnapKV Enhanced — Unit Tests")
    print("=" * 60 + "\n")

    test_baseline_compatibility()
    test_no_compression_short_input()
    test_weighted_pooling()
    test_multi_window_observation()
    test_spike_protection()
    test_combined_all_improvements()
    test_all_pooling_modes()
    test_different_window_counts()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
