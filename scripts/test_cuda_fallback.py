#!/usr/bin/env python
"""
Test script to verify CUDA operations and their PyTorch fallbacks.

Usage:
    python scripts/test_cuda_fallback.py
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import warnings


def test_deformable_attention():
    """Test deformable attention CUDA/fallback."""
    print("\n" + "=" * 70)
    print("Testing Deformable Attention")
    print("=" * 70)

    try:
        import MultiScaleDeformableAttention as MSDA
        print("✅ CUDA extension loaded: MultiScaleDeformableAttention")
        cuda_available = True
    except ImportError:
        print("⚠️  CUDA extension not available: MultiScaleDeformableAttention")
        print("   Will use PyTorch fallback")
        cuda_available = False

    # Test import of wrapper
    try:
        from model.det.rcbevdet.ops.ms_deform_attn import MSDeformAttn
        print("✅ MSDeformAttn module imported successfully")

        # Test instantiation
        attn = MSDeformAttn(d_model=256, n_levels=1, n_heads=8, n_points=4)
        print("✅ MSDeformAttn instantiated")

        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attn = attn.to(device)

        batch_size = 2
        num_queries = 100
        num_keys = 200

        query = torch.randn(batch_size, num_queries, 256, device=device)
        reference_points = torch.rand(batch_size, num_queries, 1, 2, device=device)
        input_flatten = torch.randn(batch_size, num_keys, 256, device=device)
        input_spatial_shapes = torch.tensor([[10, 20]], device=device)
        input_level_start_index = torch.tensor([0], device=device)

        with torch.no_grad():
            output = attn(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
            )

        print(f"✅ Forward pass successful: output shape = {output.shape}")
        assert output.shape == (batch_size, num_queries, 256)
        print("✅ Output shape correct")

        return True, cuda_available

    except Exception as e:
        print(f"❌ Error testing deformable attention: {e}")
        import traceback
        traceback.print_exc()
        return False, cuda_available


def test_bev_pool():
    """Test BEV pool v2 CUDA/fallback."""
    print("\n" + "=" * 70)
    print("Testing BEV Pool V2")
    print("=" * 70)

    try:
        from model.det.rcbevdet.ops.bev_pool_v2 import bev_pool_v2_ext
        print("✅ CUDA extension loaded: bev_pool_v2_ext")
        cuda_available = True
    except ImportError:
        print("⚠️  CUDA extension not available: bev_pool_v2_ext")
        print("   Will use PyTorch fallback")
        cuda_available = False

    # Test import of wrapper
    try:
        from model.det.rcbevdet.ops.bev_pool_v2.bev_pool import bev_pool_v2

        print("✅ bev_pool_v2 function imported successfully")

        # Simple test case
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        depth = torch.rand(1, 2, 4, 4, device=device)
        feat = torch.rand(1, 4, 4, 64, device=device)

        # Create simple ranks for testing
        n_points = 8
        ranks_depth = torch.randint(0, 2, (n_points,), device=device)
        ranks_feat = torch.arange(n_points, device=device)
        ranks_bev = torch.randint(0, 16, (n_points,), device=device)

        # Create intervals
        interval_starts = torch.tensor([0, 4], device=device, dtype=torch.int32)
        interval_lengths = torch.tensor([4, 4], device=device, dtype=torch.int32)

        bev_feat_shape = (1, 1, 4, 4, 64)

        with torch.no_grad():
            output = bev_pool_v2(
                depth,
                feat,
                ranks_depth.int(),
                ranks_feat.int(),
                ranks_bev.int(),
                bev_feat_shape,
                interval_starts,
                interval_lengths,
            )

        print(f"✅ Forward pass successful: output shape = {output.shape}")
        expected_shape = (1, 64, 1, 4, 4)  # (B, C, Z, Y, X)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("✅ Output shape correct")

        return True, cuda_available

    except Exception as e:
        print(f"❌ Error testing BEV pool: {e}")
        import traceback
        traceback.print_exc()
        return False, cuda_available


def test_model_factory():
    """Test that the model factory works with fallbacks."""
    print("\n" + "=" * 70)
    print("Testing RCBEVDet Model Factory")
    print("=" * 70)

    try:
        # Just test import for now
        from model.factory import rcbevdet

        print("✅ RCBEVDet factory module imported successfully")

        # Note: Full model test requires checkpoint and config
        print("ℹ️  Full model test requires checkpoint/config (skipped)")

        return True

    except Exception as e:
        print(f"❌ Error testing model factory: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("RCBEVDet CUDA Operations Test")
    print("=" * 70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests
    results = {}

    deform_ok, deform_cuda = test_deformable_attention()
    results["Deformable Attention"] = (deform_ok, deform_cuda)

    bev_ok, bev_cuda = test_bev_pool()
    results["BEV Pool V2"] = (bev_ok, bev_cuda)

    factory_ok = test_model_factory()
    results["Model Factory"] = (factory_ok, None)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = True
    cuda_count = 0
    fallback_count = 0

    for name, result in results.items():
        if result[1] is None:
            status = "✅ PASS" if result[0] else "❌ FAIL"
            mode = ""
        else:
            status = "✅ PASS" if result[0] else "❌ FAIL"
            mode = " (CUDA)" if result[1] else " (PyTorch fallback)"
            if result[1]:
                cuda_count += 1
            else:
                fallback_count += 1

        print(f"{name:25} {status}{mode}")
        if not result[0]:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed!")
        if cuda_count > 0:
            print(f"   - {cuda_count} CUDA extension(s) compiled")
        if fallback_count > 0:
            print(f"   - {fallback_count} PyTorch fallback(s) used")
    else:
        print("❌ Some tests failed. See errors above.")
        return 1

    print("\nℹ️  Note: PyTorch fallbacks are slower but work without compilation.")
    print("   To compile CUDA extensions: bash scripts/compile_rcbevdet_cuda_v2.sh")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
