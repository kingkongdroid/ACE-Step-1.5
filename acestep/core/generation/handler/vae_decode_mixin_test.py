"""Unit tests for ``VaeDecodeMixin`` orchestration behavior."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.core.generation.handler.vae_decode import _TILED_DECODE_MIN_FREE_VRAM_GB
from acestep.core.generation.handler.vae_decode_test_helpers import _DecodeHost

# VAE_DECODE_MAX_CHUNK_SIZE is defined in MemoryUtilsMixin; mirror it here so
# the test host and test logic share the same constant without importing the
# full handler package (which has heavy optional deps).
_VAE_DECODE_MAX_CHUNK_SIZE = 512


class _DecodeHostWithMaxChunk(_DecodeHost):
    """Extend base host with VAE_DECODE_MAX_CHUNK_SIZE so tiled_decode can read it."""

    VAE_DECODE_MAX_CHUNK_SIZE = _VAE_DECODE_MAX_CHUNK_SIZE


def _make_cuda_vae():
    """Build a VAE mock whose parameters() reports device.type == 'cuda'."""
    fake_device = MagicMock()
    fake_device.type = "cuda"
    fake_device.__str__ = lambda self: "cuda"

    param = MagicMock()
    param.device = fake_device

    vae = MagicMock()
    vae.parameters.return_value = iter([param])
    return vae


class _CudaDecodeHost(_DecodeHostWithMaxChunk):
    """Extend _DecodeHost with minimal CPU-forced decode stubs for VRAM guard tests."""

    def __init__(self):
        """Configure host as a CUDA device with VAE tracking."""
        super().__init__()
        self.device = "cuda"
        self.vae = _make_cuda_vae()
        self._recursive_to_device_calls = []
        self._empty_cache_calls = 0
        self._tiled_inner_called = False

    def _recursive_to_device(self, model, device, dtype=None):
        """Record device migration requests from the VRAM guard."""
        # Re-build the VAE mock for the next parameters() call so restoration works
        self.vae = _make_cuda_vae()
        self._recursive_to_device_calls.append(str(device))

    def _get_vae_dtype(self, device=None):
        """Return float32 for any device in tests."""
        return torch.float32

    def _empty_cache(self):
        """Count cache-empty calls from the VRAM guard."""
        self._empty_cache_calls += 1

    def _tiled_decode_inner(self, latents, chunk_size, overlap, offload_wav_to_cpu):
        """Record call and return sentinel."""
        self._tiled_inner_called = True
        self.recorded["chunk_size"] = chunk_size
        self.recorded["offload"] = offload_wav_to_cpu
        return torch.ones(1, 2, 8)


# Patch target for get_effective_free_vram_gb: must be the name as bound in
# vae_decode.py (module-level import), not the source location.
_VRAM_PATCH = "acestep.core.generation.handler.vae_decode.get_effective_free_vram_gb"
# Patch targets for the memory_utils helpers (inline import inside tiled_decode).
_CUDA_DEVICE_PATCH = "acestep.core.generation.handler.memory_utils._is_cuda_device"
_CUDA_IDX_PATCH = "acestep.core.generation.handler.memory_utils._cuda_device_index"


class VaeDecodeMixinTests(unittest.TestCase):
    """Verify decode orchestrator paths, fallback policy, and error propagation."""

    def test_tiled_decode_reduces_mps_chunk_and_overlap(self):
        """MPS path clamps chunk/overlap to safe configured limits."""
        host = _DecodeHostWithMaxChunk()
        out = host.tiled_decode(torch.zeros(1, 4, 128), chunk_size=64, overlap=16)
        self.assertEqual(host.recorded["chunk_size"], 32)
        self.assertEqual(host.recorded["overlap"], 8)
        self.assertFalse(host.recorded["offload"])
        self.assertEqual(tuple(out.shape), (1, 2, 8))

    def test_tiled_decode_mps_runtime_failure_uses_cpu_fallback(self):
        """MPS runtime failures fallback to CPU decode helper."""
        host = _DecodeHostWithMaxChunk()

        def _raise(*args, **kwargs):
            """Simulate runtime failure inside tiled decode implementation."""
            _ = args, kwargs
            raise RuntimeError("mps decode failure")

        host._tiled_decode_inner = _raise
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 8), 2.0)))

    def test_tiled_decode_uses_mlx_fast_path_when_available(self):
        """MLX decode should short-circuit before PyTorch path when enabled."""
        host = _DecodeHostWithMaxChunk()
        host.use_mlx_vae = True
        host.mlx_vae = object()
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 6), 3.0)))

    def test_tiled_decode_falls_back_when_mlx_decode_fails(self):
        """MLX decode errors should fallback to normal tiled decode path."""
        host = _DecodeHostWithMaxChunk()
        host.use_mlx_vae = True
        host.mlx_vae = object()

        def _mlx_raise(_latents):
            """Raise MLX failure to exercise fallback path."""
            raise ValueError("mlx failed")

        host._mlx_vae_decode = _mlx_raise
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertEqual(tuple(out.shape), (1, 2, 8))
        self.assertEqual(host.recorded["chunk_size"], 32)

    def test_tiled_decode_non_mps_runtime_error_is_raised(self):
        """Non-MPS runtime errors should bubble to caller unchanged."""
        host = _DecodeHostWithMaxChunk()
        host.device = "cuda"

        def _raise(*args, **kwargs):
            """Raise runtime failure for non-MPS path assertion."""
            _ = args, kwargs
            raise RuntimeError("cuda decode failure")

        host._tiled_decode_inner = _raise
        with self.assertRaises(RuntimeError):
            host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)

    # ------------------------------------------------------------------
    # VRAM guard tests (added to fix GPU freeze with tight memory)
    # ------------------------------------------------------------------

    def test_vram_guard_moves_vae_to_cpu_when_critically_low(self):
        """When chunk_size is minimum and VRAM is critically low, VAE moves to CPU."""
        host = _CudaDecodeHost()
        min_chunk = _VAE_DECODE_MAX_CHUNK_SIZE // 4  # 128

        with patch(_CUDA_DEVICE_PATCH, return_value=True), \
             patch(_CUDA_IDX_PATCH, return_value=0), \
             patch(_VRAM_PATCH, return_value=_TILED_DECODE_MIN_FREE_VRAM_GB - 0.1):
            host.tiled_decode(torch.zeros(1, 4, 256), chunk_size=min_chunk, overlap=32)

        # _recursive_to_device must have been called to move VAE to CPU
        cpu_moves = [c for c in host._recursive_to_device_calls if "cpu" in c.lower()]
        self.assertGreater(len(cpu_moves), 0, "VAE should have been moved to CPU")

    def test_vram_guard_skipped_when_vram_is_sufficient(self):
        """When VRAM is above threshold, the VRAM guard should not move VAE."""
        host = _CudaDecodeHost()
        min_chunk = _VAE_DECODE_MAX_CHUNK_SIZE // 4  # 128

        with patch(_CUDA_DEVICE_PATCH, return_value=True), \
             patch(_CUDA_IDX_PATCH, return_value=0), \
             patch(_VRAM_PATCH, return_value=_TILED_DECODE_MIN_FREE_VRAM_GB + 1.0):
            host.tiled_decode(torch.zeros(1, 4, 256), chunk_size=min_chunk, overlap=32)

        # No device moves should occur when VRAM is sufficient
        self.assertEqual(
            len(host._recursive_to_device_calls),
            0,
            "VAE should NOT have been moved when VRAM is sufficient",
        )

    def test_vram_guard_not_triggered_for_large_chunk_size(self):
        """VRAM guard should not run when chunk_size is above minimum."""
        host = _CudaDecodeHost()
        large_chunk = _VAE_DECODE_MAX_CHUNK_SIZE  # 512, well above minimum

        # Even with critically low VRAM, guard must not fire for large chunks
        with patch(_VRAM_PATCH, return_value=0.1):
            host.tiled_decode(torch.zeros(1, 4, 256), chunk_size=large_chunk, overlap=32)

        self.assertEqual(
            len(host._recursive_to_device_calls),
            0,
            "VRAM guard should not activate for large chunk sizes",
        )

    def test_vram_guard_vae_restored_after_cpu_decode(self):
        """After CPU-forced tiled decode, VAE must be restored (at least 2 device moves)."""
        host = _CudaDecodeHost()
        min_chunk = _VAE_DECODE_MAX_CHUNK_SIZE // 4

        with patch(_CUDA_DEVICE_PATCH, return_value=True), \
             patch(_CUDA_IDX_PATCH, return_value=0), \
             patch(_VRAM_PATCH, return_value=0.5):  # critically low
            host.tiled_decode(torch.zeros(1, 4, 256), chunk_size=min_chunk, overlap=32)

        # There should be at least two _recursive_to_device calls: move to CPU then restore
        self.assertGreaterEqual(
            len(host._recursive_to_device_calls),
            2,
            "VAE should be moved to CPU and then restored",
        )
        # First call moves to CPU
        first_device = host._recursive_to_device_calls[0].lower()
        self.assertIn("cpu", first_device, f"First move should be to CPU, got '{first_device}'")
        # Last call restores to original device (cuda)
        last_device = host._recursive_to_device_calls[-1].lower()
        self.assertIn(
            "cuda",
            last_device,
            f"VAE should be restored to cuda, got '{last_device}'",
        )


if __name__ == "__main__":
    unittest.main()
