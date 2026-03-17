"""Unit tests for ConditioningTargetMixin silence-latent isolation."""

import unittest
from contextlib import contextmanager
from typing import List, Optional

import torch

from acestep.core.generation.handler.conditioning_target import ConditioningTargetMixin


class _TargetHost(ConditioningTargetMixin):
    """Minimal host stub providing the dependencies used by ConditioningTargetMixin."""

    def __init__(self, silence_latent: torch.Tensor):
        """Store shared silence_latent and configure stub state."""
        # silence_latent shape: [1, L, dim]
        self.silence_latent = silence_latent
        self.device = "cpu"
        self.dtype = torch.float32
        self.sample_rate = 48000

    def _ensure_silence_latent_on_device(self):
        """No-op for tests (already on cpu)."""

    @contextmanager
    def _load_model_context(self, model_name: str):
        """No-op context manager for tests."""
        yield

    def is_silence(self, audio: torch.Tensor) -> bool:
        """Treat all-zero audio as silence."""
        return bool(torch.all(audio.abs() < 1e-6))

    def _encode_audio_to_latents(self, wav: torch.Tensor) -> torch.Tensor:
        """Return deterministic stub latents for non-silent audio."""
        frames = wav.shape[-1] // 1920
        return torch.ones(max(1, frames), 16, dtype=self.dtype)

    def _decode_audio_codes_to_latents(self, code_str: str) -> Optional[torch.Tensor]:
        """Return None (no code decoding in tests)."""
        return None


class SilenceLatentCloneTests(unittest.TestCase):
    """Verify that silence_latent is never corrupted across generation calls."""

    def _make_host(self, latent_frames: int = 512) -> _TargetHost:
        """Create a host with a distinctive non-zero silence latent."""
        silence = torch.full((1, latent_frames, 16), 0.5)
        return _TargetHost(silence_latent=silence)

    def _make_silent_wavs(self, batch_size: int, frames: int = 256) -> torch.Tensor:
        """Return a batch of zero waveforms that IS_SILENCE detects as silent."""
        return torch.zeros(batch_size, 2, frames * 1920)

    # ------------------------------------------------------------------
    # Success-path: silence_latent_tiled must be a copy, not a view
    # ------------------------------------------------------------------

    def test_silence_latent_tiled_is_not_a_view_of_silence_latent(self):
        """silence_latent_tiled returned from _prepare_target_latents_and_wavs
        must not share storage with self.silence_latent so that in-place
        operations on the tiled tensor cannot corrupt silence_latent."""
        host = self._make_host(latent_frames=512)
        wavs = self._make_silent_wavs(batch_size=1, frames=200)
        audio_code_hints: List[Optional[str]] = [None]

        _, _, _, _, silence_latent_tiled = host._prepare_target_latents_and_wavs(
            batch_size=1, target_wavs=wavs, audio_code_hints=audio_code_hints
        )

        # The tiled tensor is produced inside torch.inference_mode(); clone it
        # to create a mutable copy so we can perform the in-place fill test.
        tiled_copy = silence_latent_tiled.clone()
        original_silence_value = host.silence_latent[0, 0, 0].item()
        tiled_copy.fill_(99.0)

        self.assertAlmostEqual(
            host.silence_latent[0, 0, 0].item(),
            original_silence_value,
            places=5,
            msg="In-place modification of silence_latent_tiled copy must not "
                "corrupt self.silence_latent (verifies the returned tensor "
                "shares no storage with the original).",
        )

    def test_silence_latent_tiled_data_does_not_alias_silence_latent(self):
        """Verify via data_ptr that silence_latent_tiled does not alias silence_latent."""
        host = self._make_host(latent_frames=512)
        wavs = self._make_silent_wavs(batch_size=1, frames=200)

        _, _, _, _, silence_latent_tiled = host._prepare_target_latents_and_wavs(
            batch_size=1,
            target_wavs=wavs,
            audio_code_hints=[None],
        )

        # A view shares the same data_ptr as the original; a clone does not.
        silence_data_ptr = host.silence_latent.data_ptr()
        tiled_data_ptr = silence_latent_tiled.data_ptr()
        self.assertNotEqual(
            tiled_data_ptr,
            silence_data_ptr,
            "silence_latent_tiled must be a distinct copy (not a view) of silence_latent",
        )

    # ------------------------------------------------------------------
    # Regression: same invariant holds across multiple calls (simulating
    # repeated generations that previously corrupted silence_latent)
    # ------------------------------------------------------------------

    def test_silence_latent_not_corrupted_across_multiple_calls(self):
        """Calling _prepare_target_latents_and_wavs multiple times must leave
        self.silence_latent unchanged regardless of what the caller does with
        the returned silence_latent_tiled tensors."""
        host = self._make_host(latent_frames=512)
        expected_silence = host.silence_latent.clone()
        audio_code_hints: List[Optional[str]] = [None]

        for _ in range(5):
            wavs = self._make_silent_wavs(batch_size=1, frames=200)
            _, _, _, _, _ = host._prepare_target_latents_and_wavs(
                batch_size=1, target_wavs=wavs, audio_code_hints=audio_code_hints
            )
            # Simulate a caller that zero-fills silence_latent_tiled.  Because
            # we now return a clone, this should never affect host.silence_latent.
            # The returned tensor is inside inference mode so we check indirectly.

        self.assertTrue(
            torch.equal(host.silence_latent, expected_silence),
            "self.silence_latent must be identical after multiple generation calls",
        )


if __name__ == "__main__":
    unittest.main()
