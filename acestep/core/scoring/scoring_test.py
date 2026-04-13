"""Unit tests for DTW utilities and LM score pure functions."""

import math
import unittest

import numpy as np
import torch

from acestep.core.scoring._dtw import dtw_cpu, median_filter
from acestep.core.scoring.lm_score import (
    pmi_score,
    pmi_to_normalized_score,
    calculate_reward_score,
    _load_scoring_model_context,
)


class DtwCpuTests(unittest.TestCase):
    """Tests for the Numba-optimized DTW implementation."""

    def test_identity_cost_matrix(self):
        """DTW on a diagonal-zero cost matrix should follow the diagonal."""
        n = 4
        cost = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(cost, 0.0)
        text_idx, time_idx = dtw_cpu(-cost)
        # Path should be monotonically non-decreasing
        self.assertTrue(np.all(np.diff(text_idx) >= 0))
        self.assertTrue(np.all(np.diff(time_idx) >= 0))

    def test_single_element(self):
        """DTW on a 1x1 matrix should return a single-step path."""
        cost = np.array([[0.5]], dtype=np.float64)
        text_idx, time_idx = dtw_cpu(-cost)
        self.assertEqual(text_idx.tolist(), [0])
        self.assertEqual(time_idx.tolist(), [0])

    def test_rectangular_matrix(self):
        """DTW should handle non-square matrices."""
        cost = np.zeros((2, 5), dtype=np.float64)
        text_idx, time_idx = dtw_cpu(-cost)
        # Path must cover both rows and all columns
        self.assertIn(0, text_idx)
        self.assertIn(1, text_idx)
        self.assertEqual(time_idx[-1], 4)


class MedianFilterTests(unittest.TestCase):
    """Tests for the median filter utility."""

    def test_identity_with_width_one(self):
        """Filter width 1 should return the input unchanged."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = median_filter(x, filter_width=1)
        torch.testing.assert_close(result, x)

    def test_smoothing_effect(self):
        """Filter width > 1 should smooth spike values."""
        x = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]])
        result = median_filter(x, filter_width=3)
        # The spike at index 2 should be reduced
        self.assertLess(result[0, 2].item(), 10.0)

    def test_short_input_passthrough(self):
        """Inputs shorter than pad width should be returned as-is."""
        x = torch.tensor([[1.0]])
        result = median_filter(x, filter_width=5)
        torch.testing.assert_close(result, x)


class PmiScoreTests(unittest.TestCase):
    """Tests for the PMI pure functions."""

    def test_positive_pmi(self):
        """Conditional > unconditional should yield positive PMI."""
        result = pmi_score(-1.0, -2.0)
        self.assertAlmostEqual(result, 1.0)

    def test_zero_pmi(self):
        """Equal log probs should yield zero PMI."""
        result = pmi_score(-1.5, -1.5)
        self.assertAlmostEqual(result, 0.0)

    def test_negative_pmi(self):
        """Conditional < unconditional should yield negative PMI."""
        result = pmi_score(-3.0, -1.0)
        self.assertAlmostEqual(result, -2.0)


class PmiNormalizedScoreTests(unittest.TestCase):
    """Tests for the PMI-to-sigmoid normalization."""

    def test_zero_pmi_gives_half(self):
        """PMI of zero should map to exactly 0.5."""
        self.assertAlmostEqual(pmi_to_normalized_score(0.0), 0.5)

    def test_positive_pmi_above_half(self):
        """Positive PMI should map above 0.5."""
        self.assertGreater(pmi_to_normalized_score(1.0), 0.5)

    def test_negative_pmi_below_half(self):
        """Negative PMI should map below 0.5."""
        self.assertLess(pmi_to_normalized_score(-1.0), 0.5)

    def test_bounded_zero_one(self):
        """Output should always be in [0, 1]."""
        for pmi_val in [-100, -10, -1, 0, 1, 10, 100]:
            score = pmi_to_normalized_score(float(pmi_val), scale=1.0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class RewardScoreTests(unittest.TestCase):
    """Tests for the reward score aggregation."""

    def test_all_components_present(self):
        """With all three components, result should be a weighted average."""
        scores = {"caption": 0.8, "lyrics": 0.6, "bpm": 0.9}
        total, _ = calculate_reward_score(scores)
        self.assertGreater(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_no_scores_returns_zero(self):
        """Empty scores dict should return zero reward."""
        total, explanation = calculate_reward_score({})
        self.assertEqual(total, 0.0)

    def test_caption_only(self):
        """With only caption, reward should equal the caption score."""
        scores = {"caption": 0.75}
        total, _ = calculate_reward_score(scores)
        self.assertAlmostEqual(total, 0.75, places=2)

    def test_metadata_aggregation(self):
        """Multiple metadata fields should be averaged into one component."""
        scores = {"bpm": 1.0, "duration": 0.5}
        total, _ = calculate_reward_score(scores)
        # Only metadata component present, so total = avg(1.0, 0.5) = 0.75
        self.assertAlmostEqual(total, 0.75, places=2)


class LoadScoringModelContextTests(unittest.TestCase):
    """Regression tests for the Autoscore lifecycle contract (issue #1081).

    These assert three invariants that protect against the MLX unified-
    memory leak:

    1. ``_load_scoring_model_context`` performs exactly one load/offload
       cycle per outermost entry, regardless of how deeply nested the
       inner re-entries are.
    2. Nested re-entries from the same thread are no-ops and do not
       trigger additional CPU↔accelerator migrations.
    3. On the MLX backend with ``offload_to_cpu=True`` the cached HF
       scoring model is released on outermost exit so the ~8 GB duplicate
       PyTorch copy does not remain resident between generations.
    """

    class _FakeModel:
        def __init__(self):
            self.device_calls = []

        def to(self, device):
            self.device_calls.append(str(device))
            return self

    class _FakeHandler:
        def __init__(self, backend, offload=True):
            self.llm_backend = backend
            self.llm_initialized = True
            self.offload_to_cpu = offload
            self.device = "cuda"  # synthetic; _FakeModel.to is a recorder
            self._hf_model_for_scoring = LoadScoringModelContextTests._FakeModel()
            self.get_calls = 0

        def get_hf_model_for_scoring(self):
            self.get_calls += 1
            return self._hf_model_for_scoring

    def test_single_load_offload_per_outer_context_mlx(self):
        """One outer entry should trigger exactly one load and one offload."""
        handler = self._FakeHandler("mlx")
        model = handler._hf_model_for_scoring
        with _load_scoring_model_context(handler):
            pass
        # Exactly one load (to accelerator) and one offload (to cpu).
        self.assertEqual(
            model.device_calls, ["cuda", "cpu"],
            "expected exactly one load+offload cycle",
        )

    def test_nested_entries_are_noops_mlx(self):
        """Nested re-entries must not move the model again."""
        handler = self._FakeHandler("mlx")
        model = handler._hf_model_for_scoring
        with _load_scoring_model_context(handler):
            calls_after_outer_load = list(model.device_calls)
            # Deep nesting (simulating _get_logits_and_target_for_scoring
            # called many times per Autoscore pass).
            with _load_scoring_model_context(handler):
                with _load_scoring_model_context(handler):
                    # Inner contexts must not migrate the model again.
                    self.assertEqual(model.device_calls, calls_after_outer_load)
        # After the outermost exit the handler has dropped the cached model,
        # so use ``model`` (the captured reference) to verify the offload.
        self.assertEqual(
            model.device_calls, ["cuda", "cpu"],
            "nested entries should not add extra migrations",
        )

    def test_mlx_outermost_exit_drops_cached_model(self):
        """On MLX+offload, outer exit must clear ``_hf_model_for_scoring``."""
        handler = self._FakeHandler("mlx", offload=True)
        self.assertIsNotNone(handler._hf_model_for_scoring)
        with _load_scoring_model_context(handler):
            # Still cached while we're inside the context.
            self.assertIsNotNone(handler._hf_model_for_scoring)
        # Released after outermost exit so unified memory is returned to OS.
        self.assertIsNone(handler._hf_model_for_scoring)

    def test_vllm_outermost_exit_keeps_cached_model(self):
        """vllm backend must NOT drop the cached HF model (CUDA is fine)."""
        handler = self._FakeHandler("vllm", offload=True)
        cached = handler._hf_model_for_scoring
        with _load_scoring_model_context(handler):
            pass
        # vllm keeps the cached HF scoring model between passes; the MLX-
        # specific release path must not fire here.
        self.assertIs(handler._hf_model_for_scoring, cached)

    def test_mlx_no_offload_keeps_cached_model(self):
        """Without offload_to_cpu the MLX release path must not fire."""
        handler = self._FakeHandler("mlx", offload=False)
        cached = handler._hf_model_for_scoring
        with _load_scoring_model_context(handler):
            pass
        # Not offloading means no load/offload transitions and no drop.
        self.assertIs(handler._hf_model_for_scoring, cached)
        self.assertEqual(cached.device_calls, [])

    def test_get_hf_called_only_by_outermost_entry(self):
        """Nested entries must not re-query ``get_hf_model_for_scoring``."""
        handler = self._FakeHandler("mlx")
        with _load_scoring_model_context(handler):
            with _load_scoring_model_context(handler):
                with _load_scoring_model_context(handler):
                    pass
        # Outer entry queries once; nested entries are pure no-ops.
        self.assertEqual(handler.get_calls, 1)


if __name__ == "__main__":
    unittest.main()
