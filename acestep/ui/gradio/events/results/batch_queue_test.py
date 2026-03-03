"""Unit tests for batch_queue module."""

import unittest

from acestep.ui.gradio.events.results.batch_queue import (
    store_batch_in_queue,
    update_batch_indicator,
    update_navigation_buttons,
)


class UpdateNavigationButtonsTests(unittest.TestCase):
    """Tests for update_navigation_buttons."""

    def test_first_batch(self):
        """At first batch, prev should be False."""
        can_prev, can_next = update_navigation_buttons(0, 3)
        self.assertFalse(can_prev)
        self.assertTrue(can_next)

    def test_last_batch(self):
        """At last batch, next should be False."""
        can_prev, can_next = update_navigation_buttons(2, 3)
        self.assertTrue(can_prev)
        self.assertFalse(can_next)

    def test_middle_batch(self):
        """At a middle batch, both should be True."""
        can_prev, can_next = update_navigation_buttons(1, 3)
        self.assertTrue(can_prev)
        self.assertTrue(can_next)

    def test_single_batch(self):
        """With only one batch, both should be False."""
        can_prev, can_next = update_navigation_buttons(0, 1)
        self.assertFalse(can_prev)
        self.assertFalse(can_next)


class StoreBatchInQueueTests(unittest.TestCase):
    """Tests for store_batch_in_queue."""

    def test_store_first_batch(self):
        """Storing the first batch should create a new queue entry."""
        queue = {}
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=0,
            audio_paths=["/tmp/audio1.mp3"],
            generation_info="test info",
            seeds="42",
        )
        self.assertIsInstance(result, dict)
        self.assertIn(0, result)
        self.assertEqual(result[0]["audio_paths"], ["/tmp/audio1.mp3"])
        self.assertEqual(result[0]["generation_info"], "test info")

    def test_store_preserves_existing_batches(self):
        """Storing a new batch should not remove existing batches."""
        queue = {
            0: {
                "audio_paths": ["/tmp/old.mp3"],
                "generation_info": "old info",
                "seeds": "1",
                "status": "completed",
            }
        }
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=1,
            audio_paths=["/tmp/new.mp3"],
            generation_info="new info",
            seeds="99",
        )
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertEqual(result[0]["audio_paths"], ["/tmp/old.mp3"])
        self.assertEqual(result[1]["audio_paths"], ["/tmp/new.mp3"])

    def test_store_with_scores_and_codes(self):
        """Storing with scores and codes should preserve them."""
        queue = {}
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=0,
            audio_paths=["/tmp/a.mp3"],
            generation_info="info",
            seeds="42",
            codes=["code1"],
            scores=["8.5/10"],
        )
        self.assertEqual(result[0]["codes"], ["code1"])
        self.assertEqual(result[0]["scores"], ["8.5/10"])

    def test_store_defaults_scores_to_empty(self):
        """Storing without scores should default to 8 empty strings."""
        queue = {}
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=0,
            audio_paths=["/tmp/a.mp3"],
            generation_info="info",
            seeds="42",
        )
        self.assertEqual(result[0]["scores"], [""] * 8)

    def test_store_frees_previous_extra_outputs_tensors(self):
        """Storing a new batch should free tensor extra_outputs from the previous batch."""
        import torch
        tensor = torch.zeros(2, 4)
        queue = {
            0: {
                "audio_paths": ["/tmp/old.mp3"],
                "generation_info": "old info",
                "seeds": "1",
                "status": "completed",
                "extra_outputs": {"pred_latents": tensor},
            }
        }
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=1,
            audio_paths=["/tmp/new.mp3"],
            generation_info="new info",
            seeds="99",
        )
        self.assertEqual(result[0]["extra_outputs"], {})
        self.assertIn(1, result)

    def test_store_frees_only_previous_batch_extra_outputs(self):
        """Only the immediately preceding batch's extra_outputs should be cleared."""
        import torch
        tensor_a = torch.ones(1)
        tensor_b = torch.ones(2)
        queue = {
            0: {"audio_paths": [], "extra_outputs": {"pred_latents": tensor_a}, "status": "completed"},
            1: {"audio_paths": [], "extra_outputs": {"pred_latents": tensor_b}, "status": "completed"},
        }
        store_batch_in_queue(
            batch_queue=queue,
            batch_index=2,
            audio_paths=["/tmp/c.mp3"],
            generation_info="info",
            seeds="7",
        )
        # Batch 1 (prev_index=1) should have its extra_outputs cleared
        self.assertEqual(queue[1]["extra_outputs"], {})
        # Batch 0 should be untouched
        self.assertIn("pred_latents", queue[0]["extra_outputs"])

    def test_store_first_batch_no_prev_cleanup(self):
        """Storing the first batch (index 0) should not fail when there is no previous batch."""
        queue = {}
        result = store_batch_in_queue(
            batch_queue=queue,
            batch_index=0,
            audio_paths=["/tmp/a.mp3"],
            generation_info="info",
            seeds="42",
        )
        self.assertIn(0, result)

    def test_store_clears_non_tensor_extra_outputs(self):
        """Storing a new batch should clear all extra_outputs from the previous batch, including non-tensor values."""
        queue = {
            0: {
                "audio_paths": [],
                "extra_outputs": {
                    "lrcs": ["[00:00.00] hello"],
                    "subtitles": [None],
                },
                "status": "completed",
            }
        }
        store_batch_in_queue(
            batch_queue=queue,
            batch_index=1,
            audio_paths=["/tmp/b.mp3"],
            generation_info="info",
            seeds="5",
        )
        # extra_outputs cleared entirely (the dict is replaced with {})
        self.assertEqual(queue[0]["extra_outputs"], {})


if __name__ == "__main__":
    unittest.main()
