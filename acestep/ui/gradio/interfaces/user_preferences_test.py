"""Unit tests for user preference persistence head-script generation."""

import importlib.util
from pathlib import Path
import unittest


def _load_module():
    """Load the target module directly by file path for isolated testing."""
    module_path = Path(__file__).with_name("user_preferences.py")
    spec = importlib.util.spec_from_file_location("user_preferences", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_MODULE = _load_module()
get_user_preferences_head = _MODULE.get_user_preferences_head
_load_preferences_script = _MODULE._load_preferences_script
_SCRIPT_PATH = Path(__file__).with_name("user_preferences.js")


class UserPreferencesHeadTests(unittest.TestCase):
    """Tests for browser script generation used by Gradio ``Blocks(head=...)``."""

    def test_external_script_asset_exists(self):
        """The externalized JavaScript asset should exist and be non-empty."""
        self.assertTrue(_SCRIPT_PATH.is_file())
        script_asset = _load_preferences_script()
        self.assertTrue(script_asset)

    def test_script_contains_localstorage_persistence(self):
        """Script should include localStorage save/restore logic."""
        script = get_user_preferences_head()
        self.assertIn("<script>", script)
        self.assertIn("localStorage", script)
        self.assertIn("acestep.ui.user_preferences", script)

    def test_script_contains_all_preference_elem_ids(self):
        """Script should reference all known preference element IDs."""
        script = get_user_preferences_head()
        expected_ids = [
            "acestep-audio-format",
            "acestep-mp3-bitrate",
            "acestep-mp3-sample-rate",
            "acestep-score-scale",
            "acestep-enable-normalization",
            "acestep-normalization-db",
            "acestep-fade-in-duration",
            "acestep-fade-out-duration",
            "acestep-latent-shift",
            "acestep-latent-rescale",
            "acestep-lm-batch-chunk-size",
        ]
        for elem_id in expected_ids:
            self.assertIn(elem_id, script, f"Missing elem_id: {elem_id}")

    def test_script_handles_all_control_types(self):
        """Script should handle dropdown, slider, checkbox, and number inputs."""
        script = get_user_preferences_head()
        self.assertIn('"dropdown"', script)
        self.assertIn('"slider"', script)
        self.assertIn('"checkbox"', script)
        self.assertIn('"number"', script)

    def test_script_debounces_saves(self):
        """Script should debounce save operations to avoid excessive I/O."""
        script = get_user_preferences_head()
        self.assertIn("DEBOUNCE_MS", script)
        self.assertIn("clearTimeout", script)

    def test_script_gracefully_handles_storage_failure(self):
        """Script should catch localStorage errors for private browsing mode."""
        script = get_user_preferences_head()
        self.assertIn("catch", script)

    def test_script_generation_is_stable(self):
        """Function should be deterministic for repeated calls."""
        script_1 = get_user_preferences_head()
        script_2 = get_user_preferences_head()
        self.assertEqual(script_1, script_2)


if __name__ == "__main__":
    unittest.main()
