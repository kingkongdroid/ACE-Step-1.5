"""Unit tests for user preference persistence."""

import importlib.util
import json
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
_build_restore_js = _MODULE._build_restore_js
restore_preferences = _MODULE.restore_preferences
PREF_KEYS = _MODULE.PREF_KEYS
_DEFAULTS = _MODULE._DEFAULTS
_SCRIPT_PATH = Path(__file__).with_name("user_preferences.js")


class SaveScriptTests(unittest.TestCase):
    """Tests for the save-side JavaScript injected via Gradio head."""

    def test_external_script_asset_exists(self):
        self.assertTrue(_SCRIPT_PATH.is_file())
        script_asset = _load_preferences_script()
        self.assertTrue(script_asset)

    def test_script_contains_localstorage_persistence(self):
        script = get_user_preferences_head()
        self.assertIn("<script>", script)
        self.assertIn("localStorage", script)
        self.assertIn("acestep.ui.user_preferences", script)

    def test_script_contains_all_preference_elem_ids(self):
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

    def test_script_is_save_only_no_restore_logic(self):
        """The JS should only save; restore is handled by Gradio .load()."""
        script = get_user_preferences_head()
        self.assertNotIn("restoreAll", script)
        self.assertNotIn("applyValue", script)
        self.assertNotIn("nativeInputValueSetter", script)

    def test_script_includes_schema_version(self):
        script = get_user_preferences_head()
        self.assertIn("SCHEMA_VERSION", script)
        self.assertIn("_version", script)

    def test_script_debounces_saves(self):
        script = get_user_preferences_head()
        self.assertIn("DEBOUNCE_MS", script)
        self.assertIn("clearTimeout", script)

    def test_script_uses_mutation_observer(self):
        """MutationObserver ensures listeners survive Gradio re-renders."""
        script = get_user_preferences_head()
        self.assertIn("MutationObserver", script)
        self.assertIn("wiredElements", script)

    def test_script_gracefully_handles_storage_failure(self):
        script = get_user_preferences_head()
        self.assertIn("catch", script)

    def test_script_generation_is_stable(self):
        script_1 = get_user_preferences_head()
        script_2 = get_user_preferences_head()
        self.assertEqual(script_1, script_2)


class RestoreTests(unittest.TestCase):
    """Tests for the Gradio-native restore mechanism."""

    def test_restore_js_returns_valid_javascript(self):
        js = _build_restore_js()
        self.assertIn("localStorage", js)
        self.assertIn("SCHEMA_VERSION", js)
        self.assertIn("acestep.ui.user_preferences", js)

    def test_restore_js_includes_all_pref_keys(self):
        js = _build_restore_js()
        for key in PREF_KEYS:
            self.assertIn(f'"{key}"', js, f"Missing key in restore JS: {key}")

    def test_restore_js_includes_defaults_for_all_keys(self):
        js = _build_restore_js()
        for key in PREF_KEYS:
            default = _DEFAULTS[key]
            self.assertIn(json.dumps(default), js,
                          f"Missing default for {key}={default!r}")

    def test_restore_js_rejects_wrong_schema_version(self):
        js = _build_restore_js()
        self.assertIn("_version", js)
        self.assertIn("DEFAULTS", js)

    def test_restore_preferences_is_identity(self):
        """The Python fn is a pass-through; values come from the JS side."""
        values = ("flac", "320k", 44100, 0.8, False, -3.0, 0.5, 1.0, 0.05, 0.95, 4)
        result = restore_preferences(*values)
        self.assertEqual(result, values)

    def test_pref_keys_match_defaults(self):
        """Every PREF_KEY must have a corresponding default."""
        for key in PREF_KEYS:
            self.assertIn(key, _DEFAULTS, f"Key {key!r} missing from _DEFAULTS")

    def test_restore_js_generation_is_stable(self):
        js_1 = _build_restore_js()
        js_2 = _build_restore_js()
        self.assertEqual(js_1, js_2)


if __name__ == "__main__":
    unittest.main()
