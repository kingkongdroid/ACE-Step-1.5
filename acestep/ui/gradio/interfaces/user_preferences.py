"""Frontend user-preference persistence helpers for the Gradio UI.

Save side:  A ``<script>`` injected via ``Blocks(head=…)`` listens for DOM
changes and writes the current preference values to ``localStorage``.

Restore side:  ``wire_preference_restore`` attaches a ``demo.load()`` handler
whose *js* parameter reads ``localStorage`` on page load and feeds the saved
values straight into the Gradio component outputs.  Because Gradio itself
applies the updates through its own Svelte reactivity, every component type
(dropdown, slider, checkbox, number) is updated correctly—no fragile DOM
hacking required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ASSET_FILENAME = "user_preferences.js"
_STORAGE_KEY = "acestep.ui.user_preferences"
_SCHEMA_VERSION = 1

# Ordered list of preference keys.  The order here MUST match the order of
# *outputs* passed to ``demo.load()`` in ``wire_preference_restore``.
PREF_KEYS: list[str] = [
    "audio_format",
    "mp3_bitrate",
    "mp3_sample_rate",
    "score_scale",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
    "latent_shift",
    "latent_rescale",
    "lm_batch_chunk_size",
]

# Default values used when localStorage is empty or the schema version has
# changed.  Keys must match ``PREF_KEYS``.
_DEFAULTS: dict[str, Any] = {
    "audio_format": "mp3",
    "mp3_bitrate": "128k",
    "mp3_sample_rate": 48000,
    "score_scale": 0.5,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "latent_shift": 0.0,
    "latent_rescale": 1.0,
    "lm_batch_chunk_size": 8,
}


# ── Save-side: head script injection ────────────────────────────────────


def _load_preferences_script() -> str:
    """Load the external save-preferences JavaScript asset."""
    asset_path = Path(__file__).with_name(_ASSET_FILENAME)
    return asset_path.read_text(encoding="utf-8").strip()


def get_user_preferences_head() -> str:
    """Return Gradio head HTML that injects save-side preference persistence."""
    script_source = _load_preferences_script()
    return f"<script>\n{script_source}\n</script>"


# ── Restore-side: Gradio .load() wiring ─────────────────────────────────


def _build_restore_js() -> str:
    """Build the client-side JS that reads localStorage and returns values.

    The returned function is passed as the ``js`` parameter to
    ``demo.load()``.  It returns an array whose element order matches
    ``PREF_KEYS`` (and therefore the *outputs* list).
    """
    defaults_json = json.dumps(
        {k: _DEFAULTS[k] for k in PREF_KEYS},
        ensure_ascii=False,
    )
    keys_json = json.dumps(PREF_KEYS)
    return f"""() => {{
        const STORAGE_KEY = {json.dumps(_STORAGE_KEY)};
        const SCHEMA_VERSION = {_SCHEMA_VERSION};
        const DEFAULTS = {defaults_json};
        const KEYS = {keys_json};
        try {{
            const raw = window.localStorage.getItem(STORAGE_KEY);
            if (!raw) return KEYS.map(k => DEFAULTS[k]);
            const prefs = JSON.parse(raw);
            if (prefs._version !== SCHEMA_VERSION) return KEYS.map(k => DEFAULTS[k]);
            return KEYS.map(k => (k in prefs) ? prefs[k] : DEFAULTS[k]);
        }} catch (_e) {{
            return KEYS.map(k => DEFAULTS[k]);
        }}
    }}"""


def restore_preferences(*values: Any) -> tuple[Any, ...]:
    """Identity pass-through – Gradio requires a Python *fn* even when
    the heavy lifting is done client-side by the *js* parameter.

    The JS function reads localStorage and produces an array of values.
    Gradio calls this Python function with those values, and we return
    them unchanged so they flow into the *outputs* components.
    """
    return tuple(values)


def wire_preference_restore(
    demo: Any,
    generation_section: dict[str, Any],
) -> None:
    """Attach a ``demo.load()`` handler that restores saved preferences.

    Must be called **inside** the ``with gr.Blocks() as demo:`` context,
    after all generation components have been created.

    Args:
        demo: The ``gr.Blocks`` instance.
        generation_section: Merged component dict that includes the output
            control components (``audio_format``, ``mp3_bitrate``, etc.).
    """
    outputs = []
    for key in PREF_KEYS:
        component = generation_section.get(key)
        if component is None:
            raise KeyError(
                f"wire_preference_restore: missing component {key!r} in "
                f"generation_section (available: {sorted(generation_section)})"
            )
        outputs.append(component)

    demo.load(
        fn=restore_preferences,
        inputs=None,
        outputs=outputs,
        js=_build_restore_js(),
    )
