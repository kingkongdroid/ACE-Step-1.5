"""Frontend user-preference persistence helpers for the Gradio UI."""

from pathlib import Path


_ASSET_FILENAME = "user_preferences.js"


def _load_preferences_script() -> str:
    """Load the external user-preferences JavaScript asset.

    Returns:
        JavaScript source text used by the Gradio ``head`` injection.
    """
    asset_path = Path(__file__).with_name(_ASSET_FILENAME)
    return asset_path.read_text(encoding="utf-8").strip()


def get_user_preferences_head() -> str:
    """Return Gradio head HTML that injects user preference persistence.

    Returns:
        HTML snippet with a single ``<script>`` tag containing the externalized
        user preference JavaScript payload.
    """
    script_source = _load_preferences_script()
    return f"<script>\n{script_source}\n</script>"
