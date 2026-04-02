/**
 * User preferences persistence via browser localStorage.
 *
 * Saves and restores Gradio UI settings (output format, normalization, fades,
 * latent controls, etc.) so they survive page reloads and app restarts.
 *
 * Extends the existing audio_player_preferences.js pattern.
 */
(() => {
    const STORAGE_KEY = "acestep.ui.user_preferences";
    const DEBOUNCE_MS = 500;

    /**
     * Map of preference key -> { elemId, type }.
     *   elemId : the HTML elem_id set in Gradio
     *   type   : "dropdown" | "slider" | "checkbox" | "number"
     */
    const PREFS = {
        audio_format:        { elemId: "acestep-audio-format",        type: "dropdown" },
        mp3_bitrate:         { elemId: "acestep-mp3-bitrate",         type: "dropdown" },
        mp3_sample_rate:     { elemId: "acestep-mp3-sample-rate",     type: "dropdown" },
        score_scale:         { elemId: "acestep-score-scale",         type: "slider"   },
        enable_normalization:{ elemId: "acestep-enable-normalization", type: "checkbox" },
        normalization_db:    { elemId: "acestep-normalization-db",     type: "slider"   },
        fade_in_duration:    { elemId: "acestep-fade-in-duration",    type: "slider"   },
        fade_out_duration:   { elemId: "acestep-fade-out-duration",   type: "slider"   },
        latent_shift:        { elemId: "acestep-latent-shift",        type: "slider"   },
        latent_rescale:      { elemId: "acestep-latent-rescale",      type: "slider"   },
        lm_batch_chunk_size: { elemId: "acestep-lm-batch-chunk-size", type: "number"   },
    };

    let saveTimer = null;

    // ── Storage helpers ──────────────────────────────────────────────

    const loadAll = () => {
        try {
            const raw = window.localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : {};
        } catch (_e) {
            return {};
        }
    };

    const saveAll = (prefs) => {
        try {
            window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
        } catch (_e) {
            // Private browsing or quota exceeded – silently ignore.
        }
    };

    // ── DOM helpers ──────────────────────────────────────────────────

    /**
     * Find the actual input element inside a Gradio component wrapper.
     * Gradio wraps inputs in divs with the elem_id; the real control
     * is a child <input>, <select>, or <textarea>.
     */
    const findInput = (elemId, type) => {
        const wrapper = document.getElementById(elemId);
        if (!wrapper) return null;

        if (type === "dropdown") {
            // Gradio dropdown uses an <input> inside the wrapper.
            return wrapper.querySelector("input");
        }
        if (type === "slider") {
            // Range slider <input type="range"> or the number buddy <input type="number">.
            return wrapper.querySelector("input[type='range']")
                || wrapper.querySelector("input[type='number']");
        }
        if (type === "checkbox") {
            return wrapper.querySelector("input[type='checkbox']");
        }
        if (type === "number") {
            return wrapper.querySelector("input[type='number']");
        }
        return null;
    };

    /**
     * Read the current UI value for a preference key.
     */
    const readValue = (key) => {
        const spec = PREFS[key];
        if (!spec) return undefined;
        const el = findInput(spec.elemId, spec.type);
        if (!el) return undefined;

        if (spec.type === "checkbox") return el.checked;
        if (spec.type === "slider" || spec.type === "number") {
            const v = Number(el.value);
            return Number.isFinite(v) ? v : undefined;
        }
        // dropdown / text
        return el.value || undefined;
    };

    /**
     * Apply a saved value to a Gradio component by programmatically
     * setting the input and dispatching `input` + `change` events so
     * Gradio's Python backend picks up the new value.
     */
    const applyValue = (key, value) => {
        const spec = PREFS[key];
        if (!spec || value === undefined || value === null) return;
        const el = findInput(spec.elemId, spec.type);
        if (!el) return;

        if (spec.type === "checkbox") {
            if (el.checked === value) return;
            el.checked = value;
        } else {
            if (String(el.value) === String(value)) return;
            // For Gradio dropdowns we need to set nativeInputValueSetter
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, "value"
            ).set;
            nativeSetter.call(el, String(value));
        }
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
    };

    // ── Save (debounced) ─────────────────────────────────────────────

    const scheduleSave = () => {
        if (saveTimer !== null) {
            clearTimeout(saveTimer);
        }
        saveTimer = setTimeout(() => {
            saveTimer = null;
            const prefs = {};
            for (const key of Object.keys(PREFS)) {
                const v = readValue(key);
                if (v !== undefined) {
                    prefs[key] = v;
                }
            }
            if (Object.keys(prefs).length > 0) {
                saveAll(prefs);
            }
        }, DEBOUNCE_MS);
    };

    // ── Restore ──────────────────────────────────────────────────────

    const restoreAll = () => {
        const prefs = loadAll();
        if (!prefs || Object.keys(prefs).length === 0) return;

        for (const key of Object.keys(PREFS)) {
            if (key in prefs) {
                applyValue(key, prefs[key]);
            }
        }
    };

    // ── Wire up change listeners ─────────────────────────────────────

    const wireListeners = () => {
        for (const key of Object.keys(PREFS)) {
            const spec = PREFS[key];
            const el = findInput(spec.elemId, spec.type);
            if (!el) continue;
            el.addEventListener("input", scheduleSave, { passive: true });
            el.addEventListener("change", scheduleSave, { passive: true });
        }
    };

    // ── Boot ─────────────────────────────────────────────────────────

    const BOOT_POLL_MS = 200;
    const BOOT_TIMEOUT_MS = 10000;

    const boot = () => {
        // Gradio renders components async – poll until elements appear.
        const started = Date.now();
        const poll = () => {
            // Check if at least the audio_format dropdown exists.
            const probe = document.getElementById(PREFS.audio_format.elemId);
            if (!probe) {
                if (Date.now() - started < BOOT_TIMEOUT_MS) {
                    setTimeout(poll, BOOT_POLL_MS);
                }
                return;
            }
            // Elements are ready – restore saved prefs, then wire listeners.
            restoreAll();
            wireListeners();
        };
        poll();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        boot();
    }
})();
