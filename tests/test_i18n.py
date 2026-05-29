from double_ender_sync.i18n.catalog import TranslationCatalog
from double_ender_sync.i18n.resolver import SUPPORTED_LANGUAGES, extract_explicit_lang, resolve_language
from double_ender_sync.i18n.validate import validate_locales


def test_resolve_language_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert resolve_language(explicit_lang="en") == "en"


def test_resolve_language_uses_system_locale(monkeypatch) -> None:
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert resolve_language() == "ja"


def test_resolve_language_fallback_to_english(monkeypatch) -> None:
    monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert resolve_language() == "en"




def test_resolve_language_normalizes_region_code(monkeypatch) -> None:
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert resolve_language(explicit_lang="en-US") == "en"


def test_resolve_language_unsupported_explicit_falls_back_to_system(monkeypatch) -> None:
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert resolve_language(explicit_lang="fr-FR") == "ja"


def test_extract_explicit_lang_supports_equal_and_space_forms() -> None:
    assert extract_explicit_lang(["--lang", "ja"]) == "ja"
    assert extract_explicit_lang(["--foo", "1", "--lang=en-US"]) == "en-US"
    assert extract_explicit_lang(["--foo", "1"]) is None
def test_catalog_falls_back_per_key() -> None:
    catalog = TranslationCatalog("ja")
    assert catalog.t("gui.run_alignment") == "アライメント実行"
    assert catalog.t("missing.key") == "missing.key"


def test_catalog_logs_warning_when_missing_everywhere(caplog) -> None:
    catalog = TranslationCatalog("ja")
    with caplog.at_level("WARNING"):
        result = catalog.t("unknown.key")
    assert result == "unknown.key"
    assert "Missing translation key 'unknown.key'" in caplog.text


def test_catalog_default_locale_warning_does_not_reference_fallback(caplog) -> None:
    catalog = TranslationCatalog("en")
    with caplog.at_level("WARNING"):
        result = catalog.t("unknown.key")
    assert result == "unknown.key"
    assert "default locale 'en'" in caplog.text
    assert "Falling back" not in caplog.text


def test_catalog_formats_params() -> None:
    catalog = TranslationCatalog("en")
    assert catalog.t("gui.error.failed", exit_code=2) == "Alignment failed with exit code 2."


def test_catalog_returns_template_when_formatting_fails() -> None:
    catalog = TranslationCatalog("en")
    catalog._messages["broken"] = "Missing key {value}"
    assert catalog.t("broken", wrong=1) == "Missing key {value}"


def test_validate_locales_passes_for_bundled_locales() -> None:
    result = validate_locales(sorted(SUPPORTED_LANGUAGES))
    assert result.ok


def test_validate_locales_uses_supported_languages_by_default() -> None:
    result = validate_locales()
    assert result.ok


def test_validate_locales_reports_load_error(monkeypatch) -> None:
    def _boom(language: str):
        if language == "ja":
            raise FileNotFoundError("missing")
        return {"gui.ok": "ok"}

    monkeypatch.setattr("double_ender_sync.i18n.validate._load_locale", _boom)
    result = validate_locales(["en", "ja"])
    assert not result.ok
    assert any("ja: failed to load locale file" in err for err in result.errors)


NEW_CLI_ERROR_KEYS = [
    "cli.error.analysis_sample_rate_positive",
    "cli.error.stretch_threshold_non_negative",
    "cli.error.invalid_anchor_selection",
    "cli.error.invalid_drift_model",
    "cli.error.invalid_anchor_matching",
    "cli.error.pyannote_model_invalid",
    "cli.error.audio_load_failed",
    "cli.error.stretch_no_tty",
    "cli.error.aborted_by_user",
    "cli.error.gui_dependencies_missing",
]

NEW_CLI_WARN_KEYS = [
    "cli.warn.stretch_ratio",
]

NEW_CLI_PROMPT_KEYS = [
    "cli.prompt.continue_alignment",
]

NEW_CLI_OUTPUT_KEYS = [
    "cli.output.wrote",
    "cli.output.reports_generated",
]

NEW_CLI_VALIDATE_KEYS = [
    "cli.validate.passed",
    "cli.validate.failed",
    "cli.validate.failed_with_error",
]

NEW_WARNINGS_KEYS = [
    "warnings.drift_model_diagnostics_fallback",
    "warnings.drift_fit_diagnostics_fallback",
    "warnings.anchor_coverage_fallback",
]

NEW_GUI_KEYS = [
    "gui.progress_done_percent",
]


def test_new_i18n_messages_exist_in_en() -> None:
    catalog = TranslationCatalog("en")
    all_keys = (
        NEW_CLI_ERROR_KEYS
        + NEW_CLI_WARN_KEYS
        + NEW_CLI_PROMPT_KEYS
        + NEW_CLI_OUTPUT_KEYS
        + NEW_CLI_VALIDATE_KEYS
        + NEW_WARNINGS_KEYS
        + NEW_GUI_KEYS
    )
    for key in all_keys:
        result = catalog.t(key)
        assert result != key, f"Key '{key}' should have a translation in en.json, got raw key"


def test_new_i18n_messages_exist_in_ja() -> None:
    catalog = TranslationCatalog("ja")
    all_keys = (
        NEW_CLI_ERROR_KEYS
        + NEW_CLI_WARN_KEYS
        + NEW_CLI_PROMPT_KEYS
        + NEW_CLI_OUTPUT_KEYS
        + NEW_CLI_VALIDATE_KEYS
        + NEW_WARNINGS_KEYS
        + NEW_GUI_KEYS
    )
    for key in all_keys:
        result = catalog.t(key)
        assert result != key, f"Key '{key}' should have a translation in ja.json, got raw key"


def test_cli_error_messages_format_params() -> None:
    catalog = TranslationCatalog("en")
    assert "error: invalid anchor" in catalog.t("cli.error.invalid_anchor_selection", exc="test error")
    assert "error: test error" == catalog.t("cli.error.audio_load_failed", exc="test error")


def test_cli_warn_stretch_ratio_format() -> None:
    catalog = TranslationCatalog("en")
    result = catalog.t(
        "cli.warn.stretch_ratio",
        name="speaker-a",
        stretch_ratio="1.000000",
        delta="0.000000",
        threshold="0.003000",
    )
    assert "speaker-a" in result
    assert "1.000000" in result


def test_cli_output_wrote_format() -> None:
    catalog = TranslationCatalog("en")
    result = catalog.t("cli.output.wrote", path="/tmp/test.wav")
    assert "/tmp/test.wav" in result


def test_cli_validate_failed_with_error_format() -> None:
    catalog = TranslationCatalog("en")
    result = catalog.t("cli.validate.failed_with_error", exc="test")
    assert "test" in result


def test_warnings_fallback_messages_not_raw_keys() -> None:
    catalog = TranslationCatalog("en")
    assert catalog.t("warnings.drift_model_diagnostics_fallback") != "warnings.drift_model_diagnostics_fallback"
    assert catalog.t("warnings.drift_fit_diagnostics_fallback") != "warnings.drift_fit_diagnostics_fallback"
    assert catalog.t("warnings.anchor_coverage_fallback") != "warnings.anchor_coverage_fallback"


def test_gui_progress_done_percent_equals_value() -> None:
    catalog = TranslationCatalog("en")
    assert catalog.t("gui.progress_done_percent") == "100.0%"
