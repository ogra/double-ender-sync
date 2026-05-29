from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import resources

from double_ender_sync.i18n.catalog import TranslationCatalog, extract_placeholders
from double_ender_sync.i18n.resolver import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, resolve_language

REQUIRED_PREFIXES = ("gui.", "cli.", "api.", "errors.", "warnings.")
PACKAGE = "double_ender_sync.i18n.locales"


@dataclass
class LocaleValidationResult:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_locale(language: str) -> dict[str, str]:
    path = resources.files(PACKAGE).joinpath(f"{language}.json")
    if not path.is_file():
        raise FileNotFoundError(f"Locale file not found: {language}.json")
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def validate_locales(languages: list[str] | None = None) -> LocaleValidationResult:
    if not languages:
        languages = sorted(SUPPORTED_LANGUAGES)
    errors: list[str] = []

    try:
        base = _load_locale(DEFAULT_LANGUAGE)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return LocaleValidationResult(errors=[f"{DEFAULT_LANGUAGE}: failed to load locale file ({exc})"])

    for key in base:
        if not key.startswith(REQUIRED_PREFIXES):
            errors.append(f"{DEFAULT_LANGUAGE}: key '{key}' does not use a required prefix {REQUIRED_PREFIXES}")

    for language in languages:
        try:
            current = _load_locale(language)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"{language}: failed to load locale file ({exc})")
            continue

        for key in current:
            if not key.startswith(REQUIRED_PREFIXES):
                errors.append(f"{language}: key '{key}' does not use a required prefix {REQUIRED_PREFIXES}")

        missing = sorted(set(base) - set(current))
        for key in missing:
            errors.append(f"{language}: missing required key '{key}'")

        for key in sorted(set(base) & set(current)):
            expected = extract_placeholders(base[key])
            actual = extract_placeholders(current[key])
            if actual != expected:
                errors.append(
                    f"{language}: placeholder mismatch for key '{key}' expected {sorted(expected)} got {sorted(actual)}"
                )

    return LocaleValidationResult(errors=errors)


def main() -> int:
    def t(key: str, **kwargs: object) -> str:
        if key == "cli.validate.failed_with_error":
            exc = kwargs.get("exc")
            return f"Validation failed with error: {exc}"
        if key == "cli.validate.passed":
            return "Validation passed"
        if key == "cli.validate.failed":
            return "Validation failed"
        return key

    try:
        catalog = TranslationCatalog(resolve_language())
        t = catalog.t
        result = validate_locales()
    except Exception as exc:  # defensive guard to keep CLI reporting stable
        print(t("cli.validate.failed_with_error", exc=exc), file=sys.stderr)
        return 1

    if result.ok:
        print(t("cli.validate.passed"))
        return 0

    print(t("cli.validate.failed"), file=sys.stderr)
    for error in result.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
