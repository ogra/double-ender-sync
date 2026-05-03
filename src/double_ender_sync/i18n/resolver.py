from __future__ import annotations

import os

SUPPORTED_LANGUAGES = {"en", "ja"}
DEFAULT_LANGUAGE = "en"


def resolve_language(explicit_lang: str | None = None) -> str:
    """Resolve language with priority: explicit -> system locale -> default."""
    candidates: list[str | None] = [explicit_lang, os.environ.get("LC_ALL"), os.environ.get("LANG")]
    for candidate in candidates:
        normalized = _normalize_locale(candidate)
        if normalized in SUPPORTED_LANGUAGES:
            return normalized
    return DEFAULT_LANGUAGE


def extract_explicit_lang(argv: list[str]) -> str | None:
    """Extract `--lang` value from CLI-like args.

    Supports both `--lang xx` and `--lang=xx`.
    """
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def _normalize_locale(value: str | None) -> str | None:
    if not value:
        return None
    base = value.split(".", 1)[0].strip()
    if not base:
        return None
    language = base.split("_", 1)[0].split("-", 1)[0].lower()
    return language or None
