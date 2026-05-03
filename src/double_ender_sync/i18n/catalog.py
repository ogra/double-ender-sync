from __future__ import annotations

import json
import logging
import string
from importlib import resources

from double_ender_sync.i18n.resolver import DEFAULT_LANGUAGE

LOGGER = logging.getLogger(__name__)


class TranslationCatalog:
    def __init__(self, language: str) -> None:
        self.language = language
        self._messages = _load_messages(language)
        self._fallback = _load_messages(DEFAULT_LANGUAGE) if language != DEFAULT_LANGUAGE else self._messages

    def t(self, key: str, **kwargs: object) -> str:
        template = self._messages.get(key)
        if template is None:
            template = self._fallback.get(key)
            if template is not None:
                if self.language == DEFAULT_LANGUAGE:
                    LOGGER.warning("Missing translation key '%s' in default locale '%s'.", key, DEFAULT_LANGUAGE)
                else:
                    LOGGER.warning(
                        "Missing translation for key '%s' in locale '%s'. Falling back to '%s'.",
                        key,
                        self.language,
                        DEFAULT_LANGUAGE,
                    )
            else:
                if self.language == DEFAULT_LANGUAGE:
                    LOGGER.warning("Missing translation key '%s' in default locale '%s'.", key, DEFAULT_LANGUAGE)
                else:
                    LOGGER.warning(
                        "Missing translation key '%s' in locale '%s' and fallback locale '%s'.",
                        key,
                        self.language,
                        DEFAULT_LANGUAGE,
                    )
                template = key
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            return template


def extract_placeholders(template: str) -> set[str]:
    formatter = string.Formatter()
    placeholders: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:
            placeholders.add(field_name)
    return placeholders


def _load_messages(language: str) -> dict[str, str]:
    resource_name = f"{language}.json"
    package = "double_ender_sync.i18n.locales"
    if not resources.files(package).joinpath(resource_name).is_file():
        resource_name = f"{DEFAULT_LANGUAGE}.json"
    data = json.loads(resources.files(package).joinpath(resource_name).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}
