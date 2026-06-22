from __future__ import annotations

from pathlib import Path
from typing import Literal

SupportedLanguage = Literal["python", "javascript", "typescript", "java", "csharp", "go", "text"]

EXTENSION_TO_LANGUAGE: dict[str, SupportedLanguage] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
}

TEXT_EXTENSIONS = {
    ".md", ".mdx", ".txt", ".rst", ".toml", ".yaml", ".yml",
    ".json", ".ini", ".cfg", ".properties", ".env.example",
}

SPECIAL_TEXT_FILES = {"Dockerfile", "Makefile", "README", "LICENSE"}


def detect_language(path: str | Path) -> SupportedLanguage | None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[suffix]
    if suffix in TEXT_EXTENSIONS or p.name in SPECIAL_TEXT_FILES:
        return "text"
    return None


def is_supported_source(path: str | Path) -> bool:
    return detect_language(path) is not None
