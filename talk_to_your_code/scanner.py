from __future__ import annotations

import hashlib
from pathlib import Path

import pathspec

from .config import Settings
from .languages import detect_language
from .schemas import CodeFile

DEFAULT_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "target", "out", ".next", ".nuxt", "coverage",
}


class RepoScanner:
    """Scans a repo and returns a list of CodeFile objects for supported files.
    
    It respects .gitignore if present, and skips common binary directories.
    It also enforces a max file size limit to avoid reading huge files into memory.

    Attributes:
        settings: Runtime settings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def scan(self, repo_path: str | Path) -> list[CodeFile]:
        """Scans the given repository path and returns a list of CodeFile objects for supported files.
        
        Args:
            repo_path: The file system path to the repository to scan.

        Returns:
            A list of CodeFile objects representing the supported files found in the repository.
        """
        root = Path(repo_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Repo path does not exist or is not a directory: {root}")

        gitignore = self._load_gitignore(root)
        files: list[CodeFile] = []

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if self._is_skipped_path(root, path, gitignore):
                continue

            language = detect_language(path)
            if language is None:
                continue
            if path.stat().st_size > self.settings.max_file_bytes:
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError:
                continue

            rel = path.relative_to(root).as_posix()
            encoded = content.encode("utf-8", errors="ignore")
            files.append(
                CodeFile(
                    path=rel,
                    language=language,
                    content=content,
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    size_bytes=len(encoded),
                    modified_time=path.stat().st_mtime,
                )
            )

        return files

    def _load_gitignore(self, root: Path) -> pathspec.PathSpec | None:
        gitignore_path = root / ".gitignore"
        if not gitignore_path.exists():
            return None
        try:
            patterns = gitignore_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def _is_skipped_path(self, root: Path, path: Path, gitignore: pathspec.PathSpec | None) -> bool:
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if parts & DEFAULT_SKIP_DIRS:
            return True
        if gitignore is not None and gitignore.match_file(rel):
            return True
        return False
