from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Settings
from .schemas import CodeChunk, CodeFile, RepoRecord, Symbol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteStore:
    """SQLite persistence layer. 
    Stores repo metadata, file contents, code chunks, symbols, and embeddings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize the database schema if it doesn't already exist.
        """
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS repos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    symbol_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    UNIQUE(repo_id, path),
                    FOREIGN KEY(repo_id) REFERENCES repos(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    symbol_name TEXT,
                    symbol_kind TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    FOREIGN KEY(repo_id) REFERENCES repos(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    FOREIGN KEY(repo_id) REFERENCES repos(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_files_repo_path ON files(repo_id, path);
                CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks(repo_id, file_path);
                CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols(repo_id, name);
                """
            )

    def get_or_create_repo(self, repo_path: str | Path) -> RepoRecord:
        """Find existing repo by root path or create a new one if it doesn't exist.
        
        Args:
            repo_path: Filesystem path to the root of the code repository.

        Returns:
            RepoRecord for the existing or newly created repository.
        """
        root = Path(repo_path).expanduser().resolve()
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM repos WHERE root_path = ?", (str(root),)).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO repos(name, root_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (root.name, str(root), now, now),
                )
                repo_id = int(cur.lastrowid)
                row = conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
            return self._repo_from_row(row)

    def list_repos(self) -> list[RepoRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM repos ORDER BY updated_at DESC").fetchall()
        return [self._repo_from_row(row) for row in rows]

    def get_repo(self, repo_id: int) -> RepoRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
        return self._repo_from_row(row) if row else None

    def clear_repo_data(self, repo_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))
            conn.execute(
                "UPDATE repos SET file_count = 0, chunk_count = 0, symbol_count = 0, updated_at = ? WHERE id = ?",
                (utc_now(), repo_id),
            )

    def save_index(
        self,
        repo_id: int,
        files: list[CodeFile],
        chunks: list[CodeChunk],
        symbols: list[Symbol],
        embeddings: np.ndarray,
    ) -> None:
        """Save the results of scanning and indexing a repository, including file contents, code chunks, symbols, and their embeddings.
        
        Args:
            repo_id: ID of the repository to update.
            files: List of CodeFile objects representing the scanned files in the repository.
            chunks: List of CodeChunk objects representing the code chunks extracted from the files.
            symbols: List of Symbol objects representing the code symbols (functions, classes, etc.) extracted from the files.
            embeddings: 2D numpy array where each row corresponds to the embedding of a code chunk. The order must match the `chunks` list.  
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        with self.connect() as conn:
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))

            conn.executemany(
                """
                INSERT INTO files(repo_id, path, language, content, sha256, size_bytes, modified_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (repo_id, f.path, f.language, f.content, f.sha256, f.size_bytes, f.modified_time)
                    for f in files
                ],
            )

            conn.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, repo_id, file_path, language, start_line, end_line, text,
                    symbol_name, symbol_kind, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        repo_id,
                        c.file_path,
                        c.language,
                        c.start_line,
                        c.end_line,
                        c.text,
                        c.symbol_name,
                        c.symbol_kind,
                        np.asarray(emb, dtype=np.float32).tobytes(),
                        int(len(emb)),
                    )
                    for c, emb in zip(chunks, embeddings)
                ],
            )

            conn.executemany(
                """
                INSERT INTO symbols(repo_id, name, kind, file_path, start_line, end_line, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (repo_id, s.name, s.kind, s.file_path, s.start_line, s.end_line, s.language)
                    for s in symbols
                ],
            )

            conn.execute(
                """
                UPDATE repos
                SET file_count = ?, chunk_count = ?, symbol_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (len(files), len(chunks), len(symbols), utc_now(), repo_id),
            )

    def load_chunks_and_embeddings(self, repo_id: int) -> tuple[list[CodeChunk], np.ndarray]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, file_path, language, start_line, end_line, text,
                       symbol_name, symbol_kind, embedding, embedding_dim
                FROM chunks WHERE repo_id = ? ORDER BY file_path, start_line
                """,
                (repo_id,),
            ).fetchall()

        chunks: list[CodeChunk] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            chunks.append(
                CodeChunk(
                    chunk_id=row["chunk_id"],
                    file_path=row["file_path"],
                    language=row["language"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    text=row["text"],
                    symbol_name=row["symbol_name"],
                    symbol_kind=row["symbol_kind"],
                )
            )
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"]).copy())

        if vectors:
            embeddings = np.vstack(vectors).astype(np.float32)
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        return chunks, embeddings

    def load_symbols(self, repo_id: int) -> list[Symbol]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT name, kind, file_path, start_line, end_line, language FROM symbols WHERE repo_id = ?",
                (repo_id,),
            ).fetchall()
        return [
            Symbol(
                name=row["name"],
                kind=row["kind"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
            )
            for row in rows
        ]

    def load_repo_file_paths(self, repo_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT path FROM files WHERE repo_id = ? ORDER BY path", (repo_id,)).fetchall()
        return [row["path"] for row in rows]

    def _repo_from_row(self, row: sqlite3.Row) -> RepoRecord:
        return RepoRecord(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            file_count=row["file_count"],
            chunk_count=row["chunk_count"],
            symbol_count=row["symbol_count"],
        )
