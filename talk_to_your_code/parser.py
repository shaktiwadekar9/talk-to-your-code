from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass

from .schemas import CodeChunk, CodeFile, Symbol


@dataclass(frozen=True)
class _RawSymbol:
    name: str
    kind: str
    start_line: int
    end_line: int


class MultiLanguageParser:
    """Extracts symbols and chunks for Python, JS, TS, Java, C#, and Go.

    Python uses the standard `ast` module. Other languages use regex plus brace
    matching. This is intentionally minimal and easy to replace later with
    Tree-sitter without changing the rest of the architecture.
    """

    def symbols_for_file(self, code_file: CodeFile) -> list[Symbol]:
        return [
            Symbol(
                name=s.name,
                kind=s.kind,
                file_path=code_file.path,
                start_line=s.start_line,
                end_line=s.end_line,
                language=code_file.language,
            )
            for s in self._parse_raw_symbols(code_file)
        ]

    def chunks_for_file(self, code_file: CodeFile) -> list[CodeChunk]:
        """The file is split into chunks based on the symbols. 
        Each chunk corresponds to a symbol's full text (including decorators/modifiers). 
        If no symbols are found, the file is split into fixed-size line windows.

        Args:
            code_file: The CodeFile object containing the file's path, language, and content.
        
        Returns:
            A list of CodeChunk objects representing the extracted code chunks from the file.
        """
        symbols = self._parse_raw_symbols(code_file)
        lines = code_file.content.splitlines()
        chunks: list[CodeChunk] = []
        seen: set[tuple[int, int, str]] = set()

        for symbol in symbols:
            start = max(1, symbol.start_line)
            end = min(len(lines), max(symbol.end_line, start))
            key = (start, end, symbol.name)
            if key in seen:
                continue
            seen.add(key)
            text = "\n".join(lines[start - 1 : end])
            chunks.append(
                CodeChunk(
                    chunk_id=self._chunk_id(code_file.path, start, end, symbol.name),
                    file_path=code_file.path,
                    language=code_file.language,
                    start_line=start,
                    end_line=end,
                    text=text,
                    symbol_name=symbol.name,
                    symbol_kind=symbol.kind,
                )
            )

        # Text/config files and symbol-poor files still need to be searchable.
        if not chunks or code_file.language == "text":
            chunks.extend(self._line_window_chunks(code_file))

        return self._dedupe_chunks(chunks)

    def _parse_raw_symbols(self, code_file: CodeFile) -> list[_RawSymbol]:
        if code_file.language == "python":
            return self._parse_python(code_file.content)
        if code_file.language in {"javascript", "typescript"}:
            return self._parse_js_ts(code_file.content, code_file.language)
        if code_file.language == "java":
            return self._parse_java(code_file.content)
        if code_file.language == "csharp":
            return self._parse_csharp(code_file.content)
        if code_file.language == "go":
            return self._parse_go(code_file.content)
        return []

    def _parse_python(self, content: str) -> list[_RawSymbol]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        symbols: list[_RawSymbol] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(_RawSymbol(node.name, "class", node.lineno, getattr(node, "end_lineno", node.lineno)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                symbols.append(_RawSymbol(node.name, kind, node.lineno, getattr(node, "end_lineno", node.lineno)))
        return self._dedupe_symbols(sorted(symbols, key=lambda s: (s.start_line, s.end_line, s.name)))

    def _parse_js_ts(self, content: str, language: str) -> list[_RawSymbol]:
        patterns = [
            ("class", r"\b(?:export\s+default\s+|export\s+)?class\s+([A-Za-z_$][\w$]*)"),
            ("function", r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("),
            ("arrow_function", r"\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
        ]
        if language == "typescript":
            patterns += [
                ("interface", r"\b(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"),
                ("type", r"\b(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="),
                ("enum", r"\b(?:export\s+)?enum\s+([A-Za-z_$][\w$]*)"),
            ]
        return self._symbols_by_patterns(content, patterns)

    def _parse_java(self, content: str) -> list[_RawSymbol]:
        patterns = [
            ("class", r"\b(?:public|private|protected|abstract|final|static|\s)*class\s+([A-Za-z_][\w]*)"),
            ("interface", r"\b(?:public|private|protected|\s)*interface\s+([A-Za-z_][\w]*)"),
            ("enum", r"\b(?:public|private|protected|\s)*enum\s+([A-Za-z_][\w]*)"),
            ("method", r"\b(?:public|private|protected|static|final|abstract|synchronized|native|strictfp|\s)+[<>,\w\[\]?\s]+\s+([A-Za-z_][\w]*)\s*\([^;{}]*\)\s*(?:throws\s+[\w.,\s]+)?\{"),
        ]
        return self._symbols_by_patterns(content, patterns)

    def _parse_csharp(self, content: str) -> list[_RawSymbol]:
        patterns = [
            ("class", r"\b(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*class\s+([A-Za-z_][\w]*)"),
            ("interface", r"\b(?:public|private|protected|internal|partial|\s)*interface\s+([A-Za-z_][\w]*)"),
            ("enum", r"\b(?:public|private|protected|internal|\s)*enum\s+([A-Za-z_][\w]*)"),
            ("struct", r"\b(?:public|private|protected|internal|readonly|partial|\s)*struct\s+([A-Za-z_][\w]*)"),
            ("method", r"\b(?:public|private|protected|internal|static|virtual|override|async|sealed|partial|extern|\s)+[<>,\w\[\]?\s]+\s+([A-Za-z_][\w]*)\s*\([^;{}]*\)\s*\{"),
        ]
        return self._symbols_by_patterns(content, patterns)

    def _parse_go(self, content: str) -> list[_RawSymbol]:
        patterns = [
            ("function", r"\bfunc\s+([A-Za-z_][\w]*)\s*\("),
            ("method", r"\bfunc\s*\([^)]*\)\s*([A-Za-z_][\w]*)\s*\("),
            ("struct", r"\btype\s+([A-Za-z_][\w]*)\s+struct\s*\{"),
            ("interface", r"\btype\s+([A-Za-z_][\w]*)\s+interface\s*\{"),
        ]
        return self._symbols_by_patterns(content, patterns)

    def _symbols_by_patterns(self, content: str, patterns: list[tuple[str, str]]) -> list[_RawSymbol]:
        symbols: list[_RawSymbol] = []
        for kind, pattern in patterns:
            for match in re.finditer(pattern, content, flags=re.MULTILINE):
                name = match.group(1)
                start_line = content.count("\n", 0, match.start()) + 1
                end_line = start_line

                current_line_end = content.find("\n", match.end())
                if current_line_end == -1:
                    current_line_end = len(content)
                current_line_tail = content[match.end() : current_line_end]

                if kind == "type" and "{" not in current_line_tail:
                    end_line = start_line
                else:
                    brace_pos = content.find("{", match.end() - 1)
                    if brace_pos != -1 and brace_pos - match.end() <= 500:
                        close_pos = self._find_matching_brace(content, brace_pos)
                        if close_pos != -1:
                            end_line = content.count("\n", 0, close_pos) + 1

                symbols.append(_RawSymbol(name, kind, start_line, max(start_line, end_line)))
        return self._dedupe_symbols(sorted(symbols, key=lambda s: (s.start_line, s.end_line, s.name)))

    def _line_window_chunks(self, code_file: CodeFile, window: int = 90, overlap: int = 12) -> list[CodeChunk]:
        """Splits the file into overlapping line windows when no symbols are found.
        
        Args:
            code_file: The CodeFile object containing the file's path, language, and content.
            window: The number of lines in each chunk window.
            overlap: The number of overlapping lines between consecutive windows to preserve context.
            
        Returns:
            A list of CodeChunk objects representing the line window chunks from the file.
        """
        lines = code_file.content.splitlines()
        if not lines:
            return []
        chunks: list[CodeChunk] = []
        start = 1
        while start <= len(lines):
            end = min(len(lines), start + window - 1)
            text = "\n".join(lines[start - 1 : end])
            chunks.append(
                CodeChunk(
                    chunk_id=self._chunk_id(code_file.path, start, end, "window"),
                    file_path=code_file.path,
                    language=code_file.language,
                    start_line=start,
                    end_line=end,
                    text=text,
                    symbol_name=None,
                    symbol_kind="window",
                )
            )
            if end == len(lines):
                break
            start = max(end - overlap + 1, start + 1)
        return chunks

    @staticmethod
    def _find_matching_brace(content: str, open_pos: int) -> int:
        depth = 0
        in_single = in_double = in_backtick = False
        escaped = False
        in_line_comment = in_block_comment = False
        i = open_pos
        while i < len(content):
            ch = content[i]
            nxt = content[i + 1] if i + 1 < len(content) else ""

            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue
            if not (in_single or in_double or in_backtick):
                if ch == "/" and nxt == "/":
                    in_line_comment = True
                    i += 2
                    continue
                if ch == "/" and nxt == "*":
                    in_block_comment = True
                    i += 2
                    continue

            if escaped:
                escaped = False
            elif ch == "\\" and (in_single or in_double or in_backtick):
                escaped = True
            elif ch == "'" and not (in_double or in_backtick):
                in_single = not in_single
            elif ch == '"' and not (in_single or in_backtick):
                in_double = not in_double
            elif ch == "`" and not (in_single or in_double):
                in_backtick = not in_backtick
            elif not (in_single or in_double or in_backtick):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    @staticmethod
    def _chunk_id(file_path: str, start: int, end: int, name: str) -> str:
        raw = f"{file_path}:{start}:{end}:{name}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:20]

    @staticmethod
    def _dedupe_symbols(symbols: list[_RawSymbol]) -> list[_RawSymbol]:
        seen: set[tuple[str, str, int, int]] = set()
        out: list[_RawSymbol] = []
        for symbol in symbols:
            key = (symbol.name, symbol.kind, symbol.start_line, symbol.end_line)
            if key not in seen:
                seen.add(key)
                out.append(symbol)
        return out

    @staticmethod
    def _dedupe_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
        seen: set[str] = set()
        out: list[CodeChunk] = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                out.append(chunk)
        return out
