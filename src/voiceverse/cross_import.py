# src/voiceverse/cross_import.py
from __future__ import annotations
import os
import sys
import types
import functools
import importlib
from pathlib import Path
import contextlib
from typing import Any, Callable, Iterable, Optional, Tuple

# Simple in-process cache to avoid re-importing the same file repeatedly.
_MODULE_CACHE: dict[Tuple[str, str], types.ModuleType] = {}

# Markers considered to detect a repo root.
_REPO_MARKERS = (".git", "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt", "README.md")

def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in (start, *start.parents):
        for m in _REPO_MARKERS:
            if (p / m).exists():
                return p
    # Fallback: if no markers found, treat the containing dir as the root
    return start

def _find_code_root(file_path: Path, repo_root: Path) -> Path:
    # Heuristic: if there's a "src" between repo_root and file, prefer that as the code root.
    # Otherwise, the repo_root works for common layouts and absolute imports within the repo.
    file_path = file_path.resolve()
    for ancestor in file_path.parents:
        if ancestor == repo_root:
            break
        if ancestor.name == "src":
            return ancestor
    return repo_root

def _module_name_from_path(file_path: Path, code_root: Path) -> str:
    try:
        rel = file_path.resolve().relative_to(code_root.resolve())
    except ValueError:
        raise ValueError(f"File path {file_path} is not within code_root {code_root}")
    # Convert path/to/module.py -> path.to.module
    return ".".join(rel.with_suffix("").parts)

@contextlib.contextmanager
def _temp_sys_path(paths: Iterable[Path]):
    original = list(sys.path)
    try:
        # Prepend in order so earlier items win
        for p in list(reversed([str(Path(p)) for p in paths])):
            if p not in sys.path:
                sys.path.insert(0, p)
        yield
    finally:
        sys.path[:] = original

@contextlib.contextmanager
def _temp_cwd(path: Path):
    prev = Path.cwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(str(prev))

def _collect_pth_paths(repo_root: Path) -> list[Path]:
    # Best-effort: add directories referenced by simple *.pth lines (ignoring "import" lines).
    paths: list[Path] = []
    try:
        for pth in repo_root.rglob("*.pth"):
            try:
                for line in pth.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("import"):
                        continue
                    candidate = Path(line)
                    if not candidate.is_absolute():
                        candidate = (pth.parent / candidate).resolve()
                    if candidate.exists() and candidate.is_dir():
                        paths.append(candidate)
            except Exception:
                # ignore malformed .pth files
                pass
    except Exception:
        pass
    return paths

def import_symbol_from_file(
    file_path: str | Path,
    symbol_name: str,
    *,
    add_pth_paths: bool = False,
    safe_globals: Optional[list] = None,
) -> Any:
    """
    Import a symbol (function/class/variable) by name from a .py file in another repository.

    - Detects the repository root (by common markers) and chooses a code root (repo root or nearest 'src' folder).
    - Temporarily prepends the code root (and optional .pth-derived paths) to sys.path to let intra-repo imports resolve.
    - Imports the module by its dotted name relative to the code root so that relative/absolute imports inside it work.
    - Returns the symbol; if callable, it’s wrapped so calls run with CWD at the repo root.
    - If safe_globals is provided, wraps calls with torch.serialization.safe_globals for safe deserialization.

    Args:
        file_path: Path to the .py file.
        symbol_name: Name of the symbol to import.
        add_pth_paths: Whether to add paths from .pth files.
        safe_globals: List of safe globals for torch.serialization.safe_globals (e.g., [builtins.getattr]).

    Example:
        load_ASR_models = import_symbol_from_file("/path/to/StyleTTS2/models.py", "load_ASR_models",
                                                  safe_globals=[builtins.getattr])
        model = load_ASR_models(...)
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    repo_root = _find_repo_root(file_path.parent)
    code_root = _find_code_root(file_path, repo_root)
    module_name = _module_name_from_path(file_path, code_root)

    extra_paths = _collect_pth_paths(repo_root) if add_pth_paths else []
    with _temp_sys_path([code_root, *extra_paths]):
        cache_key = (str(file_path), module_name)
        module = _MODULE_CACHE.get(cache_key)
        if module is None:
            try:
                # Import via dotted name so the module's relative imports work as intended.
                module = importlib.import_module(module_name)
            except Exception as e:
                raise ImportError(
                    f"Failed to import module '{module_name}' from '{file_path}' with code_root='{code_root}'"
                ) from e
            _MODULE_CACHE[cache_key] = module

        try:
            obj = getattr(module, symbol_name)
        except AttributeError as e:
            raise AttributeError(f"'{module_name}' has no attribute '{symbol_name}'") from e

    # If it’s callable, wrap it so calls run with sys.path and cwd contexts applied.
    if callable(obj):
        @functools.wraps(obj)
        def _wrapped(*args, **kwargs):
            with _temp_sys_path([code_root, *extra_paths]), _temp_cwd(repo_root):
                if safe_globals:
                    try:
                        import torch
                        from torch.serialization import safe_globals as _sg
                        with _sg(safe_globals):
                            return obj(*args, **kwargs)
                    except ImportError:
                        # If torch is not available, proceed without safe_globals
                        return obj(*args, **kwargs)
                return obj(*args, **kwargs)
        return _wrapped
        return _wrapped

    return obj
