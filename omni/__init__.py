"""Coding agent — drives Qwen Coder (or any OpenAI-compatible model)
through a scoped set of file/shell tools via MCP."""

# Single source of truth is pyproject.toml's [project] version — read it
# back from the installed distribution rather than restating it here, which
# is how this drifted to 0.1.0 while the package shipped 0.5.x.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("omni-coder")
except Exception:  # pragma: no cover - not installed (source checkout)
    __version__ = "0.0.0+unknown"
