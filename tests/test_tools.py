"""Tools: path scoping (the security boundary), file IO, search, git, shell."""

import os

import pytest

from omni.config import AgentConfig
from omni.tools import PathScopeError, Tools, _resolve_in_scope, _truncate


# ---------------- path scoping ----------------

def test_resolve_in_scope_allows_root_and_children(tmp_path):
    (tmp_path / "sub").mkdir()
    assert _resolve_in_scope(str(tmp_path), ".") == os.path.realpath(tmp_path)
    assert _resolve_in_scope(str(tmp_path), "sub") == os.path.realpath(tmp_path / "sub")
    assert _resolve_in_scope(str(tmp_path), "a/b/c.txt").startswith(os.path.realpath(tmp_path) + os.sep)


@pytest.mark.parametrize("escape", [
    "..",
    "../outside.txt",
    "sub/../../outside.txt",
    "/etc/passwd",
    "/tmp",
])
def test_resolve_in_scope_rejects_escapes(tmp_path, escape):
    root = tmp_path / "proj"
    root.mkdir()
    with pytest.raises(PathScopeError):
        _resolve_in_scope(str(root), escape)


def test_resolve_in_scope_rejects_symlink_out(tmp_path):
    """A symlink inside the root pointing outside must not be a way through —
    realpath() resolves it before the prefix check."""
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("classified")
    (root / "link").symlink_to(outside)
    with pytest.raises(PathScopeError):
        _resolve_in_scope(str(root), "link")


def test_resolve_in_scope_rejects_sibling_with_shared_prefix(tmp_path):
    """`/x/proj-evil` must not pass the `/x/proj` prefix check."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj-evil").mkdir()
    with pytest.raises(PathScopeError):
        _resolve_in_scope(str(tmp_path / "proj"), str(tmp_path / "proj-evil"))


def test_truncate():
    assert _truncate("abc", 10) == "abc"
    out = _truncate("x" * 20, 10)
    assert out.startswith("x" * 10) and "10 more chars" in out


def test_every_path_taking_tool_is_scoped(tools):
    """Each tool that accepts a path must refuse to escape the root, rather
    than raising an unhandled error or silently reading outside."""
    for call in (
        lambda: tools.read_file("../../etc/passwd"),
        lambda: tools.list_dir("/etc"),
        lambda: tools.write_file("../evil.txt", "x"),
        lambda: tools.edit_file("/etc/hosts", "a", "b"),
        lambda: tools.search_files("x", path="/etc"),
        lambda: tools.glob_files("*", path="/etc"),
        lambda: tools.preview_edit("../x", "a", "b"),
        lambda: tools.preview_write("../x", "c"),
    ):
        with pytest.raises(PathScopeError):
            call()


def test_file_exists_returns_false_for_out_of_scope(tools):
    """file_exists swallows the scope error by design (it validates parsed
    intent) — but must answer False, never True, for an outside path."""
    assert tools.file_exists("/etc/passwd") is False
    assert tools.file_exists("hello.py") is True
    assert tools.file_exists("nope.py") is False


# ---------------- read_file / list_dir ----------------

def test_read_file_whole_and_ranges(tools):
    assert tools.read_file("notes.txt") == "alpha\nbeta\ngamma\n"
    assert tools.read_file("notes.txt", start_line=2) == "beta\ngamma\n"
    assert tools.read_file("notes.txt", start_line=1, end_line=2) == "alpha\nbeta\n"
    assert tools.read_file("notes.txt", end_line=1) == "alpha\n"


def test_read_file_missing(tools):
    assert tools.read_file("nope.txt").startswith("ERROR:")


def test_read_file_truncates_to_budget(cfg, project_root):
    cfg.max_output_chars = 20
    (project_root / "big.txt").write_text("y" * 500)
    assert "truncated" in Tools(cfg).read_file("big.txt")


def test_read_file_tolerates_invalid_utf8(tools, project_root):
    (project_root / "bin.dat").write_bytes(b"ok\xff\xfebytes")
    assert "ok" in tools.read_file("bin.dat")  # errors="replace", no exception


def test_list_dir(tools):
    out = tools.list_dir(".")
    assert "hello.py" in out and "pkg" in out
    assert tools.list_dir("hello.py").startswith("ERROR:")  # not a directory
    empty = os.path.join(tools.cfg.project_root, "empty")
    os.makedirs(empty)
    assert tools.list_dir("empty") == "(empty)"


# ---------------- search_files (ripgrep) ----------------

def test_search_files_basic(tools):
    out = tools.search_files("def hello")
    assert "hello.py:1:" in out


def test_search_files_no_match(tools):
    assert tools.search_files("zzz-not-here-zzz") == "(no matches)"


def test_search_files_glob_filter(tools):
    assert "notes.txt" not in tools.search_files("alpha", glob="*.py")
    assert "notes.txt" in tools.search_files("alpha", glob="*.txt")


def test_search_files_case_insensitive(tools):
    assert tools.search_files("DEF HELLO") == "(no matches)"
    assert "hello.py" in tools.search_files("DEF HELLO", case_insensitive=True)


def test_search_files_invalid_regex(tools):
    assert tools.search_files("(unclosed").startswith("ERROR: invalid regex")


def test_search_files_accepts_a_single_file_as_the_path(tools):
    """ripgrep drops the filename prefix when handed one file, so the result
    line is "<lineno>:<text>" — unpacking it as path:lineno:text used to raise
    ValueError on the very common "search within this file" call."""
    out = tools.search_files("def hello", path="hello.py")
    assert out == "hello.py:1:def hello():"


def test_search_files_single_file_tolerates_a_path_prefixed_line(tools, mocker):
    """Defensive: if ripgrep ever does prefix the filename for a single file,
    fall back to the path:lineno:text shape rather than mangling the output."""
    target = os.path.join(os.path.realpath(tools.cfg.project_root), "hello.py")
    mocker.patch("omni.tools._rg_search", return_value=[f"{target}:1:def hello():"])
    assert tools.search_files("def hello", path="hello.py") == "hello.py:1:def hello():"


def test_search_files_single_file_no_match(tools):
    assert tools.search_files("zzz-not-here-zzz", path="hello.py") == "(no matches)"


def test_search_and_glob_stay_relative_through_a_symlinked_root(tmp_path):
    """A project root reached through a symlink (/tmp on macOS, a symlinked
    checkout) used to yield paths like ../../../private/tmp/... because the
    match was made relative to the raw root while ripgrep reported the
    resolved one."""
    real = tmp_path / "real"
    (real / "pkg").mkdir(parents=True)
    (real / "pkg" / "mod.py").write_text("MARKER = 1\n")
    link = tmp_path / "link"
    link.symlink_to(real)

    t = Tools(AgentConfig(project_root=str(link)))
    assert t.search_files("MARKER") == "pkg/mod.py:1:MARKER = 1"
    assert t.glob_files("**/*.py") == "pkg/mod.py"


def test_search_files_skips_noise_dirs(tools, project_root):
    noisy = project_root / "node_modules" / "dep"
    noisy.mkdir(parents=True)
    (noisy / "index.js").write_text("UNIQUEMARKER\n")
    (project_root / "src.py").write_text("UNIQUEMARKER\n")
    out = tools.search_files("UNIQUEMARKER")
    assert "src.py" in out
    assert "node_modules" not in out


def test_search_files_paths_are_relative_to_root(tools):
    out = tools.search_files("import os")
    assert out.startswith("pkg/mod.py:") and os.path.isabs(out) is False


# ---------------- glob_files ----------------

def test_glob_files(tools):
    assert "hello.py" in tools.glob_files("*.py")
    assert "pkg/mod.py" in tools.glob_files("**/*.py").replace(os.sep, "/")
    assert tools.glob_files("*.nomatch") == "(no matches)"


def test_glob_files_skips_noise_dirs(tools, project_root):
    d = project_root / "__pycache__"
    d.mkdir()
    (d / "x.py").write_text("")
    assert "__pycache__" not in tools.glob_files("**/*.py")


# ---------------- write_file / edit_file + previews ----------------

def test_write_file_new_returns_all_addition_diff(tools, project_root):
    out = tools.write_file("new.py", "a = 1\nb = 2\n")
    assert (project_root / "new.py").read_text() == "a = 1\nb = 2\n"
    assert "+a = 1" in out and "+b = 2" in out and "-" not in out.split("@@")[-1]


def test_write_file_refuses_clobber_without_overwrite(tools, project_root):
    out = tools.write_file("hello.py", "replaced")
    assert out.startswith("ERROR:") and "overwrite=true" in out
    assert project_root.joinpath("hello.py").read_text() != "replaced"


def test_write_file_overwrite_shows_diff(tools, project_root):
    out = tools.write_file("hello.py", "def hello():\n    return 2\n", overwrite=True)
    assert "-    return 1" in out and "+    return 2" in out
    assert "return 2" in (project_root / "hello.py").read_text()


def test_write_file_creates_parent_dirs(tools, project_root):
    tools.write_file("deep/nested/x.txt", "hi")
    assert (project_root / "deep" / "nested" / "x.txt").read_text() == "hi"


def test_edit_file_unique_match(tools, project_root):
    out = tools.edit_file("hello.py", "return 1", "return 42")
    assert "Edited" in out and "+    return 42" in out
    assert "return 42" in (project_root / "hello.py").read_text()


def test_edit_file_not_found_leaves_file_alone(tools, project_root):
    before = (project_root / "hello.py").read_text()
    assert tools.edit_file("hello.py", "nonexistent", "x").startswith("ERROR:")
    assert (project_root / "hello.py").read_text() == before


def test_edit_file_ambiguous_match_refuses(tools, project_root):
    (project_root / "dup.py").write_text("x = 1\nx = 1\n")
    out = tools.edit_file("dup.py", "x = 1", "x = 2")
    assert out.startswith("ERROR:") and "2 locations" in out
    assert (project_root / "dup.py").read_text() == "x = 1\nx = 1\n"


def test_edit_file_missing_file(tools):
    assert tools.edit_file("nope.py", "a", "b").startswith("ERROR:")


def test_preview_edit_does_not_write(tools, project_root):
    ok, diff = tools.preview_edit("hello.py", "return 1", "return 9")
    assert ok and "+    return 9" in diff
    assert "return 1" in (project_root / "hello.py").read_text()  # unchanged


def test_preview_edit_error_cases(tools, project_root):
    assert tools.preview_edit("nope.py", "a", "b") == (False, "nope.py does not exist")
    ok, msg = tools.preview_edit("hello.py", "zzz", "b")
    assert not ok and "not found" in msg
    (project_root / "dup.py").write_text("q\nq\n")
    ok, msg = tools.preview_edit("dup.py", "q", "z")
    assert not ok and "not unique" in msg


def test_preview_write_new_vs_overwrite(tools, project_root):
    is_new, diff = tools.preview_write("brand.py", "x = 1\n")
    assert is_new and "+x = 1" in diff
    assert not (project_root / "brand.py").exists()  # dry run

    is_new, diff = tools.preview_write("hello.py", "changed\n", overwrite=True)
    assert not is_new and "-def hello():" in diff


# ---------------- save_memory ----------------

def test_save_memory_is_confined_to_the_project_root(cfg):
    """memory_path is config, not model input — but it's still a path, and it
    was the one path in this module that skipped the scope check."""
    cfg.memory_path = "../escaped-memory.md"
    with pytest.raises(PathScopeError):
        Tools(cfg).save_memory("a note")


def test_save_memory_appends_dated_bullets(tools, project_root):
    tools.save_memory("uses pytest")
    tools.save_memory("config in .env")
    body = (project_root / tools.cfg.memory_path).read_text()
    assert body.count("- [") == 2
    assert "uses pytest" in body and "config in .env" in body


# ---------------- run_shell ----------------

def test_run_shell_captures_output(tools):
    out = tools.run_shell("echo hello-shell")
    assert "exit_code: 0" in out and "hello-shell" in out


def test_run_shell_reports_nonzero_exit(tools):
    assert "exit_code: 3" in tools.run_shell("exit 3")


def test_run_shell_runs_in_project_root(tools):
    assert "hello.py" in tools.run_shell("ls")


@pytest.mark.parametrize("cmd", ["rm -rf /", "sudo rm x", "mkfs.ext4 /dev/sda", "dd if=/dev/zero"])
def test_run_shell_blocks_denied_patterns(tools, cmd):
    out = tools.run_shell(cmd)
    assert out.startswith("ERROR:") and "blocked by policy" in out


def test_run_shell_timeout(cfg):
    cfg.shell_timeout_s = 1
    out = Tools(cfg).run_shell("sleep 5")
    assert "timed out" in out


def test_run_shell_truncates_output(cfg):
    cfg.max_output_chars = 50
    assert "truncated" in Tools(cfg).run_shell("seq 1 500")
