"""Git tools, with subprocess.run mocked via pytest-mock's `mocker`.

Mocking the boundary keeps these hermetic and fast (no repo fixtures, no
network for fetch/pull/push) and lets us assert the exact argv each tool
builds plus every failure branch — a non-zero exit, a timeout, a raised
OSError — which a real repo can't reproduce on demand.
"""

import subprocess

import pytest

from omni.tools import PathScopeError, Tools


def completed(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr=stderr)


@pytest.fixture
def run_mock(mocker):
    """subprocess.run patched at the module boundary (pytest-mock undoes it
    at teardown), defaulting to a clean success."""
    m = mocker.patch("omni.tools.subprocess.run")
    m.return_value = completed()
    return m


def argv(run_mock):
    return run_mock.call_args.args[0]


# ---------------- read-only git tools: argv + happy path ----------------

def test_git_diff(tools, run_mock):
    run_mock.return_value = completed(stdout="diff --git a/x b/x\n")
    assert "diff --git" in tools.git_diff()
    assert argv(run_mock) == ["git", "diff"]


def test_git_diff_empty_is_labelled(tools, run_mock):
    assert tools.git_diff() == "(no changes)"


def test_git_status(tools, run_mock):
    run_mock.return_value = completed(stdout="## main\n M x.py\n")
    assert "## main" in tools.git_status()
    assert argv(run_mock) == ["git", "status", "--short", "--branch"]


def test_git_status_empty_is_labelled(tools, run_mock):
    assert tools.git_status() == "(clean)"


def test_git_log_passes_max_count(tools, run_mock):
    run_mock.return_value = completed(stdout="abc123 2026-01-01 Dev: init\n")
    assert "abc123" in tools.git_log(max_count=5)
    assert argv(run_mock)[:3] == ["git", "log", "-5"]


def test_git_log_empty_is_labelled(tools, run_mock):
    assert tools.git_log() == "(no commits)"


def test_git_show_passes_ref(tools, run_mock):
    run_mock.return_value = completed(stdout="commit abc\n")
    assert "commit abc" in tools.git_show(ref="HEAD~2")
    assert argv(run_mock) == ["git", "show", "HEAD~2"]


def test_git_branch(tools, run_mock):
    run_mock.return_value = completed(stdout="* main abc [origin/main]\n")
    assert "main" in tools.git_branch()
    assert argv(run_mock) == ["git", "branch", "-vv"]


def test_git_fetch_passes_remote(tools, run_mock):
    run_mock.return_value = completed(stdout="", stderr="From github.com\n")
    tools.git_fetch(remote="upstream")
    assert argv(run_mock) == ["git", "fetch", "upstream"]


# ---------------- write git tools ----------------

def test_git_add_all_uses_dash_A(tools, run_mock):
    assert tools.git_add(".") == "Staged: ."
    assert argv(run_mock) == ["git", "add", "-A"]


def test_git_add_specific_paths_are_scoped_and_relative(tools, run_mock):
    tools.git_add("hello.py pkg/mod.py")
    assert argv(run_mock) == ["git", "add", "hello.py", "pkg/mod.py"]


def test_git_add_rejects_out_of_scope_path(tools, run_mock):
    with pytest.raises(PathScopeError):
        tools.git_add("../../etc/passwd")
    run_mock.assert_not_called()  # refused before any git ran


def test_git_commit_argv_and_output(tools, run_mock):
    run_mock.return_value = completed(stdout="[main abc] msg\n")
    out = tools.git_commit("my message")
    assert argv(run_mock) == ["git", "commit", "-m", "my message"]
    assert "exit_code: 0" in out and "[main abc]" in out


def test_git_commit_rejects_empty_message(tools, run_mock):
    assert tools.git_commit("   ").startswith("ERROR:")
    run_mock.assert_not_called()


def test_git_pull_and_push_argv(tools, run_mock):
    tools.git_pull()
    assert argv(run_mock) == ["git", "pull", "origin"]
    tools.git_pull(remote="up", branch="dev")
    assert argv(run_mock) == ["git", "pull", "up", "dev"]
    tools.git_push()
    assert argv(run_mock) == ["git", "push", "origin"]
    tools.git_push(remote="up", branch="dev")
    assert argv(run_mock) == ["git", "push", "up", "dev"]


# ---------------- failure branches ----------------

@pytest.mark.parametrize("call", [
    lambda t: t.git_log(),
    lambda t: t.git_show(),
    lambda t: t.git_branch(),
    lambda t: t.git_add("hello.py"),
])
def test_nonzero_exit_surfaces_stderr(tools, run_mock, call):
    run_mock.return_value = completed(stderr="fatal: not a git repository\n", code=128)
    out = call(tools)
    assert out.startswith("ERROR:") and "not a git repository" in out


@pytest.mark.parametrize("call", [
    lambda t: t.git_diff(),
    lambda t: t.git_status(),
    lambda t: t.git_log(),
    lambda t: t.git_show(),
    lambda t: t.git_branch(),
    lambda t: t.git_fetch(),
    lambda t: t.git_add("."),
    lambda t: t.git_commit("m"),
    lambda t: t.git_pull(),
    lambda t: t.git_push(),
])
def test_timeout_is_reported_not_raised(tools, run_mock, call):
    run_mock.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=15)
    out = call(tools)
    assert out.startswith("ERROR:")


@pytest.mark.parametrize("call", [
    lambda t: t.git_diff(),
    lambda t: t.git_status(),
    lambda t: t.git_commit("m"),
])
def test_missing_git_binary_is_reported_not_raised(tools, run_mock, call):
    run_mock.side_effect = FileNotFoundError("git not found")
    assert call(tools).startswith("ERROR:")


def test_git_tools_run_in_resolved_scope(tools, run_mock):
    """cwd must be the resolved in-scope path, never the caller's cwd."""
    tools.git_diff("pkg")
    assert run_mock.call_args.kwargs["cwd"].endswith("/pkg")


def test_read_only_git_output_is_truncated(cfg, run_mock):
    cfg.max_output_chars = 30
    run_mock.return_value = completed(stdout="z" * 500)
    assert "truncated" in Tools(cfg).git_diff()
