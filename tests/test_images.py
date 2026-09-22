"""Pasting an image into the prompt, and the request it turns into.

The path has four joints and each is checked here: the clipboard read (which
is a subprocess on every platform), the placeholder the keypress leaves in
the input line, the message payload built from the bytes, and the history
that has to store a message which is no longer a plain string.
"""

import asyncio
import base64
import sys
import copy
import json

import pytest
from prompt_toolkit.keys import Keys

from omni import clipboard, ui
from omni import agent as agent_mod
from omni.agent import CodingAgent, build_user_message, image_part
from omni.session_store import SessionStore
from omni.tui import TuiApp

PNG = b"\x89PNG\r\n\x1a\n" + b"pretend pixels"


# ---------------- building the payload ----------------

def test_text_only_stays_a_plain_string():
    """Nothing about a text session changes: servers that know nothing about
    content parts still see exactly what they saw before."""
    assert build_user_message("just words") == {"role": "user", "content": "just words"}


def test_an_image_turns_the_message_into_content_parts():
    message = build_user_message("what is this?", [{"mime": "image/png", "data": PNG}])
    assert message["role"] == "user"
    assert message["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
    ]


def test_the_text_part_comes_first():
    parts = build_user_message("q", [{"mime": "image/png", "data": PNG}])["content"]
    assert parts[0]["type"] == "text"


def test_every_image_gets_its_own_part():
    parts = build_user_message("q", [{"mime": "image/png", "data": PNG},
                                      {"mime": "image/jpeg", "data": b"jpeg"}])["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_declared_mime_is_the_attachment_s_own():
    part = image_part("image/gif", b"gif")
    assert part["image_url"]["url"].startswith("data:image/gif;base64,")


def test_a_missing_mime_falls_back_to_png():
    assert image_part(None, b"x")["image_url"]["url"].startswith("data:image/png;base64,")


def test_an_attachment_without_bytes_is_dropped():
    """An empty part is worse than no part — the server would try to decode
    it — and if nothing survives the message goes back to a plain string."""
    assert build_user_message("q", [{"mime": "image/png", "data": b""}]) == {
        "role": "user", "content": "q"}


def test_the_base64_is_decodable_back_to_the_original_bytes():
    url = build_user_message("q", [{"mime": "image/png", "data": PNG}])["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


# ---------------- the agent's request ----------------

@pytest.fixture
def agent(cfg, mocker):
    mocker.patch.object(agent_mod.ui, "step_display")
    mocker.patch.object(agent_mod.ui, "elapsed_note")
    mocker.patch.object(agent_mod.ui, "banner")
    return CodingAgent(cfg)


@pytest.fixture
def client(mocker):
    c = mocker.AsyncMock()
    c.list_llm_tools.return_value = []
    return c


def replies(mocker, *messages):
    """chat() answering with each message in turn, snapshotting what it was
    sent (the agent mutates the list in place)."""
    async def fake(*args, **kwargs):
        fake.sent.append(copy.deepcopy(kwargs.get("messages", [])))
        return messages[min(fake.calls, len(messages) - 1)]

    fake.calls = 0
    fake.sent = []
    m = mocker.patch.object(agent_mod, "chat", side_effect=fake)
    m.sent = fake.sent
    return m


async def test_the_agent_sends_the_parts_to_the_llm(agent, client, mocker):
    chat = replies(mocker, {"role": "assistant", "content": "a red square"})
    await agent.run("what is this?", client=client,
                    attachments=[{"mime": "image/png", "data": PNG}])
    user = [m for m in chat.sent[0] if m["role"] == "user"][-1]
    assert isinstance(user["content"], list)
    assert user["content"][0] == {"type": "text", "text": "what is this?"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_without_attachments_the_user_message_is_unchanged(agent, client, mocker):
    chat = replies(mocker, {"role": "assistant", "content": "ok"})
    await agent.run("plain question", client=client)
    user = [m for m in chat.sent[0] if m["role"] == "user"][-1]
    assert user["content"] == "plain question"


async def test_the_image_is_still_there_on_the_next_turn(agent, client, mocker):
    """The message stays in the conversation, so a follow-up question about
    the same picture doesn't need it pasted again."""
    chat = replies(mocker, {"role": "assistant", "content": "a red square"})
    await agent.run("what is this?", client=client,
                    attachments=[{"mime": "image/png", "data": PNG}])
    session_id = agent.session_id
    await agent.run("and what colour?", resume_session_id=session_id, client=client)
    earlier = [m for m in chat.sent[-1] if m["role"] == "user"]
    assert any(isinstance(m["content"], list) for m in earlier)


# ---------------- storing it ----------------

@pytest.fixture
def store(tmp_path):
    return SessionStore(str(tmp_path / "s.db"))


def test_content_parts_survive_a_round_trip(store):
    """A resumed session has to send the model what it saw the first time,
    images included, or the conversation stops making sense."""
    sid = store.create_session("/p", "m", "t")
    message = build_user_message("what is this?", [{"mime": "image/png", "data": PNG}])
    store.append_message(sid, 0, message)
    assert store.load_messages(sid)[0]["content"] == message["content"]


def test_the_stored_text_stays_readable(store):
    """`content` keeps the words plus a count, so anything reading the
    conversation as text (the resumed-history panel, /sessions) still can."""
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, build_user_message("what is this?",
                                                     [{"mime": "image/png", "data": PNG}]))
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute("SELECT content, content_parts FROM messages").fetchone()
    assert row[0] == "what is this? [1 image]"
    assert json.loads(row[1])[0]["text"] == "what is this?"


def test_two_images_are_counted_in_the_stored_text(store):
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, build_user_message("these", [{"mime": "image/png", "data": PNG},
                                                               {"mime": "image/png", "data": PNG}]))
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT content FROM messages").fetchone()[0] == "these [2 images]"


def test_replace_messages_stores_parts_too(store):
    """Compaction rewrites the whole history through this path."""
    sid = store.create_session("/p", "m", "t")
    message = build_user_message("q", [{"mime": "image/png", "data": PNG}])
    store.replace_messages(sid, [{"role": "system", "content": "s"}, message])
    assert store.load_messages(sid)[1]["content"] == message["content"]


def test_plain_messages_store_no_parts(store):
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "user", "content": "hi"})
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT content_parts FROM messages").fetchone()[0] is None


def test_an_older_database_gains_the_column(tmp_path):
    """The migration pattern the other added columns use: an existing database
    must open, not fail."""
    import sqlite3
    path = str(tmp_path / "old.db")
    SessionStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE messages DROP COLUMN content_parts")
    store = SessionStore(path)                     # migrates on open
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, build_user_message("q", [{"mime": "image/png", "data": PNG}]))
    assert isinstance(store.load_messages(sid)[0]["content"], list)


# ---------------- the keypress ----------------

@pytest.fixture
def app(mocker):
    mocker.patch("omni.ui.instruction")
    application = TuiApp({"/exit": "leave"}, "s", "m")
    ui.use_tui(application)
    yield application
    ui.use_tui(None)


def paste_binding(application):
    for b in application._app.key_bindings.bindings:
        if b.keys == (Keys.ControlV,) and b.filter():
            return b.handler
    raise AssertionError("ctrl+v is not bound in this mode")


async def test_pasting_an_image_leaves_a_placeholder(app, mocker):
    mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
    paste_binding(app)(mocker.Mock())
    assert app._buffer.text == "[Image #1]"
    assert app.main.attachments == [{"mime": "image/png", "data": PNG}]


async def test_placeholders_are_numbered_as_you_paste(app, mocker):
    mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
    handler = paste_binding(app)
    handler(mocker.Mock())
    app._buffer.insert_text(" and ")
    handler(mocker.Mock())
    assert app._buffer.text == "[Image #1] and [Image #2]"
    assert len(app.main.attachments) == 2


async def test_the_placeholder_sits_where_the_cursor_is(app, mocker):
    mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
    app._buffer.insert_text("describe ")
    paste_binding(app)(mocker.Mock())
    app._buffer.insert_text(" please")
    assert app._buffer.text == "describe [Image #1] please"


async def test_no_image_on_the_clipboard_pastes_text_instead(app, mocker):
    mocker.patch("omni.clipboard.grab_image", return_value=None)
    mocker.patch("omni.clipboard.clipboard_text", return_value="some copied text")
    paste_binding(app)(mocker.Mock())
    assert app._buffer.text == "some copied text"
    assert app.main.attachments == []


async def test_a_pasted_newline_does_not_submit_the_line(app, mocker):
    """Enter is submit, so a multi-line paste has to arrive as one line."""
    mocker.patch("omni.clipboard.grab_image", return_value=None)
    mocker.patch("omni.clipboard.clipboard_text", return_value="one\ntwo")
    paste_binding(app)(mocker.Mock())
    assert app._buffer.text == "one two"


def _active_binding(application, key):
    for b in application._app.key_bindings.bindings:
        if b.keys == (key,) and b.filter():
            return b.handler
    return None


async def test_ctrl_b_is_an_alternate_paste_on_windows_only(app, mocker):
    """Windows Terminal binds Ctrl+V to its own paste and swallows it, so the
    c-v binding never fires there — Ctrl+B is offered as an alternate trigger
    for the same paste, and only on Windows (elsewhere Ctrl+V works and Ctrl+B
    stays free)."""
    handler = _active_binding(app, Keys.ControlB)
    if sys.platform == "win32":
        assert handler is not None, "Ctrl+B should be an alternate paste on Windows"
        mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
        handler(mocker.Mock())
        assert app._buffer.text == "[Image #1]"
        assert app.main.attachments == [{"mime": "image/png", "data": PNG}]
    else:
        assert handler is None, "Ctrl+B should not be bound off Windows"


async def test_an_empty_clipboard_types_nothing(app, mocker):
    mocker.patch("omni.clipboard.grab_image", return_value=None)
    mocker.patch("omni.clipboard.clipboard_text", return_value="")
    paste_binding(app)(mocker.Mock())
    assert app._buffer.text == ""


async def test_the_image_goes_to_the_pane_you_pasted_at(app, mocker):
    """Each agent has its own prompt line; an image belongs to the one that
    is going to be asked about it."""
    mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
    other = app.add_pane("agent 1")
    app.focus(app.panes.index(other))
    paste_binding(app)(mocker.Mock())
    assert other.attachments and app.main.attachments == []


async def test_the_echo_says_how_many_images_went_along(app, mocker):
    echo = mocker.patch("omni.ui.instruction")
    mocker.patch("omni.clipboard.grab_image", return_value=("image/png", PNG))
    paste_binding(app)(mocker.Mock())
    for b in app._app.key_bindings.bindings:
        if b.keys == (Keys.ControlM,) and b.filter():
            b.handler(mocker.Mock())
            break
    echo.assert_called_once_with("[Image #1]", images=1)


# ---------------- reading the clipboard ----------------

def test_grab_image_returns_none_when_the_tool_is_missing(mocker):
    mocker.patch("omni.clipboard._run", side_effect=OSError("no such binary"))
    mocker.patch("omni.clipboard._file_paths", return_value=[])
    mocker.patch("omni.clipboard.clipboard_text", return_value="")
    assert clipboard.grab_image() is None


def test_grab_image_never_raises(mocker):
    """The clipboard is not worth an exception: a failure here has to look
    like an ordinary text paste, not a crashed session."""
    mocker.patch("omni.clipboard._file_paths", return_value=[])
    mocker.patch("omni.clipboard._from_macos", side_effect=RuntimeError("boom"))
    mocker.patch("omni.clipboard._from_linux", side_effect=RuntimeError("boom"))
    mocker.patch("omni.clipboard._from_windows", side_effect=RuntimeError("boom"))
    assert clipboard.grab_image() is None


def test_an_oversized_image_is_refused(mocker):
    """It would be base64'd into the request and kept there for the rest of
    the session."""
    mocker.patch("omni.clipboard._file_paths", return_value=[])
    mocker.patch("omni.clipboard._from_macos",
                  return_value=("image/png", b"x" * (clipboard.MAX_BYTES + 1)))
    mocker.patch("omni.clipboard._from_linux",
                  return_value=("image/png", b"x" * (clipboard.MAX_BYTES + 1)))
    mocker.patch("omni.clipboard._from_windows",
                  return_value=("image/png", b"x" * (clipboard.MAX_BYTES + 1)))
    assert clipboard.grab_image() is None


def test_a_copied_file_beats_the_icon_the_clipboard_offers(mocker, tmp_path):
    """Cmd+C (or right-click → Copy) on a file in Finder puts a *reference*
    to the file on the clipboard and, alongside it, a picture of the file's
    icon. Coercing the clipboard to a PNG hands you the icon — a model then
    dutifully describes "a JPEG placeholder icon" instead of the photo. The
    file reference has to win."""
    path = tmp_path / "white-house.jpeg"
    path.write_bytes(b"\xff\xd8\xff the real photo")
    mocker.patch("omni.clipboard._file_paths", return_value=[str(path)])
    mocker.patch("omni.clipboard._from_macos", return_value=("image/png", b"an icon"))
    mocker.patch("omni.clipboard._from_linux", return_value=("image/png", b"an icon"))
    mocker.patch("omni.clipboard._from_windows", return_value=("image/png", b"an icon"))
    assert clipboard.grab_image() == ("image/jpeg", b"\xff\xd8\xff the real photo")


def test_a_copied_file_that_is_not_an_image_falls_through(mocker, tmp_path):
    """Copying a .txt in Finder shouldn't stop an image on the clipboard from
    being found."""
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    mocker.patch("omni.clipboard._file_paths", return_value=[str(path)])
    reader = {"darwin": "_from_macos", "win32": "_from_windows"}.get(sys.platform, "_from_linux")
    mocker.patch(f"omni.clipboard.{reader}", return_value=("image/png", PNG))
    assert clipboard.grab_image() == ("image/png", PNG)


def test_a_file_reference_with_no_image_flavour_at_all(mocker, tmp_path):
    """The other half of the same bug: when the clipboard holds *only* a file
    reference, pbpaste is empty, so before this there was nothing to find and
    the paste silently did nothing."""
    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    mocker.patch("omni.clipboard._file_paths", return_value=[str(path)])
    mocker.patch("omni.clipboard._from_macos", return_value=None)
    mocker.patch("omni.clipboard._from_linux", return_value=None)
    mocker.patch("omni.clipboard._from_windows", return_value=None)
    mocker.patch("omni.clipboard.clipboard_text", return_value="")
    assert clipboard.grab_image() == ("image/png", PNG)


def test_macos_asks_the_clipboard_for_a_file_url(mocker):
    run = mocker.patch("omni.clipboard._run",
                        return_value=mocker.Mock(returncode=0, stdout=b"/tmp/a.png\n"))
    mocker.patch.object(clipboard.sys, "platform", "darwin")
    assert clipboard._file_paths() == ["/tmp/a.png"]
    assert "furl" in run.call_args.args[0][-1]


def test_windows_asks_for_the_file_drop_list(mocker):
    mocker.patch("omni.clipboard._run",
                  return_value=mocker.Mock(returncode=0, stdout=b"C:\\pics\\a.png\r\n"))
    mocker.patch.object(clipboard.sys, "platform", "win32")
    assert clipboard._file_paths() == ["C:\\pics\\a.png"]


def test_linux_reads_a_uri_list_and_falls_back_to_xclip(mocker):
    def fake_run(argv, **kw):
        ok = argv[0] == "xclip"
        return mocker.Mock(returncode=0 if ok else 1,
                            stdout=b"file:///tmp/a%20b.png\n" if ok else b"")

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard._file_paths() == ["file:///tmp/a%20b.png"]


def test_no_uri_list_tool_on_linux_is_not_an_error(mocker):
    """Neither wl-paste nor xclip installed: no files, no exception."""
    mocker.patch("omni.clipboard._run", side_effect=FileNotFoundError())
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard._file_paths() == []


def test_linux_survives_xclip_being_absent_after_wl_paste_fails(mocker):
    def fake_run(argv, **kw):
        if argv[0] == "xclip":
            raise FileNotFoundError()
        return mocker.Mock(returncode=1, stdout=b"")

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard._file_paths() == []


def test_an_empty_uri_list_is_no_files(mocker):
    mocker.patch("omni.clipboard._run", return_value=mocker.Mock(returncode=1, stdout=b""))
    mocker.patch.object(clipboard.sys, "platform", "darwin")
    assert clipboard._file_paths() == []


def test_a_gnome_copied_files_header_is_not_a_path(mocker):
    """Nautilus prefixes the list with the operation."""
    mocker.patch("omni.clipboard._run",
                  return_value=mocker.Mock(returncode=0, stdout=b"copy\nfile:///tmp/a.png\n"))
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard._file_paths() == ["file:///tmp/a.png"]


def test_no_file_on_the_clipboard_is_not_an_error(mocker):
    mocker.patch("omni.clipboard._run", side_effect=FileNotFoundError())
    assert clipboard._file_paths() == []


def test_a_percent_encoded_path_is_decoded(tmp_path):
    """file:// URIs escape spaces, and "a%20b.png" is not a filename."""
    path = tmp_path / "white house.png"
    path.write_bytes(PNG)
    uri = "file://" + str(path).replace(" ", "%20")
    assert clipboard._from_text_path(uri) == ("image/png", PNG)


def test_a_clipboard_holding_a_path_to_an_image_counts(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    assert clipboard._from_text_path(str(path)) == ("image/png", PNG)


def test_a_file_url_is_accepted(tmp_path):
    path = tmp_path / "shot.jpg"
    path.write_bytes(b"jpeg")
    assert clipboard._from_text_path(f"file://{path}") == ("image/jpeg", b"jpeg")


def test_a_quoted_path_is_accepted(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    assert clipboard._from_text_path(f'"{path}"')[0] == "image/png"


def test_a_path_to_something_that_is_not_an_image_is_not_one(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    assert clipboard._from_text_path(str(path)) is None


def test_ordinary_text_is_not_a_path():
    assert clipboard._from_text_path("please describe the diagram") is None


def test_a_path_that_does_not_exist_is_not_an_image(tmp_path):
    assert clipboard._from_text_path(str(tmp_path / "missing.png")) is None


def test_macos_writes_the_clipboard_png_to_a_file(mocker, tmp_path):
    """osascript coerces the clipboard to a PNG and writes it; the bytes come
    back from the file it wrote."""
    written = {}

    def fake_run(argv, **kw):
        path = [a for a in argv if a.startswith("set f to")][0].split('"')[1]
        written["path"] = path
        with open(path, "wb") as handle:
            handle.write(PNG)
        return mocker.Mock(returncode=0)

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    assert clipboard._from_macos() == ("image/png", PNG)


def test_macos_tries_the_next_flavour_when_a_coercion_fails(mocker):
    """Some apps offer only GIF or TIFF. A failed coercion still leaves the
    file behind, so an empty one must not count as an image."""
    tried = []

    def fake_run(argv, **kw):
        flavour = [a for a in argv if "write (the clipboard" in a][0]
        tried.append(flavour)
        if "GIFf" in flavour:
            path = [a for a in argv if a.startswith("set f to")][0].split('"')[1]
            with open(path, "wb") as handle:
                handle.write(b"GIF89a")
            return mocker.Mock(returncode=0)
        return mocker.Mock(returncode=1)

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    assert clipboard._from_macos() == ("image/gif", b"GIF89a")
    assert len(tried) == 2                      # PNG refused, GIF taken


def test_macos_gives_up_quietly_when_nothing_coerces(mocker):
    mocker.patch("omni.clipboard._run", return_value=mocker.Mock(returncode=1))
    assert clipboard._from_macos() is None


def test_macos_without_osascript_is_not_an_error(mocker):
    mocker.patch("omni.clipboard._run", side_effect=FileNotFoundError())
    assert clipboard._from_macos() is None


def test_windows_saves_the_clipboard_image_as_png(mocker):
    def fake_run(argv, **kw):
        path = argv[-1].split("$image.Save('")[1].split("'")[0].replace("\\\\", "\\")
        with open(path, "wb") as handle:
            handle.write(PNG)
        return mocker.Mock(returncode=0)

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    assert clipboard._from_windows() == ("image/png", PNG)


def test_windows_with_no_image_on_the_clipboard(mocker):
    """The script exits 1 when GetImage() returns null."""
    mocker.patch("omni.clipboard._run", return_value=mocker.Mock(returncode=1))
    assert clipboard._from_windows() is None


def test_windows_without_powershell_is_not_an_error(mocker):
    mocker.patch("omni.clipboard._run", side_effect=OSError())
    assert clipboard._from_windows() is None


def test_clipboard_text_reads_the_platform_tool(mocker):
    mocker.patch("omni.clipboard._run",
                  return_value=mocker.Mock(returncode=0, stdout=b"copied words"))
    assert clipboard.clipboard_text() == "copied words"


def test_clipboard_text_falls_back_to_xclip(mocker):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv[0])
        ok = argv[0] == "xclip"
        return mocker.Mock(returncode=0 if ok else 1, stdout=b"x11 words" if ok else b"")

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard.clipboard_text() == "x11 words"
    assert calls[-1] == "xclip"


def test_clipboard_text_is_empty_when_no_tool_exists(mocker):
    mocker.patch("omni.clipboard._run", side_effect=FileNotFoundError())
    assert clipboard.clipboard_text() == ""


def test_clipboard_text_survives_undecodable_bytes(mocker):
    mocker.patch("omni.clipboard._run",
                  return_value=mocker.Mock(returncode=0, stdout=b"caf\xff"))
    assert clipboard.clipboard_text().startswith("caf")


def test_grab_image_falls_back_to_a_path_on_the_clipboard(mocker, tmp_path):
    path = tmp_path / "diagram.png"
    path.write_bytes(PNG)
    mocker.patch("omni.clipboard._file_paths", return_value=[])
    mocker.patch("omni.clipboard._from_macos", return_value=None)
    mocker.patch("omni.clipboard._from_linux", return_value=None)
    mocker.patch("omni.clipboard._from_windows", return_value=None)
    mocker.patch("omni.clipboard.clipboard_text", return_value=str(path))
    assert clipboard.grab_image() == ("image/png", PNG)


def test_grab_image_returns_an_image_from_the_clipboard_itself(mocker):
    mocker.patch("omni.clipboard._file_paths", return_value=[])
    reader = {"darwin": "_from_macos", "win32": "_from_windows"}.get(sys.platform, "_from_linux")
    mocker.patch(f"omni.clipboard.{reader}", return_value=("image/png", PNG))
    assert clipboard.grab_image() == ("image/png", PNG)


def test_linux_skips_a_tool_that_is_not_installed(mocker):
    """wl-paste missing under X11 is the normal case, not a failure."""
    def fake_run(argv, **kw):
        if argv[0] == "wl-paste":
            raise FileNotFoundError()
        return mocker.Mock(returncode=0, stdout=PNG)

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    assert clipboard._from_linux() == ("image/png", PNG)


def test_linux_with_no_image_anywhere(mocker):
    mocker.patch("omni.clipboard._run", return_value=mocker.Mock(returncode=0, stdout=b""))
    assert clipboard._from_linux() is None


def test_clipboard_text_when_the_fallback_also_fails(mocker):
    def fake_run(argv, **kw):
        if argv[0] == "xclip":
            raise FileNotFoundError()
        return mocker.Mock(returncode=1, stdout=b"")

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    mocker.patch.object(clipboard.sys, "platform", "linux")
    assert clipboard.clipboard_text() == ""


def test_run_actually_shells_out():
    """The one line that touches subprocess for real, so the argv shape and
    the capture flags are checked rather than assumed."""
    result = clipboard._run([sys.executable, "-c", "print('hi')"])
    assert result.returncode == 0 and result.stdout.strip() == b"hi"


def test_linux_prefers_wayland_then_falls_back_to_xclip(mocker):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv[0])
        ok = argv[0] == "xclip"
        return mocker.Mock(returncode=0 if ok else 1, stdout=PNG if ok else b"")

    mocker.patch("omni.clipboard._run", side_effect=fake_run)
    assert clipboard._from_linux() == ("image/png", PNG)
    assert calls == ["wl-paste", "xclip"]


# ---------------- from the prompt to the turn ----------------
#
# The pasted image sits on the pane until a line is submitted; these check it
# reaches that turn and only that turn.

def run_repl(mocker, cfg, lines, on_line=None):
    """The interactive loop over a scripted list of lines, with agent.run
    mocked. `on_line` runs just before each line is "typed", which is where a
    paste would have happened."""
    from omni import cli as cli_mod

    mocker.patch.object(cli_mod, "_print_header")
    mocker.patch("omni.ui.final_result")
    mocker.patch("omni.ui.thinking")
    mocker.patch.object(cli_mod, "_make_tui", lambda *a, **k: None)
    c = mocker.AsyncMock()
    c.list_llm_tools.return_value = []
    c.list_prompts.return_value = {}
    c.list_resources.return_value = {}
    c.server_status = mocker.Mock(return_value=[])
    c.server_names = mocker.Mock(return_value=[])
    c.__aenter__.return_value = c
    c.__aexit__.return_value = False
    mocker.patch.object(cli_mod, "MCPToolClient", return_value=c)
    remaining = iter(list(lines) + ["/exit"])

    async def fake_next(tui, main_pane):
        for _ in range(200):
            if main_pane.task is None or main_pane.task.done():
                break
            await asyncio.sleep(0.005)
        if on_line:
            on_line(main_pane)
        try:
            return main_pane, next(remaining)
        except StopIteration:
            return None, None

    mocker.patch.object(cli_mod, "_next_instruction", fake_next)
    agent_run = mocker.patch.object(cli_mod.CodingAgent, "run",
                                     mocker.AsyncMock(return_value="ok"))
    asyncio.run(cli_mod._interactive(cfg, None, None))
    return agent_run


def test_a_pasted_image_reaches_the_turn(mocker, cfg):
    def paste(pane):
        if not pane.attachments:
            pane.attachments.append({"mime": "image/png", "data": PNG})

    agent_run = run_repl(mocker, cfg, ["describe [Image #1]"], on_line=paste)
    assert agent_run.await_args.kwargs["attachments"] == [{"mime": "image/png", "data": PNG}]


def test_the_image_goes_with_one_turn_only(mocker, cfg):
    """It is already in the conversation after that turn; sending the bytes
    again would pay for the same picture twice."""
    pasted = []

    def paste_once(pane):
        if not pasted:
            pasted.append(True)
            pane.attachments.append({"mime": "image/png", "data": PNG})

    agent_run = run_repl(mocker, cfg, ["describe [Image #1]", "and now?"],
                          on_line=paste_once)
    assert agent_run.await_args_list[0].kwargs["attachments"]
    assert agent_run.await_args_list[1].kwargs["attachments"] == []


# ---------------- everything that reads a conversation ----------------
#
# Once a message's content can be a list, every place that measured,
# summarized or displayed it as a string had to learn the other shape.

from omni.agent import _render_for_summary
from omni.llm_client import message_text


def test_message_text_of_a_plain_message():
    assert message_text({"role": "user", "content": "hello"}) == "hello"


def test_message_text_of_a_null_content():
    assert message_text({"role": "assistant", "content": None}) == ""


def test_message_text_flattens_parts_and_counts_images():
    text = message_text({"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]})
    assert text == "what is this? [1 image]"


def test_message_text_never_returns_the_base64():
    """It feeds the character budget and the screen; a 200 KB data URI in
    either would be a bug you would notice as constant compaction."""
    url = "data:image/png;base64," + "x" * 200_000
    text = message_text({"role": "user", "content": [
        {"type": "text", "text": "small"}, {"type": "image_url", "image_url": {"url": url}}]})
    assert len(text) < 40 and "base64" not in text


def test_message_text_counts_several_images():
    parts = [{"type": "text", "text": "these two"}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}] * 2
    assert message_text({"content": parts}) == "these two [2 images]"


def test_message_text_of_images_with_no_words():
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}]
    assert message_text({"content": parts}) == "[1 image]"


def test_compaction_can_summarize_a_turn_with_an_image():
    """Compaction renders each message as text for the summarizing model —
    with a list content that used to raise AttributeError."""
    message = build_user_message("what is wrong here?", [{"mime": "image/png", "data": PNG}])
    assert _render_for_summary(message) == "user: what is wrong here? [1 image]"


def test_the_resumed_history_panel_renders_an_image_turn(capsys):
    """Resuming shows the prior conversation; an image turn has to draw as
    its words plus a count, not crash on a list."""
    message = build_user_message("what is this?", [{"mime": "image/png", "data": PNG}])
    ui.history_panel([{"role": "system", "content": "s"}, message,
                       {"role": "assistant", "content": "a red square"}])
    out = capsys.readouterr().out
    assert "what is this? [1 image]" in out
    assert "base64" not in out
