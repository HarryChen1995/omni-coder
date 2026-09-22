"""Reading an image off the system clipboard.

A terminal never hands an application a pasted image. Cmd+V is the terminal's
own paste — it inserts text and nothing else — and there is no escape sequence
for image data. So the application has to go and read the clipboard itself,
which is why the binding is Ctrl+V (a key the terminal passes through) rather
than Cmd+V.

Each platform is asked in its own way, and nothing here is a hard dependency:
if the tool isn't installed or the clipboard holds no image, `grab_image`
returns None and the keypress falls back to an ordinary text paste.
"""

import os
import subprocess
import sys
import tempfile
from urllib.parse import unquote

# Bigger than this and it isn't worth sending: it has to be base64'd into the
# request, and it stays in the conversation for every later turn.
MAX_BYTES = 10 * 1024 * 1024

_EXTENSION_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _run(argv: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, timeout=10, **kwargs)


def _from_file_reference() -> tuple:
    """The file the user copied, when they copied a *file* rather than a picture.

    Cmd+C on a file in Finder doesn't put the picture on the clipboard — it
    puts a reference to the file, and alongside it an image of the file's
    *icon*. Coercing the clipboard to a PNG therefore hands you a JPEG
    placeholder icon rather than the photo, which is a confusing thing to send
    a model. So the reference is asked for first, and the file read from disk;
    only if there is no file reference does the clipboard's own image count.

    (Copying inside Preview or a browser puts no file reference on the
    clipboard, so nothing changes for that case.)"""
    for path in _file_paths():
        found = _from_text_path(path)
        if found:
            return found
    return None


def _file_paths() -> list:
    """Paths of files on the clipboard — however this platform names them."""
    if sys.platform == "darwin":
        argv = ["osascript", "-e", "POSIX path of (the clipboard as «class furl»)"]
    elif sys.platform == "win32":
        # -Sta: clipboard OLE calls require a single-threaded apartment;
        # powershell.exe (Windows PowerShell 5.1) is STA by default but not in
        # every launch context, and reading the clipboard silently returns
        # nothing under MTA. Forcing it makes the read reliable.
        argv = ["powershell", "-NoProfile", "-Sta", "-Command",
                "Get-Clipboard -Format FileDropList | ForEach-Object { $_.FullName }"]
    else:
        argv = ["wl-paste", "--no-newline", "--type", "text/uri-list"]
    try:
        result = _run(argv)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0 and sys.platform not in ("darwin", "win32"):
        try:
            result = _run(["xclip", "-selection", "clipboard", "-t", "text/uri-list", "-o"])
        except (OSError, subprocess.SubprocessError):
            return []
    if result.returncode != 0:
        return []
    text = result.stdout.decode("utf-8", "replace")
    # GNOME's flavour of the same thing starts with a "copy"/"cut" line.
    return [line for line in text.splitlines()
            if line.strip() and line.strip() not in ("copy", "cut")]


def _from_macos() -> tuple:
    """AppleScript can coerce the clipboard to a PNG and write it to a file.

    Screenshots (Cmd+Shift+Ctrl+4) and images copied from a browser both land
    on the clipboard in a flavour this can reach. TIFF is tried last: some
    apps offer only that, and a server that can't read it will say so, which
    is better than pretending there was no image."""
    for flavour, mime in (("«class PNGf»", "image/png"),
                           ("«class GIFf»", "image/gif"),
                           ("TIFF picture", "image/tiff")):
        path = os.path.join(tempfile.mkdtemp(prefix="omni-clip-"), "clip")
        script = [
            "osascript",
            "-e", f'set f to (open for access POSIX file "{path}" with write permission)',
            "-e", f"write (the clipboard as {flavour}) to f",
            "-e", "close access f",
        ]
        try:
            result = _run(script)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0 and os.path.exists(path) and os.path.getsize(path):
            with open(path, "rb") as handle:
                return mime, handle.read()
        # AppleScript leaves the file behind even when the coercion failed.
        try:
            os.remove(path)
        except OSError:
            pass
    return None


def _from_linux() -> tuple:
    """wl-paste under Wayland, xclip under X11 — whichever is installed."""
    attempts = (
        (["wl-paste", "--no-newline", "--type", "image/png"], "image/png"),
        (["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], "image/png"),
    )
    for argv, mime in attempts:
        try:
            result = _run(argv)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout:
            return mime, result.stdout
    return None


_WINDOWS_SCRIPT = """
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$image = [System.Windows.Forms.Clipboard]::GetImage()
if ($image -eq $null) { exit 1 }
$image.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Png)
"""


def _from_windows() -> tuple:
    path = os.path.join(tempfile.mkdtemp(prefix="omni-clip-"), "clip.png")
    try:
        # -Sta: [Clipboard]::GetImage() returns $null under MTA rather than
        # raising, so without this the paste silently finds "no image".
        result = _run(["powershell", "-NoProfile", "-Sta", "-Command",
                        _WINDOWS_SCRIPT.replace("{path}", path.replace("\\", "\\\\"))])
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as handle:
            return "image/png", handle.read()
    return None


def _from_text_path(text: str) -> tuple:
    """A clipboard holding a *path* to an image counts as an image.

    Copying a file in Finder or a file manager gives you its path, and people
    paste paths by hand too — reading it is more useful than reporting that
    the clipboard held no image."""
    candidate = (text or "").strip().strip('"').strip("'")
    if candidate.startswith("file://"):
        # A URI, so spaces and anything else awkward arrive percent-encoded.
        candidate = unquote(candidate[len("file://"):])
    if not candidate or "\n" in candidate:
        return None
    candidate = os.path.expanduser(candidate)
    mime = _EXTENSION_MIME.get(os.path.splitext(candidate)[1].lower())
    if not mime or not os.path.isfile(candidate):
        return None
    with open(candidate, "rb") as handle:
        return mime, handle.read()


def clipboard_text() -> str:
    """Whatever text the clipboard holds, or "" — used to spot a pasted path."""
    commands = {
        "darwin": ["pbpaste"],
        "win32": ["powershell", "-NoProfile", "-Sta", "-Command", "Get-Clipboard"],
    }
    argv = commands.get(sys.platform, ["wl-paste", "--no-newline"])
    try:
        result = _run(argv)
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        try:
            result = _run(["xclip", "-selection", "clipboard", "-o"])
        except (OSError, subprocess.SubprocessError):
            return ""
    return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else ""


def grab_image() -> tuple:
    """(mime, bytes) for an image on the clipboard, or None.

    None is the ordinary answer — the clipboard usually holds text — and the
    caller treats it as "this was a normal paste"."""
    readers = {"darwin": _from_macos, "win32": _from_windows}
    reader = readers.get(sys.platform, _from_linux)
    try:
        # A copied file first, then an image on the clipboard itself, then a
        # path someone typed or copied as text.
        grabbed = _from_file_reference() or reader() or _from_text_path(clipboard_text())
    except Exception:
        return None      # a clipboard is never worth an exception
    if not grabbed:
        return None
    mime, data = grabbed
    if not data or len(data) > MAX_BYTES:
        return None
    return mime, data


def copy_text(text: str) -> bool:
    """Put `text` on the system clipboard. True if some tool took it.

    The counterpart to clipboard_text: a full-screen app owns the terminal's
    mouse, so the transcript can also be handed to the clipboard directly
    rather than only by dragging across what happens to be on screen."""
    candidates = {
        "darwin": [["pbcopy"]],
        "win32": [["clip"]],
    }.get(sys.platform, [["wl-copy"], ["xclip", "-selection", "clipboard"],
                          ["xsel", "--clipboard", "--input"]])
    payload = text.encode("utf-8")
    for argv in candidates:
        try:
            result = _run(argv, input=payload)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return True
    return False
