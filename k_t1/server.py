#!/usr/bin/env python3
"""K-T1 round-trip test: phone -> desktop -> phone.

What it does:
- GET  /      sends index.html (the test page).
- GET  /reply.ogg  sends the fixed clip, for the plain player test.
- GET  /stall  sends stall.html (network test; it reuses POST /turn).
- POST /turn  reads the recorded audio, counts its bytes and discards it,
              then answers with one fixed spoken clip. The audio is never
              written to disk and never decoded.
- POST /log   appends one line of timing numbers to results.jsonl.

What it writes: results.jsonl in this folder (numbers, audio format, browser
name). Delete it after the numbers are copied into kids_english.md.

Run:   python3 server.py                 (listens on 127.0.0.1 only)
Then:  sudo tailscale serve --bg 8765    (HTTPS for your own devices only)
Stop:  Ctrl+C here, then: sudo tailscale serve reset
"""
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
STALL_PAGE = HERE / "stall.html"
RESULTS = HERE / "results.jsonl"
HOST = "127.0.0.1"
PORT = 8765
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_LOG_BYTES = 4096
REPLY_TEXT = "Hi! I heard you. What is your favourite animal?"
REPLY_TYPE = "audio/ogg"  # Opus, 32 kbit/s

NUMBER_FIELDS = {
    "held_ms", "encode_ms", "upload_bytes", "request_ms", "server_ms",
    "download_ms", "reply_bytes", "decode_ms", "output_latency_ms", "total_ms",
    "mic_open_ms", "rec_bps", "seq",
}
TEXT_FIELDS = {"mime", "ua", "error", "mic_mode", "play_mode", "kind", "net"}
MAX_TEXT_CHARS = 200


class StartupError(Exception):
    """The server cannot start (no ffmpeg, clip failed, page missing)."""


def make_reply_clip() -> bytes:
    """Speak REPLY_TEXT with ffmpeg's built-in flite voice; return Ogg Opus bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "reply.ogg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"flite=text='{REPLY_TEXT}':voice=slt",
            "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise StartupError("ffmpeg is not installed or not on PATH") from e
        except subprocess.CalledProcessError as e:
            raise StartupError(f"ffmpeg failed to make the reply clip: {e.stderr.strip()}") from e
        data = out.read_bytes()
    if len(data) < 1000:  # far too small for 3 s of speech
        raise StartupError("ffmpeg made an empty reply clip")
    return data


class BadRequest(Exception):
    """The request is not valid. Carries the HTTP status to send."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def body_length(handler: BaseHTTPRequestHandler, limit: int) -> int:
    raw = handler.headers.get("Content-Length")
    if raw is None:
        raise BadRequest(411, "Content-Length header missing")
    try:
        length = int(raw)
    except ValueError as e:
        raise BadRequest(400, f"Content-Length is not a number: {raw!r}") from e
    if length <= 0:
        raise BadRequest(400, "empty body")
    if length > limit:
        raise BadRequest(413, f"body too large: {length} bytes, limit {limit}")
    return length


def read_exact(handler: BaseHTTPRequestHandler, length: int) -> bytes:
    data = handler.rfile.read(length)
    if len(data) != length:
        raise BadRequest(400, f"upload ended early: got {len(data)} of {length} bytes")
    return data


def clean_log_entry(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise BadRequest(400, "log body must be a JSON object")
    unknown = set(payload) - NUMBER_FIELDS - TEXT_FIELDS
    if unknown:
        raise BadRequest(400, f"unknown log fields: {sorted(unknown)}")
    entry = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, value in payload.items():
        if key in NUMBER_FIELDS:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BadRequest(400, f"field {key} must be a number")
            entry[key] = round(float(value), 1)
        else:
            if not isinstance(value, str):
                raise BadRequest(400, f"field {key} must be text")
            entry[key] = value[:MAX_TEXT_CHARS]
    return entry


class Handler(BaseHTTPRequestHandler):
    reply_clip = b""

    def send_bytes(self, status: int, body: bytes, content_type: str, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as e:
            self.log_message("client left before the reply was sent: %s", type(e).__name__)

    def send_error_text(self, status: int, message: str):
        self.log_message("error %d: %s", status, message)
        self.send_bytes(status, (message + "\n").encode(), "text/plain; charset=utf-8")

    def do_GET(self):
        if self.path == "/reply.ogg":
            self.send_bytes(200, self.reply_clip, REPLY_TYPE)
            return
        pages = {"/": PAGE, "/index.html": PAGE, "/stall": STALL_PAGE}
        if self.path not in pages:
            self.send_error_text(404, f"no such page: {self.path}")
            return
        try:
            page = pages[self.path].read_bytes()
        except OSError as e:
            self.send_error_text(500, f"cannot read {pages[self.path].name}: {e}")
            return
        self.send_bytes(200, page, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path == "/turn":
            self.turn()
        elif self.path == "/log":
            self.log()
        else:
            self.send_error_text(404, f"no such endpoint: {self.path}")

    def turn(self):
        start = time.perf_counter()
        try:
            length = body_length(self, MAX_UPLOAD_BYTES)
            read_exact(self, length)  # read and discard; never stored
        except BadRequest as e:
            self.send_error_text(e.status, str(e))
            return
        server_ms = (time.perf_counter() - start) * 1000
        self.log_message("turn: %d bytes of %s, %.1f ms",
                         length, self.headers.get("Content-Type", "?"), server_ms)
        self.send_bytes(200, self.reply_clip, REPLY_TYPE,
                        {"X-Server-Ms": f"{server_ms:.1f}", "X-Bytes-Received": str(length)})

    def log(self):
        try:
            length = body_length(self, MAX_LOG_BYTES)
            raw = read_exact(self, length)
            payload = json.loads(raw)
            entry = clean_log_entry(payload)
        except BadRequest as e:
            self.send_error_text(e.status, str(e))
            return
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.send_error_text(400, f"log body is not valid JSON: {e}")
            return
        try:
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            self.send_error_text(500, f"cannot write results.jsonl: {e}")
            return
        self.send_bytes(204, b"", "text/plain")


def main() -> int:
    try:
        if not PAGE.is_file():
            raise StartupError(f"page not found: {PAGE}")
        Handler.reply_clip = make_reply_clip()
    except StartupError as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"cannot listen on {HOST}:{PORT}: {e}", file=sys.stderr)
        return 1
    server.daemon_threads = True
    print(f"K-T1 server on http://{HOST}:{PORT} (reply clip {len(Handler.reply_clip)} bytes)", flush=True)
    print(f"results go to {RESULTS}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
