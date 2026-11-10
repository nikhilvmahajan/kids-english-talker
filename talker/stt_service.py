#!/usr/bin/env python3
"""Talker speech-to-text service: faster-whisper on the GPU, on 127.0.0.1 only.

POST /transcribe  body: the recorded audio (any format ffmpeg reads)
                  answer: JSON {"text", "audio_s", "stt_ms"}
GET  /health      answer: JSON {"ready": true}

It decodes the audio in memory and drops it. It never prints, logs or stores
the text (K-P6). web.py starts it at "Start" and stops it at "End".

Run (by web.py):  ../.venv/bin/python stt_service.py [model]   (default: small)
"""
import ctypes
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", 8766
PROJECT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024


def preload_cublas() -> None:
    """Load the pip-installed cuBLAS 12 so ctranslate2 finds it without LD_LIBRARY_PATH."""
    import nvidia.cublas
    lib = Path(list(nvidia.cublas.__path__)[0]) / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)


class Handler(BaseHTTPRequestHandler):
    model = None
    lock = threading.Lock()

    def reply(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"ready": True})
        else:
            self.reply(404, {"error": f"no such path: {self.path}"})

    def do_POST(self):
        if self.path != "/transcribe":
            self.reply(404, {"error": f"no such path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.reply(400, {"error": "Content-Length is not a number"})
            return
        if not 0 < length <= MAX_BYTES:
            self.reply(413 if length > MAX_BYTES else 400, {"error": f"bad audio size: {length} bytes"})
            return
        audio = self.rfile.read(length)
        if len(audio) != length:
            self.reply(400, {"error": f"upload ended early: {len(audio)} of {length} bytes"})
            return
        import av
        t0 = time.perf_counter()
        try:
            with self.lock:
                # vad_filter: the built-in voice detector (Silero VAD, MIT) drops the parts with no
                # speech, so Whisper does not invent text ("Thank you.") from silence or room noise.
                segments, info = self.model.transcribe(io.BytesIO(audio), language="en", beam_size=1,
                                                       vad_filter=True,
                                                       vad_parameters={"min_silence_duration_ms": 500})
                text = " ".join(s.text.strip() for s in segments).strip()
        except av.error.FFmpegError as e:
            self.reply(400, {"error": f"audio could not be decoded: {type(e).__name__}"})
            return
        except RuntimeError as e:  # ctranslate2, for example GPU out of memory
            self.reply(500, {"error": f"speech-to-text failed: {e}"})
            return
        stt_ms = (time.perf_counter() - t0) * 1000
        self.log_message("transcribed %.1f s of audio in %.0f ms, %d characters", info.duration, stt_ms, len(text))
        self.reply(200, {"text": text, "audio_s": round(info.duration, 2), "stt_ms": round(stt_ms, 1)})


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    try:
        preload_cublas()
        import numpy as np
        from faster_whisper import WhisperModel
        model = WhisperModel(name, device="cuda", compute_type="float16",
                             download_root=str(PROJECT / "models"), local_files_only=True)
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])  # warm-up
    except (OSError, RuntimeError, ImportError) as e:
        print(f"stt: cannot start: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1
    Handler.model = model
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"stt: cannot listen on {HOST}:{PORT}: {e}", file=sys.stderr, flush=True)
        return 1
    server.daemon_threads = True
    print(f"stt: ready on {HOST}:{PORT} with Whisper {name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
