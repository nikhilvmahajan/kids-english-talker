#!/usr/bin/env python3
"""Talker voice service: Indic Parler-TTS speaker Priya, streamed, raised 5 semitones.

POST /speak   body: JSON {"text": ..., "feeling": "happy" | "curious" | "gentle"}
              answer: Ogg Opus audio, sent piece by piece as it is made; the
              answer ends when the connection closes (no Content-Length).
GET  /health  answer: JSON {"ready": true}

The model makes speech in 0.5 s pieces (ParlerTTSStreamer). One ffmpeg process
per answer takes the raw pieces as one continuous stream, raises the pitch
(asetrate + atempo, K-D10) and encodes Opus, so there are no clicks at the joins.
It never prints or stores the text. On 127.0.0.1 only; web.py starts and stops it.

Run (by web.py):  ../k_t3/.venv-parler/bin/python tts_service.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", 8767
PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "models" / "hf"
REPO = "ai4bharat/indic-parler-tts"
SHIFT_SEMITONES = 5
CHUNK_S = 0.5
MAX_TEXT = 600
CLEAN = "in a very clear, close recording with no background noise."
STYLES = {  # K-D10: style follows the feeling of the line
    "happy": "Priya's voice is cheerful, friendly and slightly high-pitched, speaking at a moderate pace with an expressive, happy tone, ",
    "curious": "Priya's voice is soft, friendly and curious, slightly high-pitched, speaking a little slowly, ",
    "gentle": "Priya's voice is gentle, calm and kind, speaking slowly in a soft, caring tone, ",
}


class Voice:
    """Holds the model; one answer is made at a time."""

    def __init__(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.torch = torch
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            REPO, cache_dir=CACHE, torch_dtype=torch.float16).to("cuda")
        self.tok = AutoTokenizer.from_pretrained(REPO, cache_dir=CACHE)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path, cache_dir=CACHE)
        self.sr = self.model.config.sampling_rate
        self.lock = threading.Lock()
        self.desc = {k: self.desc_tok(v + CLEAN, return_tensors="pt").to("cuda") for k, v in STYLES.items()}

    def ffmpeg(self) -> subprocess.Popen:
        ratio = 2 ** (SHIFT_SEMITONES / 12)
        # No input probing: by default ffmpeg reads up to 5 s of input before it starts,
        # which delayed the first audio by about 3 s in the K-T7 test.
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-probesize", "32", "-analyzeduration", "0", "-fflags", "+nobuffer",
               "-f", "f32le", "-ar", str(self.sr), "-ac", "1", "-i", "pipe:0",
               "-af", f"asetrate={self.sr * ratio:.0f},aresample=48000,atempo={1 / ratio:.5f}",
               "-c:a", "libopus", "-b:a", "32k", "-page_duration", "100000", "-flush_packets", "1",
               "-f", "ogg", "pipe:1"]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def speak(self, text: str, feeling: str, out) -> dict:
        """Write Ogg Opus to `out` as it is made. Returns timing numbers."""
        from parler_tts import ParlerTTSStreamer
        torch = self.torch
        d = self.desc.get(feeling, self.desc["happy"])
        p = self.tok(text, return_tensors="pt").to("cuda")
        steps = int(self.model.audio_encoder.config.frame_rate * CHUNK_S)
        streamer = ParlerTTSStreamer(self.model, device="cuda", play_steps=steps)
        errors, stats = [], {"audio_s": 0.0}
        t0 = time.perf_counter()

        def generate():
            try:
                torch.manual_seed(7)  # the same text gives the same voice
                with torch.no_grad():
                    self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                        prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                        streamer=streamer)
            except RuntimeError as e:
                errors.append(f"generate: {e}")
                streamer.end()

        enc = self.ffmpeg()

        def feed():
            try:
                for chunk in streamer:
                    if chunk.size:
                        if "first_piece_ms" not in stats:
                            stats["first_piece_ms"] = round((time.perf_counter() - t0) * 1000)
                        stats["audio_s"] += chunk.size / self.sr
                        enc.stdin.write(chunk.astype("float32").tobytes())
                        enc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:  # encoder gone (client left)
                errors.append(f"feed: {type(e).__name__}")
            finally:
                try:
                    enc.stdin.close()
                except (BrokenPipeError, ValueError):
                    pass

        with self.lock:
            threading.Thread(target=generate, daemon=True).start()
            feeder = threading.Thread(target=feed, daemon=True)
            feeder.start()
            fd = enc.stdout.fileno()
            try:
                while True:
                    data = os.read(fd, 4096)
                    if not data:
                        break
                    if "first_byte_ms" not in stats:
                        stats["first_byte_ms"] = round((time.perf_counter() - t0) * 1000)
                    out.write(data)
                    out.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                errors.append(f"client left: {type(e).__name__}")
                enc.kill()
            feeder.join(timeout=30)
            enc.wait(timeout=10)
            err = enc.stderr.read().decode(errors="replace").strip()
            if enc.returncode not in (0, -9) and err:
                errors.append(f"ffmpeg: {err[:200]}")
        stats["total_ms"] = round((time.perf_counter() - t0) * 1000)
        stats["errors"] = errors
        return stats


class Handler(BaseHTTPRequestHandler):
    voice: Voice = None

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
        if self.path != "/speak":
            self.reply(404, {"error": f"no such path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length)) if 0 < length < 8192 else None
        except (ValueError, json.JSONDecodeError):
            req = None
        if not isinstance(req, dict) or not isinstance(req.get("text"), str) or not req["text"].strip():
            self.reply(400, {"error": "body must be JSON with non-empty text"})
            return
        text = req["text"].strip()[:MAX_TEXT]
        feeling = req.get("feeling") if req.get("feeling") in STYLES else "happy"
        self.send_response(200)
        self.send_header("Content-Type", "audio/ogg")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        stats = self.voice.speak(text, feeling, self.wfile)
        self.log_message("spoke %.1f s of audio (%s): first piece %s ms, first byte %s ms, all %s ms%s",
                         stats["audio_s"], feeling, stats.get("first_piece_ms"), stats.get("first_byte_ms"),
                         stats["total_ms"], f", errors: {stats['errors']}" if stats["errors"] else "")


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        voice = Voice()
        voice.speak("Hello.", "happy", open(os.devnull, "wb"))  # warm-up
    except (OSError, RuntimeError, ImportError) as e:
        print(f"tts: cannot start: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1
    Handler.voice = voice
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"tts: cannot listen on {HOST}:{PORT}: {e}", file=sys.stderr, flush=True)
        return 1
    server.daemon_threads = True
    print(f"tts: ready on {HOST}:{PORT} (Priya, +{SHIFT_SEMITONES} semitones)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
