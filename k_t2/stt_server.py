#!/usr/bin/env python3
"""K-T2 step 2: speech-to-text inside the phone round trip.

The K-T1 server (../k_t1/server.py), plus Whisper: each POST /turn is
transcribed, and the text goes back to the phone in the X-Transcript header.
The reply audio is still the fixed K-T1 clip.

Privacy (K-P6): the transcript is shown on the phone only. This server never
prints it, logs it or writes it to disk. results.jsonl gets numbers only.
The uploaded audio is decoded in memory and then dropped.

GPU: if a program holds 500 MiB or more of GPU memory (for example another
program's GPU job), or less than 3000 MiB is free, the models load on the CPU instead.
The models stay in memory until the server stops; stop it when not testing.

Run (from kids_english/k_t2):  ../.venv/bin/python stt_server.py
(`tailscale serve` already forwards HTTPS to port 8765, K-D4.)
"""
import io
import sys
import time
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "k_t1"))
sys.path.insert(0, str(HERE))
import server as base  # noqa: E402  (K-T1 server)
import stt_bench  # noqa: E402  (GPU check, cuBLAS loader)

MODEL_NAMES = ("small", "large-v3-turbo")
MODELS_DIR = HERE.parent / "models"

base.PAGE = HERE / "index.html"
base.RESULTS = HERE / "results.jsonl"
base.NUMBER_FIELDS |= {"stt_ms", "wer", "audio_s", "sentence_no", "ref_words"}
base.TEXT_FIELDS |= {"model", "mode", "mark", "probe"}


def load_models() -> dict:
    import numpy as np
    from faster_whisper import WhisperModel
    from huggingface_hub.errors import LocalEntryNotFoundError

    free, used, others = stt_bench.gpu_state()
    use_gpu = free >= stt_bench.MIN_FREE_VRAM_MIB and not others
    if use_gpu:
        stt_bench.preload_cublas()
        device, compute = "cuda", "float16"
    else:
        device, compute = "cpu", "int8"
        print(f"GPU not used: {free} MiB free, big GPU programs: {others or 'none'}", flush=True)
    models = {}
    for name in MODEL_NAMES:
        try:
            m = WhisperModel(name, device=device, compute_type=compute,
                             download_root=str(MODELS_DIR), local_files_only=True, cpu_threads=8)
        except LocalEntryNotFoundError:
            print(f"model {name} is not downloaded; skipped", flush=True)
            continue
        list(m.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])  # warm-up
        models[name] = m
    if device == "cuda":
        print(f"models on GPU; GPU memory used now {stt_bench.gpu_state()[1]} MiB (was {used})", flush=True)
    return models


class SttHandler(base.Handler):
    models: dict = {}

    def turn(self):
        start = time.perf_counter()
        try:
            length = base.body_length(self, base.MAX_UPLOAD_BYTES)
            audio = base.read_exact(self, length)
        except base.BadRequest as e:
            self.send_error_text(e.status, str(e))
            return
        name = self.headers.get("X-Model", MODEL_NAMES[0])
        model = self.models.get(name)
        if model is None:
            self.send_error_text(400, f"model {name!r} is not loaded; loaded: {sorted(self.models)}")
            return
        import av
        t0 = time.perf_counter()
        try:
            segments, info = model.transcribe(io.BytesIO(audio), language="en", beam_size=1)
            text = " ".join(s.text.strip() for s in segments).strip()
        except av.error.FFmpegError as e:
            self.send_error_text(400, f"the audio could not be decoded: {type(e).__name__}")
            return
        except RuntimeError as e:  # ctranslate2 errors, for example GPU out of memory
            self.send_error_text(500, f"speech-to-text failed: {e}")
            return
        stt_ms = (time.perf_counter() - t0) * 1000
        server_ms = (time.perf_counter() - start) * 1000
        # Numbers only: the transcript is never printed or stored here.
        self.log_message("turn: %d bytes, %s, %.1f s audio, stt %.0f ms, %d characters heard",
                         length, name, info.duration, stt_ms, len(text))
        self.send_bytes(200, self.reply_clip, base.REPLY_TYPE, {
            "X-Server-Ms": f"{server_ms:.1f}", "X-Bytes-Received": str(length),
            "X-Stt-Ms": f"{stt_ms:.1f}", "X-Audio-S": f"{info.duration:.2f}",
            "X-Model": name, "X-Transcript": urllib.parse.quote(text),
        })


def main() -> int:
    try:
        if not base.PAGE.is_file():
            raise base.StartupError(f"page not found: {base.PAGE}")
        SttHandler.reply_clip = base.make_reply_clip()
        SttHandler.models = load_models()
    except (base.StartupError, stt_bench.BenchError) as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 1
    if not SttHandler.models:
        print("cannot start: no Whisper model is downloaded", file=sys.stderr)
        return 1
    try:
        server = ThreadingHTTPServer((base.HOST, base.PORT), SttHandler)
    except OSError as e:
        print(f"cannot listen on {base.HOST}:{base.PORT}: {e}", file=sys.stderr)
        return 1
    server.daemon_threads = True
    print(f"K-T2 server on http://{base.HOST}:{base.PORT}, models: {', '.join(SttHandler.models)}", flush=True)
    print(f"results go to {base.RESULTS}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
