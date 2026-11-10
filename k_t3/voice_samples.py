#!/usr/bin/env python3
"""K-T3: make voice samples with Kokoro (ONNX, CPU) and time them.

Speaks four of Toffee's replies from the K-T4 test (synthetic text) with each
voice in VOICES, at normal pitch and raised by RAISE_SEMITONES. Writes Opus
clips and an index.html to samples/, to listen on the desktop or the phone.
Prints the time to make each clip and the real-time factor (RTF: time to
make / length of the audio; below 1 is faster than speech).

Run:  .venv/bin/python voice_samples.py      (from kids_english/k_t3)
Delete samples/ after the choice; this script makes them again.
"""
import html
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "models" / "kokoro"
OUT = HERE / "samples"
RAISE_SEMITONES = 3
LINES = [
    "Kabir is taller than you! You said taller. Well done! What game do you like to play with Kabir?",
    "You went to your Nani's house yesterday! That sounds nice. Did you play games there?",
    "I did not hear. Can you say it again for me?",
    "Oh, today is a holiday! That sounds fun. Do you like holidays?",
]
VOICES = [  # (voice, language): Kokoro's own names; a = American, b = British, f = female, m = male
    ("af_heart", "en-us"), ("af_bella", "en-us"), ("af_sky", "en-us"), ("af_nova", "en-us"),
    ("bf_lily", "en-gb"), ("bf_emma", "en-gb"), ("am_puck", "en-us"),
]


class SampleError(Exception):
    """A step failed; the message says which."""


def to_opus(pcm: np.ndarray, sr: int, out: Path, semitones: int) -> None:
    """Write 16-bit mono WAV to a temp file, then encode Opus with ffmpeg, raising pitch if asked."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        filters = []
        if semitones:
            ratio = 2 ** (semitones / 12)
            # Raise pitch, keep speed: play faster (asetrate), then slow back down (atempo).
            filters = ["-af", f"asetrate={sr * ratio:.0f},aresample={sr},atempo={1 / ratio:.5f}"]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name, *filters,
               "-c:a", "libopus", "-b:a", "32k", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise SampleError(f"ffmpeg failed for {out.name}: {e}") from e


def main() -> int:
    try:
        tts = Kokoro(str(MODEL_DIR / "kokoro-v1.0.onnx"), str(MODEL_DIR / "voices-v1.0.bin"))
    except (FileNotFoundError, OSError) as e:
        print(f"cannot load Kokoro: {e}", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    rows, page = [], []
    try:
        for voice, lang in VOICES:
            make, audio = 0.0, 0.0
            page.append(f"<h2>{voice}</h2>")
            for n, line in enumerate(LINES, 1):
                t0 = time.perf_counter()
                pcm, sr = tts.create(line, voice=voice, speed=1.0, lang=lang)
                make += time.perf_counter() - t0
                audio += len(pcm) / sr
                for semis in (0, RAISE_SEMITONES):
                    name = f"{voice}_{n}{'_high' if semis else ''}.ogg"
                    to_opus(pcm, sr, OUT / name, semis)
                page.append(f"<p>{n}. {html.escape(line)}<br>normal <audio controls preload=none src='{name.replace('_high', '')}'></audio>"
                            f" higher <audio controls preload=none src='{name}'></audio></p>")
            rows.append((voice, make / len(LINES), make / audio))
            print(f"{voice:9} {lang}: {make / len(LINES):.2f} s per line on average, RTF {make / audio:.2f}", flush=True)
    except SampleError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Voice samples</title><body style='font:16px system-ui;padding:16px'>"
        f"<h1>Toffee voice samples (Kokoro)</h1><p>Higher = raised {RAISE_SEMITONES} semitones.</p>"
        + "\n".join(page), encoding="utf-8")
    print(f"\n{len(rows) * len(LINES) * 2} clips in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
