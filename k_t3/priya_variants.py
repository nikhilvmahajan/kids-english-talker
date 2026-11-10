#!/usr/bin/env python3
"""K-T3: Priya (Indic Parler-TTS) with a style per line, made cartoon-like by raising pitch.

The style follows the feeling of each line (happy, curious, gentle), as the
cat animation will. Each clip is then raised by SHIFTS semitones with ffmpeg:
asetrate raises pitch and voice resonances together (a smaller-sounding
speaker), and atempo puts the speed back.

Run:  .venv-parler/bin/python priya_variants.py      (from kids_english/k_t3)
Writes samples/priya/*.ogg and samples/priya.html.
"""
import html
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import parler_samples as ps

SHIFTS = [0, 3, 5, 7]
LINES = [  # (feeling, line)
    ("happy", ps.LINES[0]),
    ("happy", ps.LINES[1]),
    ("curious", ps.LINES[2]),
    ("gentle", ps.LINES[3]),
]
STYLES = {
    "happy": "Priya's voice is cheerful, friendly and slightly high-pitched, speaking at a moderate pace with an expressive, happy tone, ",
    "curious": "Priya's voice is soft, friendly and curious, slightly high-pitched, speaking a little slowly, ",
    "gentle": "Priya's voice is gentle, calm and kind, speaking slowly in a soft, caring tone, ",
}


def shifted_opus(pcm, sr: int, out: Path, semitones: int) -> None:
    import numpy as np
    ratio = 2 ** (semitones / 12)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        filt = ["-af", f"asetrate={sr * ratio:.0f},aresample={sr},atempo={1 / ratio:.5f}"] if semitones else []
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name, *filt,
               "-c:a", "libopus", "-b:a", "32k", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ps.SampleError(f"ffmpeg failed for {out.name}: {e}") from e


def main() -> int:
    try:
        ps.load_token()
        ps.gpu_ok()
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            ps.REPO, cache_dir=ps.CACHE, torch_dtype=torch.float16).to("cuda")
        tok = AutoTokenizer.from_pretrained(ps.REPO, cache_dir=ps.CACHE)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, cache_dir=ps.CACHE)
        sr = model.config.sampling_rate
        out = ps.OUT / "priya"
        out.mkdir(parents=True, exist_ok=True)
        page = []
        for n, (feeling, line) in enumerate(LINES, 1):
            desc = STYLES[feeling] + ps.CLEAN
            d = desc_tok(desc, return_tensors="pt").to("cuda")
            p = tok(line, return_tensors="pt").to("cuda")
            torch.manual_seed(n)
            t = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                     prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            pcm = gen.float().cpu().numpy().squeeze()
            if pcm.size == 0:
                raise ps.SampleError(f"empty audio for line {n}")
            print(f"line {n} ({feeling}): {time.perf_counter() - t:.2f} s for {pcm.size / sr:.1f} s of audio", flush=True)
            cells = []
            for semis in SHIFTS:
                name = f"priya/{n}_{feeling}_up{semis}.ogg"
                shifted_opus(pcm, sr, ps.OUT / name, semis)
                cells.append(f"{'normal' if semis == 0 else f'+{semis}'} <audio controls preload=none src='{name}'></audio>")
            page.append(f"<h3>{n}. {feeling}</h3><p>{html.escape(line)}</p><p>" + "<br>".join(cells) + "</p>")
        (ps.OUT / "priya.html").write_text(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Priya variants</title><body style='font:16px system-ui;padding:16px'>"
            "<h1>Priya: style per line, pitch raised</h1><p>+3, +5, +7 = semitones higher (smaller, more cartoon-like).</p>"
            + "\n".join(page), encoding="utf-8")
    except ps.SampleError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    print("done: open priya.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
