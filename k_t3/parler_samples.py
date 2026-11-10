#!/usr/bin/env python3
"""K-T3: voice samples with Indic Parler-TTS (AI4Bharat, Apache 2.0, gated).

Downloads the model once (needs HF_TOKEN in ../.env; the token is read into
this process only and never printed), then speaks Toffee's lines with each
voice description in VOICES, on the GPU. Writes Opus clips and parler.html to
samples/ (served to the phone on port 8765), and prints the time per line and
the real-time factor (RTF: time to make / length of the audio).

Run:  .venv-parler/bin/python parler_samples.py      (from kids_english/k_t3)
"""
import html
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CACHE = PROJECT / "models" / "hf"
OUT = HERE / "samples"
REPO = "ai4bharat/indic-parler-tts"
MIN_FREE_MIB, BIG_APP_MIB = 4000, 500
LINES = [
    "Kabir is taller than you! You said taller. Well done! What game do you like to play with Kabir?",
    "You went to your Nani's house yesterday! That sounds nice. Did you play games there?",
    "I did not hear. Can you say it again for me?",
    "I am sorry that happened. Please tell Aai or Baba, or another grown-up you trust, right now.",
]
CLEAN = "in a very clear, close recording with no background noise."
# Named Indian English speakers from the model card (English: 21 speakers; recommended: Thoma, Mary).
# Without a name, the model falls back to its generic, Western-sounding voices.
SPEAKERS = ["Mary", "Sneha", "Priya", "Kavya", "Riya", "Meera", "Thoma"]
STYLE = ("voice is cheerful, friendly and slightly high-pitched, speaking at a moderate pace "
         "with an expressive, happy tone, " + CLEAN)
VOICES = [(name, f"{name}'s {STYLE}") for name in SPEAKERS]


class SampleError(Exception):
    """A step failed; the message says which."""


def load_token() -> None:
    """Put HF_TOKEN from ../.env into this process's environment. Never print it."""
    env = PROJECT / ".env"
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "HF_TOKEN" and value.strip():
                os.environ["HF_TOKEN"] = value.strip()
                return
    except FileNotFoundError as e:
        raise SampleError(f"{env} not found; run load_creds.py first") from e
    raise SampleError(f"no HF_TOKEN in {env}")


def gpu_ok() -> None:
    try:
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip())
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.split()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        raise SampleError(f"GPU check failed: {type(e).__name__}") from e
    if free < MIN_FREE_MIB or any(int(a) >= BIG_APP_MIB for a in apps):
        raise SampleError(f"GPU busy ({free} MiB free). Not now (K-D6).")


def to_opus(pcm, sr: int, out: Path) -> None:
    import numpy as np
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name,
               "-c:a", "libopus", "-b:a", "32k", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise SampleError(f"ffmpeg failed for {out.name}: {e}") from e


def main() -> int:
    try:
        load_token()
        gpu_ok()
        import torch
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        try:
            t0 = time.perf_counter()
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                REPO, cache_dir=CACHE, torch_dtype=torch.float16).to("cuda")
            tok = AutoTokenizer.from_pretrained(REPO, cache_dir=CACHE)
            desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, cache_dir=CACHE)
        except GatedRepoError as e:
            raise SampleError("access refused: accept the model's terms on its Hugging Face page") from e
        except HfHubHTTPError as e:
            raise SampleError(f"download failed: HTTP {getattr(e.response, 'status_code', '?')}") from e
        print(f"model ready in {time.perf_counter() - t0:.1f} s; GPU memory in use: "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        sr = model.config.sampling_rate
        (OUT / "parler").mkdir(parents=True, exist_ok=True)
        page = []
        for label, desc in VOICES:
            page.append(f"<h2>{label}</h2><p><i>{html.escape(desc)}</i></p>")
            make = audio = 0.0
            d = desc_tok(desc, return_tensors="pt").to("cuda")
            for n, line in enumerate(LINES, 1):
                p = tok(line, return_tensors="pt").to("cuda")
                torch.manual_seed(n)
                t = time.perf_counter()
                with torch.no_grad():
                    gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                         prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
                make += time.perf_counter() - t
                pcm = gen.float().cpu().numpy().squeeze()
                if pcm.size == 0:
                    raise SampleError(f"empty audio for {label} line {n}")
                audio += pcm.size / sr
                name = f"parler/{label}_{n}.ogg"
                to_opus(pcm, sr, OUT / name)
                page.append(f"<p>{n}. {html.escape(line)}<br><audio controls preload=none src='{name}'></audio></p>")
            print(f"{label:8}: {make / len(LINES):.2f} s per line on average, RTF {make / audio:.2f}, "
                  f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        (OUT / "parler.html").write_text(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Parler samples</title><body style='font:16px system-ui;padding:16px'>"
            "<h1>Toffee voice samples (Indic Parler-TTS)</h1>" + "\n".join(page), encoding="utf-8")
    except SampleError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    print(f"done: open parler.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
