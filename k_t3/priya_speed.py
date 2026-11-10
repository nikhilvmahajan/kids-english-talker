#!/usr/bin/env python3
"""K-T6: how fast can Priya (Indic Parler-TTS) start speaking?

Measures, on the GPU, for Toffee's short first sentences:
  A. plain generation of the first sentence alone (no tricks);
  B. streaming: time until the first piece of audio (CHUNK_S long) is ready,
     and whether later pieces keep ahead of playback;
  C. static cache + torch.compile (Parler's documented speed-up), after warm-up.
Also times the +5 semitone ffmpeg step. Prints a table; writes nothing.

Run:  .venv-parler/bin/python priya_speed.py      (from kids_english/k_t3)
"""
import subprocess
import sys
import threading
import time

import numpy as np

import parler_samples as ps

FIRST_SENTENCES = ["Kabir is taller than you!", "You went to your Nani's house yesterday!",
                   "I did not hear.", "I am sorry that happened."]
FULL_LINE = ps.LINES[0]
DESC = ("Priya's voice is cheerful, friendly and slightly high-pitched, speaking at a moderate pace "
        "with an expressive, happy tone, " + ps.CLEAN)
CHUNK_S = 0.5
COMPILE_LIMIT_S = 600


def gen_args(model, desc_tok, tok, text):
    d = desc_tok(DESC, return_tensors="pt").to("cuda")
    p = tok(text, return_tensors="pt").to("cuda")
    return dict(input_ids=d.input_ids, attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)


def timed_generate(torch, model, args) -> tuple:
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        audio = model.generate(**args).float().cpu().numpy().squeeze()
    torch.cuda.synchronize()
    return time.perf_counter() - t, audio.size / model.config.sampling_rate


def streamed(torch, model, args, streamer_cls) -> tuple:
    """Return (seconds to first chunk, total seconds, audio seconds, worst lead in seconds)."""
    sr = model.config.sampling_rate
    steps = int(model.audio_encoder.config.frame_rate * CHUNK_S)
    streamer = streamer_cls(model, device="cuda", play_steps=steps)
    errors = []

    def run():
        try:
            with torch.no_grad():
                model.generate(**args, streamer=streamer)
        except RuntimeError as e:  # surfaced below, never silent
            errors.append(e)
            streamer.end()

    t0 = time.perf_counter()
    threading.Thread(target=run, daemon=True).start()
    first, audio_s, worst_lead = None, 0.0, float("inf")
    for chunk in streamer:
        now = time.perf_counter() - t0
        if chunk.size == 0:
            continue
        if first is None:
            first = now
        # Playback starts at `first`; this chunk must arrive before playback reaches it.
        worst_lead = min(worst_lead, (first + audio_s) - now)
        audio_s += chunk.size / sr
    if errors:
        raise ps.SampleError(f"streaming failed: {errors[0]}")
    return first, time.perf_counter() - t0, audio_s, worst_lead


def shift_time(sr: int) -> float:
    pcm = np.zeros(int(sr * 1.5), dtype=np.float32)
    t = time.perf_counter()
    ratio = 2 ** (5 / 12)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
           "-af", f"asetrate={sr * ratio:.0f},aresample={sr},atempo={1 / ratio:.5f}", "-c:a", "libopus",
           "-b:a", "32k", "-f", "ogg", "-"]
    subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, check=True)
    return time.perf_counter() - t


def main() -> int:
    try:
        ps.load_token()
        ps.gpu_ok()
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            ps.REPO, cache_dir=ps.CACHE, torch_dtype=torch.float16).to("cuda")
        tok = AutoTokenizer.from_pretrained(ps.REPO, cache_dir=ps.CACHE)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, cache_dir=ps.CACHE)
        timed_generate(torch, model, gen_args(model, desc_tok, tok, "Hello."))  # warm-up

        print("A. first sentence alone, plain:")
        for s in FIRST_SENTENCES:
            took, audio = timed_generate(torch, model, gen_args(model, desc_tok, tok, s))
            print(f"   {took:5.2f} s for {audio:4.1f} s of audio  | {s}")
        took, audio = timed_generate(torch, model, gen_args(model, desc_tok, tok, FULL_LINE))
        print(f"   whole line: {took:5.2f} s for {audio:4.1f} s of audio")

        print(f"B. streaming, {CHUNK_S} s pieces:")
        for s in [FULL_LINE] + FIRST_SENTENCES[:2]:
            first, total, audio, lead = streamed(torch, model, gen_args(model, desc_tok, tok, s), ParlerTTSStreamer)
            keeps = "keeps ahead" if lead >= 0 else f"falls behind by {-lead:.2f} s"
            print(f"   first audio after {first:4.2f} s, all {total:4.2f} s for {audio:4.1f} s of audio, {keeps} | {s[:40]}")

        print(f"   ffmpeg +5 semitones on 1.5 s of audio: {shift_time(model.config.sampling_rate) * 1000:.0f} ms")

        print("C. static cache + torch.compile:")
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
        t = time.perf_counter()
        try:
            for _ in range(2):  # compile happens during the first calls
                timed_generate(torch, model, gen_args(model, desc_tok, tok, FIRST_SENTENCES[0]))
                if time.perf_counter() - t > COMPILE_LIMIT_S:
                    raise ps.SampleError("compile warm-up took too long")
        except (RuntimeError, AssertionError, ps.SampleError) as e:
            print(f"   compile path failed after {time.perf_counter() - t:.0f} s: {type(e).__name__}: {str(e)[:200]}")
            return 0
        print(f"   warm-up (compile) took {time.perf_counter() - t:.0f} s")
        for s in FIRST_SENTENCES:
            took, audio = timed_generate(torch, model, gen_args(model, desc_tok, tok, s))
            print(f"   {took:5.2f} s for {audio:4.1f} s of audio  | {s}")
    except (ps.SampleError, subprocess.CalledProcessError) as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
