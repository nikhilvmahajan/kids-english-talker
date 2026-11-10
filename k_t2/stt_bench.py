#!/usr/bin/env python3
"""K-T2 step 1: speech-to-text speed on this machine, with synthetic speech.

Makes four short clips with ffmpeg's built-in flite voices (known text, no
real voices), then transcribes them with each Whisper model on the GPU and
on the CPU. Prints load time, time per clip, real-time factor, word error
rate (WER) and VRAM use.

The flite voices are robotic, so WER here says little about real children.
This test is for speed. Accuracy comes from the phone test with real voices.

Run:  ../.venv/bin/python stt_bench.py [model names]   (default: small large-v3-turbo)
Writes nothing to disk except temporary clips, deleted at the end.
"""
import ctypes
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT / "models"
MODELS = ["small", "large-v3-turbo"]
MIN_FREE_VRAM_MIB = 3000
BIG_APP_MIB = 500

CLIPS = [  # (flite voice, text). No commas: flite text sits inside an ffmpeg filter.
    ("slt", "My favourite animal is a cat. She is white and very fluffy."),
    ("kal", "Yesterday I went to the park with my brother and we played cricket."),
    ("awb", "I like to eat mangoes in the summer holidays."),
    ("rms", "Can you tell me a story about a big red dragon?"),
]


class BenchError(Exception):
    """A step failed; the message says which."""


def preload_cublas() -> None:
    """Load the pip-installed cuBLAS so ctranslate2 finds it (no LD_LIBRARY_PATH needed)."""
    import nvidia.cublas  # namespace package from nvidia-cublas-cu12
    lib = Path(list(nvidia.cublas.__path__)[0]) / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        try:
            ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise BenchError(f"cannot load {name} from {lib}: {e}") from e


def gpu_state() -> tuple:
    """(free MiB, used MiB, names of other programs that hold BIG_APP_MIB or more).

    Desktop programs (terminal, file manager) hold about 50 MiB for the screen.
    Only a real GPU load, such as another program's GPU job, makes this test give way.
    """
    try:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise BenchError(f"nvidia-smi failed: {type(e).__name__}") from e
    free, used = (int(x) for x in mem.strip().split(","))
    others = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        name, _, mib = line.rpartition(",")
        try:
            if int(mib) >= BIG_APP_MIB:
                others.append(f"{Path(name.strip()).name} ({int(mib)} MiB)")
        except ValueError as e:
            raise BenchError(f"unexpected nvidia-smi line: {line!r}") from e
    return free, used, others


def make_clips(tmp: Path) -> list:
    clips = []
    for n, (voice, text) in enumerate(CLIPS):
        out = tmp / f"clip{n}.wav"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", f"flite=text='{text}':voice={voice}", "-ar", "16000", "-ac", "1", str(out)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise BenchError(f"ffmpeg could not make clip {n}: {e}") from e
        clips.append((out, text))
    return clips


def words(text: str) -> list:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        raise BenchError("empty reference text")
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def clip_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def run(model_name: str, device: str, clips: list) -> dict:
    from faster_whisper import WhisperModel
    compute = "float16" if device == "cuda" else "int8"
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute,
                         download_root=str(MODELS_DIR), local_files_only=True, cpu_threads=8)
    load_s = time.perf_counter() - t0
    # Warm-up run, not timed: the first call sets up kernels and buffers.
    list(model.transcribe(str(clips[0][0]), language="en", beam_size=1)[0])
    times, wers, audio_s, misses = [], [], [], []
    for path, ref in clips:
        t = time.perf_counter()
        segments, _ = model.transcribe(str(path), language="en", beam_size=1)
        hyp = " ".join(s.text.strip() for s in segments)
        times.append(time.perf_counter() - t)
        audio_s.append(clip_seconds(path))
        e = wer(ref, hyp)
        wers.append(e)
        if e > 0:
            misses.append(f"    ref: {ref}\n    got: {hyp}")
    vram_used = gpu_state()[1] if device == "cuda" else None
    del model
    return {"load_s": load_s, "clip_s": sum(times) / len(times),
            "rtf": sum(times) / sum(audio_s), "wer": sum(wers) / len(wers),
            "vram_used": vram_used, "misses": misses}


def main() -> int:
    try:
        free, used, others = gpu_state()
        use_gpu = free >= MIN_FREE_VRAM_MIB and not others
        if not use_gpu:
            print(f"GPU skipped: {free} MiB free, other GPU programs: {others or 'none'}")
        else:
            preload_cublas()
            print(f"GPU free before: {free} MiB (used {used} MiB)")
        with tempfile.TemporaryDirectory() as tmp:
            clips = make_clips(Path(tmp))
            print(f"{len(clips)} clips, {sum(clip_seconds(p) for p, _ in clips):.1f} s of audio in all\n")
            print(f"{'model':16} {'device':6} {'load s':>7} {'s/clip':>7} {'RTF':>6} {'WER':>6} {'VRAM used MiB':>14}")
            for name in sys.argv[1:] or MODELS:
                for device in (["cuda"] if use_gpu else []) + ["cpu"]:
                    r = run(name, device, clips)
                    vram = "" if r["vram_used"] is None else str(r["vram_used"])
                    print(f"{name:16} {device:6} {r['load_s']:7.1f} {r['clip_s']:7.2f} {r['rtf']:6.2f} {r['wer']:6.0%} {vram:>14}")
                    for m in r["misses"]:
                        print(m)
    except BenchError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
