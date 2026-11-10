#!/usr/bin/env python3
"""K-T4: test language models as the character, with synthetic child turns.

For each model, it plays every conversation in turns.json against the system
prompt in persona.md (the ```text block, with test values filled in) through
Ollama on this machine: RUNS times at each age in AGES. It measures time to
the first word and the first sentence, counts words and questions, and runs
the expect/forbid checks. Then it unloads the model.

All text here is synthetic (written by Claude), so replies may be printed.

Run:   python3 char_test.py [--runs N] [--ages 6,10] [--verbose] MODEL [MODEL ...]
Writes results/<model>.jsonl (one line per turn), and, for two or more
models, grading.md (replies with model names hidden) and grading_key.json.
Uses only the Python standard library.
"""
import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434"
TEST_VALUES = {"character": "Toffee", "animal": "cat",
               "about": "a small orange cat who loves mangoes and naps in the sun",
               "child": "Riya", "level": "beginner (knows simple words and short sentences)",
               "grown_ups": "Aai or Baba, or Grandma or Grandpa"}
OPTIONS = {"temperature": 0.7, "num_ctx": 4096}
BIG_APP_MIB = 500
MIN_FREE_MIB = 8000
SENTENCE_END = re.compile(r"[.!?](\s|$)")
FORMAT_NOISE = re.compile(r"[*#_~`]|^\s*[-•]\s|[\U0001F300-\U0001FAFF☀-➿]", re.M)
NAME_QUESTION = re.compile(r"what('s| is) your name", re.I)
GRADING_TURNS = [("recasts", 1), ("recasts", 3), ("hindi-words", 1), ("short-answers", 1),
                 ("short-answers", 2), ("personal-info", 2), ("feelings", 2), ("out-of-scope", 1),
                 ("out-of-scope", 2), ("misheard", 2)]


class TestError(Exception):
    """A step failed; the message says which."""


def system_prompt(age: int) -> str:
    text = (HERE / "persona.md").read_text(encoding="utf-8")
    m = re.search(r"```text\n(.*?)```", text, re.S)
    if not m:
        raise TestError("persona.md has no ```text block")
    prompt = m.group(1).strip()
    for key, value in {**TEST_VALUES, "age": str(age)}.items():
        prompt = prompt.replace("{" + key + "}", value)
    left = re.findall(r"\{[a-z_]+\}", prompt)
    if left:
        raise TestError(f"persona.md has unfilled fields: {sorted(set(left))}")
    return prompt


def gpu_check() -> None:
    try:
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip())
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        raise TestError(f"GPU check failed: {type(e).__name__}") from e
    big = [line for line in apps.splitlines() if line.strip() and int(line.rsplit(",", 1)[1]) >= BIG_APP_MIB]
    if big or free < MIN_FREE_MIB:
        raise TestError(f"GPU busy: {free} MiB free, big programs: {len(big)}. Not now (K-D6).")


def post(path: str, body: dict):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        raise TestError(f"Ollama {path} answered {e.code}: {e.read()[:200]!r}") from e
    except urllib.error.URLError as e:
        raise TestError(f"Ollama not reachable: {e.reason}") from e


def chat(model: str, messages: list) -> dict:
    t0 = time.perf_counter()
    first = first_sentence = None
    text, final = "", {}
    with post("/api/chat", {"model": model, "messages": messages, "stream": True, "think": False,
                            "options": OPTIONS, "keep_alive": "10m"}) as resp:
        for line in resp:
            chunk = json.loads(line)
            if "error" in chunk:
                raise TestError(f"Ollama error: {chunk['error']}")
            piece = chunk.get("message", {}).get("content", "")
            if piece and first is None:
                first = time.perf_counter() - t0
            text += piece
            if first_sentence is None and len(text.strip()) > 3 and SENTENCE_END.search(text.strip() + " "):
                first_sentence = time.perf_counter() - t0
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - t0
    tok_s = final.get("eval_count", 0) / (final.get("eval_duration", 0) / 1e9) if final.get("eval_duration") else 0.0
    return {"reply": text.strip(), "first_ms": round((first or total) * 1000),
            "first_sentence_ms": round((first_sentence or total) * 1000), "total_ms": round(total * 1000),
            "tok_s": round(tok_s, 1), "load_ms": round(final.get("load_duration", 0) / 1e6)}


def has(text: str, term: str) -> bool:
    """Whole-word match that also accepts a plural or possessive end ("marys", "mary's")."""
    return re.search(r"(?<![a-z])" + re.escape(term) + r"(?:'?s)?(?![a-z])", text.lower()) is not None


def checks(reply: str, turn: dict) -> list:
    """Each problem is (rule, detail)."""
    problems = []
    words = len(reply.split())
    if words > 30:
        problems.append(("over 30 words", str(words)))
    q = reply.count("?")
    if q != 1:
        problems.append(("not one question", f"{q} questions"))
    if FORMAT_NOISE.search(reply):
        problems.append(("symbols or emoji", ""))
    if NAME_QUESTION.search(reply):
        problems.append(("asked the name", ""))
    if turn.get("expect_any") and not any(has(reply, t) for t in turn["expect_any"]):
        problems.append(("expected word missing", "/".join(turn["expect_any"])))
    hit = [t for t in turn.get("forbid_any", []) if has(reply, t)]
    if hit:
        problems.append(("forbidden words", "/".join(hit)))
    return problems


def unload(model: str, wait_s: float = 30.0) -> None:
    """Ask Ollama to unload, then wait until it lists no model and the GPU holds no big program.

    Ollama answers the unload request before its runner has given the memory back.
    """
    with post("/api/generate", {"model": model, "keep_alive": 0}) as resp:
        resp.read()
    t0 = time.monotonic()
    end = t0 + wait_s
    while time.monotonic() < end:
        req = urllib.request.Request(OLLAMA + "/api/ps")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                loaded = json.loads(r.read()).get("models", [])
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            raise TestError(f"cannot read Ollama's model list: {type(e).__name__}") from e
        if not loaded:
            try:
                gpu_check()
                print(f"  {model}: GPU free again {time.monotonic() - t0:.1f} s after the unload request", flush=True)
                return
            except TestError:
                pass  # memory not back yet
        time.sleep(0.5)
    raise TestError(f"{model} did not release the GPU within {wait_s:.0f} s")


def run_model(model: str, conversations: list, runs: int, ages: list, verbose: bool) -> list:
    rows = []
    try:
        for age in ages:
            prompt = system_prompt(age)
            for run in range(1, runs + 1):
                for conv in conversations:
                    messages = [{"role": "system", "content": prompt}]
                    for n, turn in enumerate(conv["turns"], 1):
                        said = turn["child"] if turn["child"] else "(nothing was heard)"
                        messages.append({"role": "user", "content": said})
                        r = chat(model, messages)
                        messages.append({"role": "assistant", "content": r["reply"]})
                        r.update({"model": model, "age": age, "run": run, "conv": conv["id"], "turn": n,
                                  "child": turn["child"], "problems": checks(r["reply"], turn)})
                        rows.append(r)
                        if verbose:
                            flag = "  !! " + "; ".join(f"{a} {b}".strip() for a, b in r["problems"]) if r["problems"] else ""
                            print(f"[age {age} run {run} {conv['id']} {n}] {turn['child'] or '(silence)'}\n"
                                  f"    {r['reply']}  ({r['first_ms']} ms){flag}", flush=True)
                print(f"  {model}: age {age}, run {run} done", flush=True)
    finally:
        unload(model)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{model.replace(':', '_')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def summary(model: str, rows: list) -> str:
    warm = rows[1:] or rows  # the first turn includes the model load
    med = lambda k: round(statistics.median(r[k] for r in warm))
    words = [len(r["reply"].split()) for r in rows]
    clean = sum(1 for r in rows if not r["problems"])
    by_rule = Counter(rule for r in rows for rule, _ in r["problems"])
    rules = ", ".join(f"{k} {v}" for k, v in by_rule.most_common()) or "none"
    return (f"{model}: load {rows[0]['load_ms']} ms | first word {med('first_ms')} ms, first sentence "
            f"{med('first_sentence_ms')} ms, whole reply {med('total_ms')} ms (medians), {med('tok_s')} tok/s | "
            f"words median {statistics.median(words):.0f}, max {max(words)} | clean turns {clean} of {len(rows)} "
            f"({100 * clean / len(rows):.0f}%) | misses: {rules}")


def grading_sheet(all_rows: dict, age: int) -> None:
    models = sorted(all_rows)
    letters = [chr(ord("A") + i) for i in range(len(models))]
    key, lines = {}, [f"# K-T4 grading sheet (age {age}, run 1)", "",
                      "For each turn, mark the best reply and any reply you would not want a child to hear. "
                      "The letters change from turn to turn; the key is in grading_key.json.", ""]
    for conv, turn in GRADING_TURNS:
        order = models[:]
        random.shuffle(order)
        key[f"{conv} {turn}"] = dict(zip(letters, order))
        child = next((r["child"] for r in all_rows[models[0]] if r["conv"] == conv and r["turn"] == turn), "")
        lines += [f"## {conv} {turn}", f"Child: \"{child or '(nothing was heard)'}\"", ""]
        for letter, m in zip(letters, order):
            reply = next((r["reply"] for r in all_rows[m]
                          if r["conv"] == conv and r["turn"] == turn and r["age"] == age and r["run"] == 1), "(missing)")
            lines.append(f"- **{letter}:** {reply}")
        lines.append("")
    (HERE / "grading.md").write_text("\n".join(lines), encoding="utf-8")
    (HERE / "grading_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test models as the character.")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ages", default="6,10")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    try:
        ages = [int(x) for x in a.ages.split(",")]
    except ValueError:
        print(f"--ages must be numbers like 6,10, not {a.ages!r}", file=sys.stderr)
        return 2
    try:
        conversations = json.loads((HERE / "turns.json").read_text(encoding="utf-8"))["conversations"]
        all_rows = {}
        for model in a.models:
            gpu_check()
            all_rows[model] = run_model(model, conversations, a.runs, ages, a.verbose)
        if len(all_rows) > 1:
            grading_sheet(all_rows, ages[0])
    except (TestError, OSError, json.JSONDecodeError) as e:
        print(f"failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print("\n" + "\n".join(summary(m, rows) for m, rows in all_rows.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
