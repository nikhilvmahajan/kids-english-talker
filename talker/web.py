#!/usr/bin/env python3
"""Toffee talker: the always-on web process (K-T7).

Small, standard library only, no GPU. `tailscale serve` forwards HTTPS from the
user's own devices to 127.0.0.1:8765 (K-D4).

  GET  /             the page (static/index.html); also the app files in STATIC (manifest, icons)
  GET  /status       session state and the current child's name
  GET  /profiles     the children's names and stars
  GET  /profile?name=  one child's stars, counts and all stickers (earned or not yet)
  POST /delete       {"name"}  delete a child's profile (not while that child is playing)
  POST /start        {"name", "age", "level", "grown_ups"} (age and level for a new child)
                     check the GPU (K-D6), start the services, load Gemma; Toffee greets
  POST /turn         body: recorded audio. Speech-to-text, Gemma, code checks
                     (K-D8, one question), feeling. Answers JSON at once; the voice
                     makes one file per sentence for GET /audio/<id>/<n>.
  GET  /audio/<id>/<n>  sentence n of the reply as a complete Ogg Opus file
  POST /end          stop both services, unload Gemma, wait until the GPU is free (K-D5)
  POST /log          timing numbers from the page

Profiles (K-D12): profiles.json (mode 600; previous version in profiles.json.bak; tests use
TALKER_PROFILES=<other file>) keeps each child's name, age, level, family words,
stars, counts, days played, stickers and the next topic. Never audio, never what the child said.

Privacy (K-P6): no audio and no words are stored or printed. log.jsonl gets numbers
and the names of the checks that acted, never text. The session's talk history is
kept in memory only and dropped at "End".

Run:  python3 web.py
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HOST, PORT = "127.0.0.1", 8765
STT, TTS, OLLAMA = "http://127.0.0.1:8766", "http://127.0.0.1:8767", "http://127.0.0.1:11434"
PY_STT = PROJECT / ".venv" / "bin" / "python"
PY_TTS = PROJECT / "k_t3" / ".venv-parler" / "bin" / "python"
RUN, LOG, PAGE = HERE / "run", HERE / "log.jsonl", HERE / "static" / "index.html"
# Files for the home-screen app (install or "Add to Home screen"); only these are served.
STATIC = {"/manifest.webmanifest": "application/manifest+json", "/icon.svg": "image/svg+xml",
          "/icon-192.png": "image/png", "/icon-512.png": "image/png"}
MAX_UPLOAD = 5 * 1024 * 1024
BIG_APP_MIB, NEED_FREE_MIB = 500, 9000
HISTORY_MESSAGES = 12
START_TIMEOUT_S, RELEASE_TIMEOUT_S = 150, 30
LLM_OPTIONS = {"temperature": 0.7, "num_ctx": 4096}
FALLBACK = "That sounds nice! What do you like to do for fun?"
LOG_NUMBERS = {"total_ms", "request_ms", "audio_wait_ms", "held_ms", "upload_bytes", "server_ms"}

# Profiles and rewards (K-D12, K-D13)
# Tests set TALKER_PROFILES to a separate file, so they can never touch the real profiles.
PROFILES = Path(os.environ.get("TALKER_PROFILES", str(HERE / "profiles.json")))
MAX_PROFILES = 8
DEFAULT_GROWN_UPS = "Aai or Baba, or Grandma or Grandpa"
LEVELS = {"starting": "just starting (knows a few English words)",
          "beginner": "beginner (knows simple words and short sentences)",
          "growing": "growing (can talk in short sentences)"}
NEW_DAY_STARS, STARS_PER_CELEBRATION = 3, 10
TOPICS = ["your favourite game", "your best friend", "your favourite food", "a festival you like",
          "your favourite cartoon", "an animal you like", "something fun you did this week", "your favourite colour"]
STICKERS = [  # (id, title, test on the updated profile and this turn); all reward effort, never correctness
    ("first_chat", "First Chat", lambda p, t: p["turns"] >= 1),
    ("chatterbox", "Chatterbox", lambda p, t: p["turns"] >= 10),
    ("big_sentence", "Big Sentence", lambda p, t: t["words"] >= 8),
    ("word_explorer", "Word Explorer", lambda p, t: p["words"] >= 50),
    ("word_star", "Word Star", lambda p, t: p["words"] >= 200),
    ("brave_talker", "Brave Talker", lambda p, t: t["brave"]),
    ("back_again", "Back Again", lambda p, t: len(p["days"]) >= 3),
]
BYE = ("Yawn! I am getting sleepy, {name}. That is enough talking for today! I will miss you. "
       "Come back tomorrow and tell me about {topic}. Bye bye!")

# Code checks (K-P2, K-D8)
PERSONAL_CUES = {"school", "live", "lives", "address", "street", "road", "house", "flat", "building",
                 "society", "colony", "city", "town", "village", "phone", "number"}
PLACE_CUES = {"in", "at", "from", "near"}
COMMON = set("""a an and the my your our his her their is are was were be been am i you he she it we they me
him them to of in on at for from with by near big small very this that these those there here what who where
when how go went goes going live lives living name named called but so because also too st sri""".split())
LANDMARKS_OK = set("""temple park market garden house home shop river beach zoo mosque church gurudwara road
street lake hill farm mall station school playground ground field""".split())
BAD_WORDS = {"damn", "hell", "shit", "fuck", "bitch", "crap", "stupid", "idiot", "bastard", "dumb"}
NAME_QUESTION = re.compile(r"what('s| is) your name", re.I)
# Split after . ! ? but not after a title such as "St." in "St. Mary's School".
SENTENCE = re.compile(r"(?<!\bSt\.)(?<!\bMr\.)(?<!\bDr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<=[.!?])\s+")
NOISE = re.compile(r"[*_#~`]|[\U0001F300-\U0001FAFF☀-➿]")


class TalkerError(Exception):
    """A request cannot be served. Carries the HTTP status and a message for the page."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_line(entry: dict) -> None:
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": now_iso(), **entry}) + "\n")
    except OSError as e:
        print(f"web: cannot write {LOG}: {e}", file=sys.stderr, flush=True)


def http_json(url: str, body=None, timeout: float = 60, raw: bytes = None, ctype: str = "application/json") -> dict:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode(errors="replace")
        raise TalkerError(502, f"{url.rsplit('/', 1)[-1]} answered {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise TalkerError(502, f"{url} not reachable: {getattr(e, 'reason', e)}") from e
    except json.JSONDecodeError as e:
        raise TalkerError(502, f"{url} gave no valid JSON") from e


def gpu_state() -> tuple:
    """(free MiB, used MiB, number of programs holding BIG_APP_MIB or more)."""
    try:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.split()
        free, used = (int(x) for x in mem.strip().split(","))
        return free, used, sum(1 for a in apps if int(a) >= BIG_APP_MIB)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        raise TalkerError(500, f"GPU check failed: {type(e).__name__}") from e


def norm(word: str) -> str:
    """Same form for "Mary's", "Marys" and "Mary": no apostrophe, no final s on longer words."""
    word = word.replace("'", "")
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def words_of(text: str) -> list:
    return [norm(w) for w in re.findall(r"[a-z0-9']+", text.lower())]


def personal_words(heard: str, child_name: str) -> set:
    """Words from the child's turn that may name their school, city, town or street (K-D8)."""
    toks = words_of(heard)
    if not {norm(c) for c in PERSONAL_CUES} & set(toks):
        return set()
    found = set()
    for i, t in enumerate(toks):
        if t in PLACE_CUES:  # in, at, from, near: none ends in s, so norm() leaves them
            found.update(toks[i + 1:i + 4])
        elif t == "school":
            found.update(toks[max(0, i - 3):i] + toks[i + 1:i + 4])
    found.update(t for t in toks if re.fullmatch(r"\d{3,}", t))
    skip = {norm(w) for w in COMMON | LANDMARKS_OK} | {norm(child_name.lower())}
    return {w for w in found if len(w) >= 3 and w not in skip}


def check_reply(reply: str, heard: str, child_name: str) -> tuple:
    """Return (text to speak, names of the checks that acted). Never logs the text."""
    acted = []
    text = NOISE.sub("", reply).strip()
    if text != reply.strip():
        acted.append("symbols removed")
    if BAD_WORDS & set(words_of(text)):
        return FALLBACK, acted + ["bad word: whole reply replaced"]
    personal = personal_words(heard, child_name)
    kept = []
    for s in SENTENCE.split(text):
        sw = set(words_of(s))
        if personal & sw or re.search(r"\d{3,}", s):
            acted.append("personal detail sentence dropped")
        elif NAME_QUESTION.search(s):
            acted.append("name question dropped")
        else:
            kept.append(s)
    text = " ".join(kept).strip()
    q = text.find("?")
    if q != -1 and q < len(text) - 1:
        text = text[:q + 1]
        acted.append("cut after first question")
    if not re.search(r"[A-Za-z]", text):
        return FALLBACK, acted + ["nothing left: fallback"]
    return text, acted


def pick_feeling(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("sorry", "grown-up", "grown up", "trust")):
        return "gentle"
    if any(k in low for k in ("did not hear", "didn't hear", "say it again", "say that again", "not sure what")):
        return "curious"
    return "happy"


class SentenceAudio:
    """The reply as small, complete Ogg Opus files, one per sentence, made in order.

    Firefox for Android waited for the whole streamed answer before it played
    (K-T7 phone test: 2.9-5.6 s from text to sound). Complete small files play at once,
    and sentence 2 is made while sentence 1 plays.
    """

    def __init__(self, parts: list):
        self.parts = parts
        self.data = [None] * len(parts)
        self.errors = [None] * len(parts)
        self.cond = threading.Condition()

    def fill(self, feeling: str) -> None:
        for n, text in enumerate(self.parts):
            data, error = None, None
            req = urllib.request.Request(TTS + "/speak", data=json.dumps({"text": text, "feeling": feeling}).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if not data:
                    error = "the voice gave no audio"
            except (urllib.error.URLError, OSError) as e:
                error = f"voice service: {getattr(e, 'reason', e)}"
            if error:
                print(f"web: sentence {n + 1}: {error}", file=sys.stderr, flush=True)
            with self.cond:
                self.data[n], self.errors[n] = data, error
                self.cond.notify_all()

    def get(self, n: int, timeout: float = 60) -> bytes:
        if not 0 <= n < len(self.parts):
            raise TalkerError(404, f"no sentence {n}")
        with self.cond:
            if not self.cond.wait_for(lambda: self.data[n] is not None or self.errors[n] is not None, timeout):
                raise TalkerError(504, "the voice took too long")
            if self.errors[n]:
                raise TalkerError(502, self.errors[n])
            return self.data[n]


def split_parts(text: str) -> list:
    return [p.strip() for p in SENTENCE.split(text) if p.strip()]


STRUGGLE = re.compile(r"\b(what|huh|again|repeat|pardon|(do not|don'?t|did not|didn'?t|not) understand)\b", re.I)


def needs_words(heard: str, empty_streak: int) -> bool:
    """Keep Toffee's words on screen when the child seems to struggle (user idea, 2026-11-11)."""
    short_struggle = bool(heard) and len(heard.split()) <= 6 and STRUGGLE.search(heard) is not None
    return short_struggle or empty_streak >= 2


class ProfileError(Exception):
    """profiles.json cannot be read; the web process stops rather than overwrite it."""


def clean_name(name) -> str:
    if not isinstance(name, str):
        raise TalkerError(400, "Please type the child's name.")
    name = " ".join(name.split())
    if not 1 <= len(name) <= 20 or not any(c.isalpha() for c in name) \
            or not all(c.isalpha() or c in " -'" for c in name):
        raise TalkerError(400, "Please type the name with letters only (up to 20).")
    return name


class Profiles:
    """The children's profiles, on this machine only (K-D12). No audio, no words the child said."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError) as e:
            raise ProfileError(f"cannot read {path}: {e}") from e
        if not isinstance(self.data, dict):
            raise ProfileError(f"{path} is not a JSON object")

    def save(self) -> None:
        """Write to a temp file, keep the previous version as <name>.bak, then swap in the new one."""
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        if self.path.exists():
            shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
        os.replace(tmp, self.path)

    def listing(self) -> list:
        return [{"name": p["name"], "stars": p["stars"]}
                for p in sorted(self.data.values(), key=lambda p: p["name"].lower())]

    def find(self, name) -> dict:
        p = self.data.get(clean_name(name).lower())
        if p is None:
            raise TalkerError(404, "No child with that name.")
        return p

    def details(self, name) -> dict:
        p = self.find(name)
        return {"name": p["name"], "age": p["age"], "level": p["level"], "stars": p["stars"],
                "words": p["words"], "turns": p["turns"], "sessions": p["sessions"], "days_played": len(p["days"]),
                "stickers": [{"title": title, "earned": sid in p["stickers"]} for sid, title, _ in STICKERS]}

    def delete(self, name) -> None:
        p = self.find(name)
        del self.data[p["name"].lower()]
        self.save()

    def get_or_create(self, req: dict) -> tuple:
        """Return (profile, created). New profiles need age and level."""
        name = clean_name(req.get("name"))
        p = self.data.get(name.lower())
        created = p is None
        if created:
            if len(self.data) >= MAX_PROFILES:
                raise TalkerError(400, f"Up to {MAX_PROFILES} children for now.")
            age, level = req.get("age"), req.get("level")
            if not isinstance(age, int) or isinstance(age, bool) or not 4 <= age <= 12:
                raise TalkerError(400, "Please choose the child's age.")
            if level not in LEVELS:
                raise TalkerError(400, "Please choose an English level.")
            p = {"name": name, "age": age, "level": level, "grown_ups": DEFAULT_GROWN_UPS, "stars": 0,
                 "turns": 0, "words": 0, "sessions": 0, "days": [], "stickers": [], "next_topic": "",
                 "topics_used": [], "created": now_iso()}
        grown = req.get("grown_ups")
        if isinstance(grown, str) and grown.strip():
            grown = " ".join(grown.split())
            if len(grown) > 60 or not all(c.isalpha() or c in " ,-'" for c in grown):
                raise TalkerError(400, "Family words: letters, spaces and commas only (up to 60).")
            p["grown_ups"] = grown
        if created:
            self.data[name.lower()] = p
        return p, created


def pick_topic(p: dict) -> str:
    recent = p["topics_used"][-4:]
    topic = next((t for t in TOPICS if t not in recent), TOPICS[0])
    p["topics_used"] = (p["topics_used"] + [topic])[-8:]
    return topic


class Session:
    def __init__(self, config: dict, persona: str, profiles: Profiles):
        self.config, self.persona, self.profiles = config, persona, profiles
        self.lock, self.turn_lock = threading.Lock(), threading.Lock()
        self.state, self.procs, self.history = "off", {}, []
        self.profile, self.system, self.last_turn, self.started = None, "", 0.0, 0.0
        self.audio, self.empty_streak, self.prev_support, self.today = {}, 0, False, {}
        self.bye_topic = ""  # once the goodbye has named a topic, it stays (what the child heard)

    def speak(self, text: str, feeling: str) -> dict:
        """Start the voice on `text`; return {"id", "parts"} for GET /audio/<id>/<n>."""
        parts = split_parts(text)
        turn_id = uuid.uuid4().hex[:12]
        audio = SentenceAudio(parts)
        self.audio = {k: v for k, v in list(self.audio.items())[-2:]}
        self.audio[turn_id] = audio
        threading.Thread(target=audio.fill, args=(feeling,), daemon=True).start()
        return {"id": turn_id, "parts": parts}

    def public_profile(self) -> dict:
        p = self.profile
        titles = {sid: title for sid, title, _ in STICKERS}
        return {"name": p["name"], "stars": p["stars"], "stickers": [titles[s] for s in p["stickers"] if s in titles]}

    def greeting(self, created: bool) -> str:
        p, c = self.profile, self.config
        if created:
            return f"Hi {p['name']}! I am {c['character']}, {c['about']}. What is your favourite animal?"
        if p["next_topic"]:
            topic, p["next_topic"] = p["next_topic"], ""
            return f"Hi {p['name']}! I missed you! Last time, I asked about {topic}. Can you tell me about {topic}?"
        return f"Hi {p['name']}! Welcome back! What did you do today?"

    # --- start and end -------------------------------------------------------------
    def start(self, req: dict) -> dict:
        with self.lock:
            if self.state == "on":
                if self.profile and clean_name(req.get("name")).lower() == self.profile["name"].lower():
                    return {"state": "on", "note": "already running", "profile": self.public_profile()}
                raise TalkerError(409, "Toffee is talking with another child. Press End first.")
            profile, created = self.profiles.get_or_create(req)  # checks the form before any GPU work
            free, _, big = gpu_state()
            if big or free < NEED_FREE_MIB:
                raise TalkerError(409, "Not now: the computer is busy with other work. Please try later.")
            t0 = time.perf_counter()
            self.state = "starting"
            try:
                RUN.mkdir(exist_ok=True)
                env = {"HF_HUB_OFFLINE": "1", "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home())}
                for name, py, script, args in (("stt", PY_STT, "stt_service.py", [self.config["whisper_model"]]),
                                               ("tts", PY_TTS, "tts_service.py", [])):
                    out = (RUN / f"{name}.log").open("a", encoding="utf-8")
                    self.procs[name] = subprocess.Popen([str(py), str(HERE / script), *args], cwd=HERE, env=env,
                                                        stdout=out, stderr=subprocess.STDOUT)
                http_json(OLLAMA + "/api/generate", {"model": self.config["llm_model"], "keep_alive": "30m"}, timeout=120)
                for name, url in (("stt", STT), ("tts", TTS)):
                    self.wait_ready(name, url, t0)
            except (TalkerError, OSError) as e:
                self.stop_all()
                self.state = "off"
                raise TalkerError(500, f"could not start: {e}") from e
            self.profile = profile
            fields = {**{k: str(v) for k, v in self.config.items() if isinstance(v, (str, int))},
                      **{"child": profile["name"], "age": str(profile["age"]),
                         "level": LEVELS[profile["level"]], "grown_ups": profile["grown_ups"]}}
            self.system = re.sub(r"\{([a-z_]+)\}", lambda m: fields.get(m.group(1), m.group(0)), self.persona)
            bonus, today = 0, date.today().isoformat()
            if today not in profile["days"]:
                profile["days"].append(today)
                if len(profile["days"]) > 1:  # coming back on a new day
                    bonus = NEW_DAY_STARS
                    profile["stars"] += bonus
            self.today = {"turns": 0, "words": 0, "stars": bonus, "stickers": []}
            greeting = self.greeting(created)
            self.history = [{"role": "assistant", "content": greeting}]
            self.profiles.save()
            greet = self.speak(greeting, "happy")
            self.last_turn = self.started = time.monotonic()
            self.state, self.empty_streak, self.prev_support, self.bye_topic = "on", 0, False, ""
            start_ms = round((time.perf_counter() - t0) * 1000)
            log_line({"kind": "start", "start_ms": start_ms, "new_profile": created, "day_bonus": bonus})
            return {"state": "on", "start_ms": start_ms, "profile": self.public_profile(), "greet": greet,
                    "stars_added": bonus}

    def wait_ready(self, name: str, url: str, t0: float) -> None:
        while time.perf_counter() - t0 < START_TIMEOUT_S:
            proc = self.procs[name]
            if proc.poll() is not None:
                raise TalkerError(500, f"{name} service stopped (exit {proc.returncode}); see talker/run/{name}.log")
            try:
                http_json(url + "/health", timeout=2)
                return
            except TalkerError:
                time.sleep(0.5)
        raise TalkerError(504, f"{name} service not ready after {START_TIMEOUT_S} s")

    def stop_all(self) -> None:
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        self.procs = {}
        try:
            http_json(OLLAMA + "/api/generate", {"model": self.config["llm_model"], "keep_alive": 0}, timeout=30)
        except TalkerError as e:
            print(f"web: Gemma unload: {e}", file=sys.stderr, flush=True)

    def end(self, reason: str) -> dict:
        with self.lock:
            if self.state == "off":
                return {"state": "off", "note": "was not running"}
            self.state = "stopping"
            t0 = time.perf_counter()
            summary = None
            if self.profile:
                p = self.profile
                p["sessions"] += 1
                if not p["next_topic"]:
                    p["next_topic"] = pick_topic(p)
                try:
                    self.profiles.save()
                except OSError as e:
                    print(f"web: cannot save profiles: {e}", file=sys.stderr, flush=True)
                summary = {"name": p["name"], **self.today, "stars_total": p["stars"], "next_topic": p["next_topic"]}
            self.stop_all()
            self.history, self.audio, self.profile, self.system = [], {}, None, ""
            freed = False
            while time.perf_counter() - t0 < RELEASE_TIMEOUT_S:
                try:
                    loaded = http_json(OLLAMA + "/api/ps", timeout=5).get("models", [])
                except TalkerError:
                    loaded = ["?"]
                if not loaded and gpu_state()[2] == 0:
                    freed = True
                    break
                time.sleep(0.5)
            self.state = "off"
            used = gpu_state()[1]
            log_line({"kind": "end", "reason": reason, "end_ms": round((time.perf_counter() - t0) * 1000),
                      "gpu_used_mib": used, "gpu_freed": freed})
            return {"state": "off", "gpu_used_mib": used, "gpu_freed": freed, "summary": summary}

    def idle_watch(self) -> None:
        while True:
            time.sleep(20)
            if self.state == "on" and time.monotonic() - self.last_turn > 60 * self.config["idle_minutes"]:
                self.end("idle")

    # --- one turn ------------------------------------------------------------------
    def turn(self, audio: bytes, ctype: str) -> dict:
        if self.state != "on":
            raise TalkerError(409, "Press Start first.")
        with self.turn_lock:
            p = self.profile
            t0 = time.perf_counter()
            stt = http_json(STT + "/transcribe", raw=audio, ctype=ctype or "application/octet-stream", timeout=30)
            heard = stt.get("text", "").strip()
            said = heard if heard else "(nothing was heard)"
            words = len(heard.split())
            t1 = time.perf_counter()
            time_up = time.monotonic() - self.started >= 60 * self.config["session_minutes"]
            if time_up:  # K-D13: a playful goodbye, then the page ends the session
                if not self.bye_topic:
                    self.bye_topic = pick_topic(p)
                p["next_topic"] = self.bye_topic
                raw, acted, feeling = "", ["time up: goodbye"], "gentle"
                reply = BYE.format(name=p["name"], topic=p["next_topic"])
            else:
                messages = [{"role": "system", "content": self.system}] + self.history[-HISTORY_MESSAGES:] + \
                           [{"role": "user", "content": said}]
                out = http_json(OLLAMA + "/api/chat", {"model": self.config["llm_model"], "messages": messages,
                                                       "stream": False, "think": False, "keep_alive": "30m",
                                                       "options": LLM_OPTIONS}, timeout=60)
                raw = out.get("message", {}).get("content", "")
                reply, acted = check_reply(raw, heard, p["name"])
                feeling = pick_feeling(reply)
            t2 = time.perf_counter()
            self.history += [{"role": "user", "content": said}, {"role": "assistant", "content": reply}]
            self.empty_streak = 0 if heard else self.empty_streak + 1
            support = needs_words(heard, self.empty_streak)

            # Stars and stickers reward effort, never correctness (K-D12).
            brave = self.prev_support and words >= 3
            self.prev_support = support
            added = (1 if words else 0) + (1 if words >= 5 else 0)
            before = p["stars"]
            p["stars"] += added
            p["turns"] += 1 if words else 0
            p["words"] += words
            self.today["turns"] += 1 if words else 0
            self.today["words"] += words
            self.today["stars"] += added
            new = [(sid, title) for sid, title, test in STICKERS
                   if sid not in p["stickers"] and test(p, {"words": words, "brave": brave})]
            p["stickers"] += [sid for sid, _ in new]
            self.today["stickers"] += [title for _, title in new]
            celebrate = p["stars"] // STARS_PER_CELEBRATION > before // STARS_PER_CELEBRATION
            try:
                self.profiles.save()
            except OSError as e:
                print(f"web: cannot save profiles: {e}", file=sys.stderr, flush=True)

            text = reply
            praise = ([f"Yay, {p['name']}! You have {p['stars']} stars!"] if celebrate else []) + \
                     [f"You got the {title} sticker!" for _, title in new]
            if praise and not time_up and feeling != "gentle":  # no fanfare after hurt or sadness
                first, *rest = split_parts(reply) or [""]
                text = " ".join([first] + praise + rest)
            voice = self.speak(text, feeling)
            self.last_turn = time.monotonic()
            numbers = {"stt_ms": stt.get("stt_ms"), "stt_total_ms": round((t1 - t0) * 1000),
                       "llm_ms": round((t2 - t1) * 1000), "audio_s": stt.get("audio_s"),
                       "heard_words": words, "reply_words": len(reply.split()), "raw_words": len(raw.split())}
            log_line({"kind": "turn", "id": voice["id"], "feeling": feeling, "checks": acted,
                      "parts": len(voice["parts"]), "support": support, "stars_added": added,
                      "celebrate": celebrate, "new_stickers": len(new), "bye": time_up, **numbers})
            return {**voice, "heard": heard, "reply": text, "feeling": feeling, "support": support,
                    "stars_added": added, "stars_total": p["stars"], "celebrate": celebrate,
                    "new_stickers": [title for _, title in new], "bye": time_up, **numbers}


class Handler(BaseHTTPRequestHandler):
    session: Session = None

    def send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.log_message("page left before the answer")

    def body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as e:
            raise TalkerError(400, "Content-Length is not a number") from e
        if length > limit:
            raise TalkerError(413, f"too large: {length} bytes")
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            raise TalkerError(400, "upload ended early")
        return data

    def json_body(self) -> dict:
        raw = self.body(4096)
        try:
            obj = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            raise TalkerError(400, "body is not valid JSON") from e
        if not isinstance(obj, dict):
            raise TalkerError(400, "body must be a JSON object")
        return obj

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                page = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif self.path in STATIC:
                data = (HERE / "static" / self.path.lstrip("/")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", STATIC[self.path])
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/status":
                s = self.session
                self.send_json(200, {"state": s.state, "character": s.config["character"],
                                     "child": s.profile["name"] if s.profile else None})
            elif self.path == "/profiles":
                self.send_json(200, {"profiles": self.session.profiles.listing(), "levels": list(LEVELS)})
            elif self.path.startswith("/profile?"):
                name = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("name", [""])[0]
                self.send_json(200, self.session.profiles.details(name))
            elif self.path.startswith("/audio/"):
                m = re.fullmatch(r"/audio/([0-9a-f]{12})/(\d{1,2})", self.path)
                audio = self.session.audio.get(m.group(1)) if m else None
                if audio is None:
                    raise TalkerError(404, "no such audio")
                data = audio.get(int(m.group(2)))
                self.send_response(200)
                self.send_header("Content-Type", "audio/ogg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    self.log_message("page stopped the audio early")
            else:
                raise TalkerError(404, f"no such page: {self.path}")
        except TalkerError as e:
            self.send_json(e.status, {"error": str(e)})
        except OSError as e:
            self.send_json(500, {"error": f"cannot read the page: {e}"})

    def do_POST(self):
        try:
            if self.path == "/start":
                self.send_json(200, self.session.start(self.json_body()))
            elif self.path == "/end":
                self.send_json(200, self.session.end("button"))
            elif self.path == "/delete":
                name = clean_name(self.json_body().get("name"))
                s = self.session
                with s.lock:
                    if s.profile and s.profile["name"].lower() == name.lower():
                        raise TalkerError(409, f"{name} is talking with Toffee now. Press End first.")
                    s.profiles.delete(name)
                log_line({"kind": "profile deleted"})
                self.send_json(200, {"ok": True})
            elif self.path == "/turn":
                audio = self.body(MAX_UPLOAD)
                if not audio:
                    raise TalkerError(400, "empty recording")
                self.send_json(200, self.session.turn(audio, self.headers.get("Content-Type", "")))
            elif self.path == "/log":
                entry = self.json_body()
                numbers = {k: round(float(v), 1) for k, v in entry.items()
                           if k in LOG_NUMBERS and isinstance(v, (int, float)) and not isinstance(v, bool)}
                if not numbers:
                    raise TalkerError(400, f"no known numbers; known: {sorted(LOG_NUMBERS)}")
                log_line({"kind": "page", **numbers})
                self.send_json(200, {"ok": True})
            else:
                raise TalkerError(404, f"no such endpoint: {self.path}")
        except TalkerError as e:
            self.log_message("error %d: %s", e.status, e)
            self.send_json(e.status, {"error": str(e)})


def main() -> int:
    try:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        persona = (HERE / "persona.txt").read_text(encoding="utf-8")
        for key in ("character", "animal", "about", "whisper_model", "llm_model", "idle_minutes", "session_minutes"):
            if key not in config:
                raise KeyError(key)
        for path in (PY_STT, PY_TTS, PAGE):
            if not path.exists():
                raise FileNotFoundError(path)
        profiles = Profiles(PROFILES)
    except (OSError, json.JSONDecodeError, KeyError, ProfileError) as e:
        print(f"web: cannot start: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    Handler.session = Session(config, persona, profiles)
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"web: cannot listen on {HOST}:{PORT}: {e}", file=sys.stderr)
        return 1
    server.daemon_threads = True

    def on_term(signum, frame):  # systemd stops the service with SIGTERM: clean up like Ctrl+C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    threading.Thread(target=Handler.session.idle_watch, daemon=True).start()
    print(f"web: Toffee on http://{HOST}:{PORT} (stops after {config['idle_minutes']} idle minutes)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if Handler.session.state != "off":
            Handler.session.end("web process stopped")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
