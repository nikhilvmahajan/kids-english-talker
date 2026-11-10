# Toffee: a local talking cat that helps children practise spoken English

Toffee is a cartoon cat on a phone screen. A child holds a button, says something in English, and Toffee answers in a friendly voice, in one or two short sentences and one question. Toffee repeats mistakes back the right way ("I goed to park" → "Oh, you went to the park!"), gives the English word for a Hindi or Marathi word, and praises effort.

Everything runs on one home computer with an NVIDIA GPU. No cloud service hears the child or reads the text. The phone only shows the page and plays the voice.

It was built for a few children (ages 6 to 10) whose first language is not English. It is an early, working prototype: tested by the author on one Android phone (Firefox) and one desktop. It has not yet been used with children.

## How a turn works

```
phone: hold the button, speak, release
  → desktop: Whisper (speech to text, with a voice detector so silence gives no text)
  → Gemma 4 e4b in Ollama (Toffee's reply, with the character and safety instructions)
  → code checks (one question only; no school, city, street or number repeated)
  → Indic Parler-TTS, speaker Priya, raised 5 semitones (one small audio file per sentence)
phone: plays each sentence and shows its words as Toffee says them
```

Measured on the author's setup (RTX 3060 12 GB, phone on home Wi-Fi): Toffee's first sound starts about 1.1–1.7 s after the child releases the button. The three models together use about 7.8 GB of GPU memory.

## Privacy and safety design

- No audio is stored. No text the child says is stored or logged. The run log (`talker/log.jsonl`) keeps timing numbers and the names of the checks that acted.
- Profiles (`talker/profiles.json`, owner-only file access) keep a child's name, age, English level, family words, stars, counts, days played, stickers and the next topic. Nothing the child said.
- The character instructions (`talker/persona.txt`) forbid asking for a surname, address, school or phone number, secrets, scary stories and bad words. If a child is hurt, scared or sad, Toffee says sorry and asks them to tell a family grown-up "or another grown-up you trust". Code checks back up the most important rules, because the model follows instructions most of the time, not every time.
- Rewards are for effort, never correctness: stars for speaking, stickers for milestones, no leaderboard between children, no penalty for a missed day.
- Screen time: after 15 minutes Toffee gets sleepy and says goodbye; the session ends and the GPU is freed. The "End" button also frees all GPU memory (the models run in separate processes that stop).
- Before it starts, the talker checks that no other big GPU job is running; if one is, it says "not now".
- Parents should agree before their child uses it, and an adult should stay nearby.

## What you need

- Linux with an NVIDIA GPU with about 10 GB free (tested: RTX 3060 12 GB, Ubuntu).
- `ffmpeg`, Python 3.14 (speech to text) and Python 3.10 (voice), Ollama.
- A free Hugging Face account: Indic Parler-TTS is a gated model (accept its terms, then use a read token).
- For a phone: the page needs HTTPS for the microphone. The author uses `tailscale serve`, which gives HTTPS on your own devices only.

## Set up

```
# 1. Speech-to-text environment (Python 3.14)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Systran/faster-whisper-small', cache_dir='models')"
#    (HF_HUB_DISABLE_XET=1: the newer download method stalled at 0 bytes for the author)

# 2. Voice environment (Python 3.10)
python3.10 -m venv k_t3/.venv-parler
k_t3/.venv-parler/bin/pip install -r requirements-parler.txt
#    put HF_TOKEN=<your read token> in credentials.cred, then: python3 load_creds.py  (writes .env)
#    run k_t3/parler_samples.py once to download the model into models/hf

# 3. The language model
ollama pull gemma4:e4b

# 4. Run
python3 talker/web.py            # serves http://127.0.0.1:8765
sudo tailscale serve --bg 8765   # optional: HTTPS for your own phone
```

`talker/config.json` holds the character's name and description, the models and the time limits.

## Folders

| Folder | What it is |
|---|---|
| `talker/` | The app: `web.py` (always on, no GPU), `stt_service.py` (Whisper), `tts_service.py` (voice), `static/index.html` (the page with the animated cat), `persona.txt`, `config.json` |
| `k_t1/` | Test: phone → desktop → phone round trip and a network stall test (found that Wi-Fi power saving on the desktop caused multi-second stalls) |
| `k_t2/` | Test: Whisper speed and accuracy on the phone (read sentences and free talk) |
| `k_t3/` | Tests: voice candidates (Kokoro, Indic Parler-TTS), Priya with pitch raised, time to first sound |
| `k_t4/` | Test: language models as the character, with 27 synthetic child turns × 3 runs × ages 6 and 10, and automatic rule checks |
| `load_creds.py` | Copies `KEY=VALUE` lines from `credentials.cred` into `.env`; prints key names only |

Tests that touch profiles must set `TALKER_PROFILES` to a separate file, so they never change real children's data.

## Models and licences

| Part | Model | Licence |
|---|---|---|
| Speech to text | Whisper small (OpenAI) through faster-whisper (SYSTRAN); voice detector Silero VAD | MIT |
| Toffee's words | Gemma 4 e4b (Google) through Ollama | Apache 2.0 |
| Voice | Indic Parler-TTS (AI4Bharat), speaker Priya | Apache 2.0 (gated download) |
| GPU maths for Whisper | NVIDIA cuBLAS (pip package) | NVIDIA proprietary, free |

The models are not in this repository. Check each licence yourself before any commercial use.

## Decision IDs in the code

Comments refer to the author's design log. The ones that appear in the code:

| ID | Meaning |
|---|---|
| K-P1 | The character is original (not a brand cartoon). |
| K-P2 | Child-safety rules, backed by code checks. |
| K-P4 | Teaching method: repeat mistakes back the right way, give the English word, praise effort. |
| K-P6 | The developer never reads the children's audio or text; tests use synthetic samples. |
| K-D4 | `tailscale serve` forwards HTTPS to port 8765; only this app listens there. |
| K-D5 | "End" frees GPU memory, RAM and CPU. |
| K-D6 | If another big GPU job runs, "Start" says "not now". |
| K-D7 | No separate test with a child before the first real session. |
| K-D8 | A general landmark may be repeated; a school, city, town, street or number never. |
| K-D10 | The voice: Priya, style by feeling (happy, curious, gentle), raised 5 semitones. |
| K-D12 | Profiles, stars, stickers, goodbye topic, no leaderboard. |
| K-D13 | 15-minute limit with a playful goodbye. |
| K-Q1 | Children's ages and levels (6 to 10). |
| K-T1 … K-T7 | The tests in `k_t1` … `k_t4` and the talker build. |

## Known limits

- Whisper has not been tested on children's voices. Names are its weak point (it heard "Nagpur" as "Naktur").
- On Firefox for Android, streamed audio did not start until the whole answer arrived, so the voice is sent as one small file per sentence.
- The model follows each rule most of the time, not every time; code checks cover the most important ones, but not all.
- One child at a time. The web process runs in the foreground (no system service yet).

## Licence

MIT for the code in this repository (see `LICENSE`). The models have their own licences (see above).

## Credits

Built by the repository owner with help from Claude (Anthropic). Thanks to the makers of Whisper, faster-whisper, Gemma, Ollama, Parler-TTS and Indic Parler-TTS.
