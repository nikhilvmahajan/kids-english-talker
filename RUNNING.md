# Running Toffee

## What starts at boot

| Part | At boot | Check |
|---|---|---|
| The talker (`talker/web.py`) | starts by itself as a systemd user service (`kids-talker`). It holds no GPU and about 15 MB of RAM. | `systemctl --user is-active kids-talker` shows "active" |
| Whisper, the voice and Gemma | **do not** load at boot. They load when someone presses Start on the phone, and unload at End, at the 15-minute goodbye, or after 5 quiet minutes. | `nvidia-smi`, `ollama ps` |
| Ollama | starts by itself; holds no model until a session starts | `systemctl is-active ollama` |
| Tailscale | starts by itself | `tailscale status` |
| HTTPS forwarding to port 8765 (`tailscale serve --bg`) | Tailscale saves this setting and restores it | `tailscale serve status` shows `proxy http://127.0.0.1:8765` |

## One-time setup

```
cp deploy/kids-talker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kids-talker.service
sudo loginctl enable-linger "$USER"     # start at boot even when nobody logs in
sudo tailscale serve --bg 8765          # HTTPS for your own devices only
```

Without `enable-linger`, the service starts only when you log in to the desktop and stops when you log out.

## Put Toffee on the phone's home screen

1. On the phone, turn on the Tailscale app and open the Toffee page in Firefox.
2. Open the Firefox menu (⋮) and choose **Install** (or **Add to Home screen**).
3. A Toffee cat icon appears on the home screen. It opens the page full-screen, like an app. No typing needed.

## After a restart: check, if something does not work

1. `systemctl --user is-active kids-talker` should show "active". If not: `systemctl --user restart kids-talker`.
2. `tailscale serve status` should show `proxy http://127.0.0.1:8765`. If it shows "No serve config": `sudo tailscale serve --bg 8765`.
3. On the phone: Tailscale app on, then tap the Toffee icon.

## Everyday commands

| To | Command |
|---|---|
| See if it runs | `systemctl --user status kids-talker` |
| See its messages (numbers and errors only, never words) | `journalctl --user -u kids-talker -n 50` |
| Read a child's saved talks | open `talker/talks/<name>/` (one text file per session; only your user can read it) |
| Restart it (after a code change) | `systemctl --user restart kids-talker` |
| Stop it until the next boot | `systemctl --user stop kids-talker` |
| Stop it starting at boot | `systemctl --user disable --now kids-talker` |

Stopping the service also ends a running session: it stops the speech and voice services, unloads Gemma and frees the GPU.

## If something goes wrong

| What you see | What to do |
|---|---|
| The page does not load on the phone | Is the Tailscale app on? Steps 1 and 2 above. |
| "Not now: the computer is busy" | Another big GPU job is running. Try again when it ends. |
| "could not start" | Look at `talker/run/stt.log` and `talker/run/tts.log` (numbers and errors only). |
| Toffee does not speak | `journalctl --user -u kids-talker -n 50` |
| The GPU stays busy after End | `nvidia-smi`, and `ollama ps` (should list no model). |
