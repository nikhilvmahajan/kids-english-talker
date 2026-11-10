#!/usr/bin/env python3
"""Watch Tailscale's path to the phone during the stall test.

Every 2 s it reads `tailscale status` and sends one Tailscale ping to each
online Android device. It writes one line per device to paths.jsonl:
time, device number, direct or not, address class, ping route and ping
time, and the bytes per second that Tailscale moved to and from the device.
Address classes: home-lan (inside one of this desktop's own networks),
private-other, cgnat, public-v4, public-v6, none.
It never writes or prints a device name or an IP address.

Run:  python3 stall_watch.py [minutes]     (default 12; Ctrl+C stops it early)
Delete paths.jsonl after the results are copied into kids_english.md.
"""
import ipaddress
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "paths.jsonl"
PERIOD_S = 2.0
PONG = re.compile(r"via (?:DERP\((?P<derp>[^)]+)\)|(?P<addr>\S+)) in (?P<ms>[\d.]+)ms")


class WatchError(Exception):
    """tailscale could not be read; the sample is logged as an error."""


def tailscale_status() -> dict:
    try:
        p = subprocess.run(["tailscale", "status", "--json"],
                           capture_output=True, text=True, check=True, timeout=5)
    except FileNotFoundError as e:
        raise WatchError("tailscale command not found") from e
    except subprocess.CalledProcessError as e:
        raise WatchError(f"tailscale status exit code {e.returncode}") from e
    except subprocess.TimeoutExpired as e:
        raise WatchError("tailscale status took over 5 s") from e
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise WatchError("tailscale status gave no valid JSON") from e


CGNAT = ipaddress.ip_network("100.64.0.0/10")


def home_networks() -> list:
    """This desktop's own networks (not loopback, not Tailscale)."""
    try:
        p = subprocess.run(["ip", "-j", "addr"], capture_output=True, text=True, check=True, timeout=5)
        links = json.loads(p.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as e:
        print(f"cannot read own networks ({type(e).__name__}); no home-lan class", file=sys.stderr)
        return []
    nets = []
    for link in links:
        if link.get("ifname") in ("lo", "tailscale0"):
            continue
        for a in link.get("addr_info") or []:
            if a.get("scope") == "link":
                continue
            nets.append(ipaddress.ip_interface(f"{a['local']}/{a['prefixlen']}").network)
    return nets


def addr_class(hostport: str, nets: list) -> str:
    if not hostport:
        return "none"
    host = hostport.rsplit(":", 1)[0].strip("[]")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "unknown"
    if any(ip.version == n.version and ip in n for n in nets):
        return "home-lan"
    if ip.version == 4 and ip in CGNAT:
        return "cgnat"
    if ip.is_private:
        return "private-other"
    return f"public-v{ip.version}"


def ping(ip: str, nets: list) -> dict:
    try:
        p = subprocess.run(["tailscale", "ping", "--c", "1", "--timeout", "2s", ip],
                           capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return {"ping": "timeout"}
    text = p.stdout + p.stderr
    m = PONG.search(text)
    if not m:
        # The message itself is not kept: it can hold a name. Only its kind is.
        kind = "timed out" if "timed out" in text else f"failed, exit code {p.returncode}"
        return {"ping": f"no pong ({kind})"}
    if m.group("derp"):
        return {"ping": "derp", "ping_ms": float(m.group("ms")), "derp": m.group("derp")}
    return {"ping": addr_class(m.group("addr"), nets), "ping_ms": float(m.group("ms"))}


def sample(nets: list, prev: dict) -> list:
    """prev maps device number to (monotonic time, rx bytes, tx bytes); updated here."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        status = tailscale_status()
    except WatchError as e:
        return [{"t": now, "error": str(e)}]
    peers = sorted((key, p) for key, p in (status.get("Peer") or {}).items()
                   if (p.get("OS") or "").lower() == "android")
    rows = []
    for n, (_, p) in enumerate(peers):
        if not p.get("Online"):
            continue
        row = {"t": now, "device": n, "direct": bool(p.get("CurAddr")),
               "addr": addr_class(p.get("CurAddr") or "", nets)}
        mono, rx, tx = time.monotonic(), int(p.get("RxBytes") or 0), int(p.get("TxBytes") or 0)
        if n in prev and mono > prev[n][0]:
            dt = mono - prev[n][0]
            row["from_phone_kBps"] = round((rx - prev[n][1]) / dt / 1000, 1)
            row["to_phone_kBps"] = round((tx - prev[n][2]) / dt / 1000, 1)
        prev[n] = (mono, rx, tx)
        ips = p.get("TailscaleIPs") or []
        row.update(ping(ips[0], nets) if ips else {"ping": "no tailscale ip"})
        rows.append(row)
    if not rows:
        rows.append({"t": now, "error": "no Android device online"})
    return rows


def main() -> int:
    try:
        minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    except ValueError:
        print(f"minutes must be a number, not {sys.argv[1]!r}", file=sys.stderr)
        return 2
    end = time.monotonic() + minutes * 60
    nets, prev = home_networks(), {}
    print(f"watching for {minutes:g} min, {len(nets)} own networks known, writing {OUT}", flush=True)
    try:
        with OUT.open("a", encoding="utf-8") as f:
            while time.monotonic() < end:
                start = time.monotonic()
                for row in sample(nets, prev):
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
                time.sleep(max(0.0, PERIOD_S - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("stopped by Ctrl+C", flush=True)
    except OSError as e:
        print(f"cannot write {OUT}: {e}", file=sys.stderr)
        return 1
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
