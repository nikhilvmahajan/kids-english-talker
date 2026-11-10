#!/usr/bin/env python3
"""Copy every KEY=VALUE line from a .cred file into .env, merged with existing keys.

Generic: it hard-codes no key names. It prints key NAMES only, never values.
Lines that start with # and blank lines are skipped. Values need no quotes.

Run from the project folder:
    python3 load_creds.py [credentials.cred] [.env]
Defaults: credentials.cred and .env in the current folder.
Both files are set to owner-only access (mode 600).
"""
import os
import re
import sys
from pathlib import Path

KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CredError(Exception):
    """The input cannot be used; the message names the line, never its value."""


def parse(path: Path) -> dict:
    pairs = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key, sep, value = text.partition("=")
        key = key.strip()
        if not sep or not KEY.match(key):
            raise CredError(f"{path.name} line {n} is not KEY=VALUE")
        pairs[key] = value.strip()
    return pairs


def main() -> int:
    cred = Path(sys.argv[1] if len(sys.argv) > 1 else "credentials.cred")
    env = Path(sys.argv[2] if len(sys.argv) > 2 else ".env")
    try:
        new = parse(cred)
        old = parse(env) if env.exists() else {}
    except FileNotFoundError:
        print(f"not found: {cred}", file=sys.stderr)
        return 1
    except CredError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    empty = [k for k, v in new.items() if not v]
    if empty:
        print(f"stopped: empty value for {', '.join(empty)} in {cred.name}; fill it in first", file=sys.stderr)
        return 1
    merged = {**old, **new}
    fd = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(f"{k}={v}\n" for k, v in merged.items()))
    os.chmod(env, 0o600)
    os.chmod(cred, 0o600)
    print(f"wrote {env}: {', '.join(sorted(merged))} (added or updated: {', '.join(sorted(new))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
