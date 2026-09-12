"""Download every match that screening marked usable.

Screening leaves its verdicts in `data/_screening.json`; this takes the ones
that passed and fetches them in full, newest event first. Files are named after
the VOD id so a half-finished run can tell what it already has, and yt-dlp is
left to resume its own partial downloads.

    ~/.venvs/carom/bin/python tools/collect.py            # everything usable
    ~/.venvs/carom/bin/python tools/collect.py AKR2026 SHWC2025
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_acquisition.screen import STREAM_FORMAT, _yt_dlp  # noqa: E402

VIDEOS = os.path.join(ROOT, "data", "videos")
SCREENING = os.path.join(ROOT, "data", "_screening.json")


def wanted(only=()):
    entries = json.load(open(SCREENING, encoding="utf-8"))
    rows = [e for e in entries if e.get("usable") and e.get("vod_id")]
    if only:
        rows = [e for e in rows if e.get("event") in only or str(e["vod_id"]) in only]
    return sorted(rows, key=lambda e: -int(e["vod_id"]))


def target(entry):
    return os.path.join(VIDEOS, f"soop_{entry['vod_id']}_{entry.get('event', 'x')}.mp4")


def main(argv):
    os.makedirs(VIDEOS, exist_ok=True)
    rows = wanted(set(argv))
    print(f"{len(rows)} matches to collect")
    done = 0
    for entry in rows:
        path = target(entry)
        if os.path.exists(path):
            print(f"  have {os.path.basename(path)}")
            done += 1
            continue
        started = time.time()
        print(f"  get  {os.path.basename(path)}  {entry.get('title', '')[:50]}", flush=True)
        result = subprocess.run(
            [_yt_dlp(), "--no-warnings", "--newline", "-f", STREAM_FORMAT,
             "-o", path, entry["url"]],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        if result.returncode or not os.path.exists(path):
            print(f"       FAILED: {result.stderr.strip().splitlines()[-1:] or ''}", flush=True)
            continue
        size = os.path.getsize(path) / 2**30
        print(f"       {size:.1f} GB in {(time.time() - started) / 60:.0f} min", flush=True)
        done += 1
    print(f"{done}/{len(rows)} collected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
