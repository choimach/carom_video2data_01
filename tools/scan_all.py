"""Scan every collected match, once, and cache the result.

The expensive pass is the scan: ball positions and scoreboard readings for
every frame the table is on screen, two and a half times faster than real time.
What it writes is raw observation, not judgement, so a cached scan survives
every later change to how plays are segmented and labelled - `analyse` runs on
it in seconds. That is the whole point of keeping the two apart, and it means
scanning can start before the analysis has settled.

    ~/.venvs/carom/bin/python tools/scan_all.py
    ~/.venvs/carom/bin/python tools/scan_all.py AKR2026      # just these
"""

import glob
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import (analyse, find_calibration, find_play_start,  # noqa: E402
                          load_scan, report, scan)

VIDEOS = os.path.join(ROOT, "data", "videos")
SCANS = os.path.join(ROOT, "data", "scans")


def matches(only=()):
    found = sorted(glob.glob(os.path.join(VIDEOS, "soop_*.mp4")))
    if only:
        found = [p for p in found if any(token in p for token in only)]
    return found


def main(argv):
    os.makedirs(SCANS, exist_ok=True)
    paths = matches(set(argv))
    print(f"{len(paths)} matches on disk", flush=True)
    for video in paths:
        name = os.path.splitext(os.path.basename(video))[0]
        track = os.path.join(SCANS, f"{name}.npz")
        if os.path.exists(track):
            print(f"== {name}: already scanned", flush=True)
            continue
        print(f"== {name}", flush=True)
        began = time.time()
        try:
            calibration, _fps = find_calibration(video)
            start = find_play_start(video, calibration)
            scan(video, track, start=start, calibration=calibration)
            report(analyse(load_scan(track)))
        except Exception:
            traceback.print_exc()
            # A match that cannot be scanned should not stop the rest.
            if os.path.exists(track):
                os.remove(track)
            continue
        print(f"   done in {(time.time() - began) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
