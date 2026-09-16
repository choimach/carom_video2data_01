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
import json
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
FAILED = os.path.join(ROOT, "data", "_unscannable.json")


def failures():
    if not os.path.exists(FAILED):
        return {}
    return json.load(open(FAILED, encoding="utf-8"))


def note_failure(name, reason):
    """Remember what could not be scanned, and why.

    Finding no calibratable frame costs a pass over the whole video, and a
    match that has none has none - repeating that every round is how four
    failures came to be re-tried on every collection run. Named runs
    (`scan_all.py AKR2026`) ignore this list, so a fix can be tested.
    """
    seen = failures()
    seen[name] = reason
    json.dump(seen, open(FAILED, "w", encoding="utf-8"), indent=1)


def matches(only=()):
    found = sorted(glob.glob(os.path.join(VIDEOS, "soop_*.mp4")))
    if only:
        found = [p for p in found if any(token in p for token in only)]
    return found


def main(argv):
    os.makedirs(SCANS, exist_ok=True)
    paths = matches(set(argv))
    print(f"{len(paths)} matches on disk", flush=True)
    skip = {} if argv else failures()
    for video in paths:
        name = os.path.splitext(os.path.basename(video))[0]
        track = os.path.join(SCANS, f"{name}.npz")
        if os.path.exists(track):
            print(f"== {name}: already scanned", flush=True)
            continue
        if name in skip:
            print(f"== {name}: skipped, {skip[name]}", flush=True)
            continue
        print(f"== {name}", flush=True)
        began = time.time()
        try:
            calibration, _fps = find_calibration(video)
            start = find_play_start(video, calibration)
            scan(video, track, start=start, calibration=calibration)
        except Exception as failure:
            traceback.print_exc()
            # A match that cannot be scanned should not stop the rest. Only a
            # failure in the scan itself justifies throwing the file away.
            if os.path.exists(track):
                os.remove(track)
            note_failure(name, f"{type(failure).__name__}: {failure}"[:200])
            continue

        # Analysis is seconds of work against the scan's half hour, and it runs
        # again on the cache any time. Losing the scan because the report after
        # it raised is how three matches came to be scanned for seventy-six
        # minutes and then deleted, over a division by zero in a progress line.
        try:
            report(analyse(load_scan(track)))
        except Exception:
            traceback.print_exc()
            print(f"   scan kept at {track}; the analysis is what failed", flush=True)
        print(f"   done in {(time.time() - began) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
