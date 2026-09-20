"""Download every match that screening marked usable.

Screening leaves its verdicts in `data/_screening.json`; this takes the ones
that passed and fetches them in full, newest event first. Files are named after
the VOD id so a half-finished run can tell what it already has, and yt-dlp is
left to resume its own partial downloads.

A download that finishes is not necessarily a video. Three of the matches
fetched so far are ten gigabytes apiece that no decoder will open - the boxes
are all there and complete, and ffmpeg still stops at "error reading header" -
and the pipeline only found out half an hour into the scan, where it looked
like a table that could not be calibrated. Every download is now opened and
read from before it counts as collected.

    ~/.venvs/carom/bin/python tools/collect.py            # everything usable
    ~/.venvs/carom/bin/python tools/collect.py AKR2026 SHWC2025
"""

import glob
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
UNREADABLE = os.path.join(ROOT, "data", "_unreadable.json")
RETIRED = os.path.join(ROOT, "data", "videos.json")
ATTEMPTS = 2


def unreadable():
    if not os.path.exists(UNREADABLE):
        return {}
    return json.load(open(UNREADABLE, encoding="utf-8"))


def note_unreadable(vod_id):
    """Count the failures, so a match is not fetched forever.

    A download that comes back broken is worth one more try - the first may
    have been the connection - but not a third: ten gigabytes is too much to
    spend on a hunch every round.
    """
    seen = unreadable()
    seen[str(vod_id)] = seen.get(str(vod_id), 0) + 1
    json.dump(seen, open(UNREADABLE, "w", encoding="utf-8"), indent=1)
    return seen[str(vod_id)]


def retired():
    """스캔이 끝나 지운 경기. 다시 받으면 안 된다.

    `tools/retire_videos.py`가 영상을 지우면서 주소를 `data/videos.json`에 적어
    둔다. 그 목록을 여기서 빼지 않으면, 지운 39경기가 전부 "아직 안 받은 것"으로
    보여서 **347 GB를 그대로 다시 받는다.** 지우기와 받기는 짝이라, 한쪽만
    만들어 두면 다음 라운드가 되돌린다.
    """
    if not os.path.exists(RETIRED):
        return set()
    kept = json.load(open(RETIRED, encoding="utf-8"))
    return {str(row.get("vod_id")) for row in kept.get("matches", {}).values()
            if row.get("vod_id")}


def wanted(only=()):
    entries = json.load(open(SCREENING, encoding="utf-8"))
    rows = [e for e in entries if e.get("usable") and e.get("vod_id")]
    gone = retired()
    rows = [e for e in rows if str(e["vod_id"]) not in gone]
    if only:
        rows = [e for e in rows if e.get("event") in only or str(e["vod_id"]) in only]
    else:
        broken = unreadable()
        rows = [e for e in rows if broken.get(str(e["vod_id"]), 0) < ATTEMPTS]
    return sorted(rows, key=lambda e: -int(e["vod_id"]))


def target(entry):
    """Where this match belongs, or where it already is.

    The VOD id is what identifies a match; the rest of the name is a label,
    and the labels have changed - the first two matches were collected by
    player name and the rest by event. Looking the id up on disk keeps that
    from costing ten gigabytes of downloading a match already held.
    """
    held = glob.glob(os.path.join(VIDEOS, f"soop_{entry['vod_id']}_*.mp4"))
    if held:
        return held[0]
    return os.path.join(VIDEOS, f"soop_{entry['vod_id']}_{entry.get('event', 'x')}.mp4")


def readable(path):
    """Can frames actually be decoded out of this file?

    Both ends are worth asking about: a file whose header is broken fails on
    the first frame, and one that stopped early opens happily and then has
    nothing where the match is.
    """
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return False
        if not capture.read()[0]:
            return False
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames < 2:
            return False
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames * 0.8))
        return capture.read()[0]
    finally:
        capture.release()


def main(argv):
    os.makedirs(VIDEOS, exist_ok=True)
    rows = wanted(set(argv))
    print(f"{len(rows)} matches to collect")
    done = 0
    for entry in rows:
        path = target(entry)
        if os.path.exists(path):
            if readable(path):
                print(f"  have {os.path.basename(path)}")
                done += 1
                continue
            tries = note_unreadable(entry["vod_id"])
            print(f"  bad  {os.path.basename(path)}: no frames decode "
                  f"(attempt {tries}); fetching again", flush=True)
            os.remove(path)
            if tries >= ATTEMPTS:
                print("       giving up on this match", flush=True)
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
        if not readable(path):
            tries = note_unreadable(entry["vod_id"])
            print(f"       DOWNLOADED BUT UNREADABLE (attempt {tries}); removed",
                  flush=True)
            os.remove(path)
            continue
        done += 1
    print(f"{done}/{len(rows)} collected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
