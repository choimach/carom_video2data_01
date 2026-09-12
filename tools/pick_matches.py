"""
Choose which matches to screen next.

The channel holds some sixteen hundred full three-cushion matches, and only a
fraction of them are shot from above the table - the only view this pipeline can
use. Screening costs a gigabyte each, so the order they are tried in matters.

Two levers, both of which the catalogue supports without any extra lookups:

* Recency. The VOD id rises with time, so sorting by it orders the catalogue
  from newest without needing a date. Recency matters because the production
  changed: the 2023 coverage under the old AfreecaTV branding is shot from the
  side and cannot be calibrated at all, while the 2026 SOOP coverage is
  overhead.

* Event spread. One match per event first, so a single unusable production does
  not take the whole sample down with it - which is exactly what happened when
  ten were picked across ten events and the first one screened was unusable.

    python tools/pick_matches.py --count 20 --out tools/next20.txt
    python tools/pick_matches.py --count 20 --per-event 3 --newest-first
"""

import argparse
import collections
import os
import re
import sys

# Scotch doubles is two players alternating on one cue ball, so an inning
# does not belong to one player and the play-per-point invariant does not
# hold. It is three-cushion, and it is still no use here.
NOT_THREE_CUSHION = re.compile(
    r"artistic|5\s*pin|\bpool\b|snooker|team|mt\b|w3gp|scotch", re.I)
NOT_A_FULL_MATCH = re.compile(
    r"클립|highlight|\bH\s*/\s*L\b|\bHL\b|sketch|ceremony|interview|shorts|"
    r"\bbest\b|top\s*\d|run\b|편집|요약|replay", re.I)
LOOKS_LIKE_A_MATCH = re.compile(r"^\[[^\]]+\]\s*[A-Z]\.?\s*[A-Z].*\bvs\b", re.I)
ROUND_ORDER = (("FINAL", 0), ("SEMI", 1), ("QUARTER", 2), ("L8", 2), ("L16", 3), ("_Q", 4), ("PQ", 5))


def event_of(title):
    found = re.match(r"\[([A-Za-z0-9]+)", title)
    return found.group(1) if found else "?"


def round_rank(title):
    upper = title.upper()
    for token, rank in ROUND_ORDER:
        if token in upper:
            return rank
    return 6


def read_listing(path):
    """Lines of `vod_id|title`, as produced by yt-dlp --flat-playlist."""
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("|", 1)
        if len(parts) != 2 or not parts[0].strip().isdigit():
            continue
        vod_id, title = int(parts[0]), parts[1]
        if not LOOKS_LIKE_A_MATCH.search(title):
            continue
        if NOT_A_FULL_MATCH.search(title) or NOT_THREE_CUSHION.search(title):
            continue
        rows.append((vod_id, title))
    return rows


def pick(rows, count, per_event=1, newest_first=True, exclude=()):
    """Round-robin across events, newest events first, best round within each."""
    rows = [r for r in rows if r[0] not in exclude]
    groups = collections.defaultdict(list)
    for vod_id, title in rows:
        groups[event_of(title)].append((vod_id, title))
    for event in groups:
        groups[event].sort(key=lambda r: (round_rank(r[1]), -r[0] if newest_first else r[0]))
    # Events ordered by how recent their newest video is.
    order = sorted(groups, key=lambda e: -max(v for v, _ in groups[e]))

    chosen, round_number = [], 0
    while len(chosen) < count and round_number < per_event:
        added = False
        for event in order:
            if len(groups[event]) > round_number and len(chosen) < count:
                vod_id, title = groups[event][round_number]
                chosen.append((vod_id, event, title))
                added = True
        if not added:
            break
        round_number += 1
    return chosen


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Pick matches to screen")
    parser.add_argument("--listing", default=os.path.join(root, "tools", "catalogue.txt"),
                        help="vod_id|title lines from yt-dlp --flat-playlist")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--per-event", type=int, default=1,
                        help="how many matches to take from one event before repeating")
    parser.add_argument("--newest-first", action="store_true", default=True)
    parser.add_argument("--min-id", type=int, default=None,
                        help="skip anything older than this VOD id")
    parser.add_argument("--skip-screened", default=os.path.join(root, "data", "_screening.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not os.path.exists(args.listing):
        print(f"no catalogue at {args.listing}; write one with:\n"
              f"  yt-dlp --flat-playlist --print '%(id)s|%(title)s' "
              f"https://ch.sooplive.co.kr/afbilliards1/vods > {args.listing}", file=sys.stderr)
        return 1

    rows = read_listing(args.listing)
    if args.min_id:
        rows = [r for r in rows if r[0] >= args.min_id]
    seen = set()
    if args.skip_screened and os.path.exists(args.skip_screened):
        import json
        for entry in json.load(open(args.skip_screened)):
            if entry.get("vod_id"):
                seen.add(int(entry["vod_id"]))

    chosen = pick(rows, args.count, args.per_event, args.newest_first, exclude=seen)
    print(f"{len(rows)} full matches in the catalogue, {len(seen)} already screened, "
          f"picking {len(chosen)}")
    lines = [f"{vod_id}|{event}|{title}" for vod_id, event, title in chosen]
    for line in lines:
        print("   " + line[:96])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
