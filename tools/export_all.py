"""Turn every cached scan into the dataset.

One JSON and one .npz per match, plus an index across all of them. The JSON
holds what each play was - the layout it was played from, which ball was the
cue, what it hit, whether it scored, where the balls finished - and the .npz
holds the three balls' paths through it, in table millimetres.

Rejected plays stay in the JSON with the reason they were rejected, because
knowing what was thrown away is part of knowing what the dataset is. Only
usable plays get a trajectory.

Analysis is cheap and scans are not, so this re-analyses from the cache every
time rather than storing a verdict: whatever the labelling rules are today is
what comes out.

    ~/.venvs/carom/bin/python tools/export_all.py
"""

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import analyse, export_json, export_trajectories, load_scan  # noqa: E402


def turns_of(result):
    return result["turns"]

SCANS = os.path.join(ROOT, "data", "scans")
DATASET = os.path.join(ROOT, "data", "dataset")
SCREENING = os.path.join(ROOT, "data", "_screening.json")


def titles():
    """{vod_id: title} from the screening record, for naming the matches."""
    if not os.path.exists(SCREENING):
        return {}
    return {str(e.get("vod_id")): e.get("title", "") for e in json.load(open(SCREENING))}


def main(argv):
    os.makedirs(DATASET, exist_ok=True)
    known = titles()
    scans = sorted(glob.glob(os.path.join(SCANS, "*.npz")))
    if argv:
        scans = [p for p in scans if any(token in p for token in argv)]
    print(f"{len(scans)} scans", flush=True)

    index, totals = [], {"plays": 0, "usable": 0, "scored": 0}
    for path in scans:
        name = os.path.splitext(os.path.basename(path))[0]
        found = re.match(r"soop_(\d+)_(.+)", name)
        vod_id, event = (found.group(1), found.group(2)) if found else ("", name)

        scan_data = load_scan(path)
        result = analyse(scan_data)
        plays = [s for inning in result["innings"] for s in inning.shots]
        usable = [s for s in plays if not getattr(s, "rejections", None) and not s.inferred]

        written = export_json(result, os.path.join(DATASET, f"{name}.json"))
        traced = export_trajectories(scan_data, result, os.path.join(DATASET, f"{name}_traj.npz"))

        entry = {
            "match": name,
            "vod_id": vod_id,
            "event": event,
            "title": known.get(vod_id, ""),
            "fps": result["fps"],
            "mm_per_px": scan_data["mm_per_px"],
            "reprojection_error_mm": scan_data["reprojection_error"],
            "turns": len(result["turns"]),
            "expected_plays": result["expected_plays"],
            "detected_plays": written,
            "usable_plays": traced,
            "scored": sum(1 for s in usable if s.success),
            "missed": sum(1 for s in usable if s.success is False),
        }
        final = result.get("final_board")
        if final:
            entry["final_score"] = {"white": final["white"], "yellow": final["yellow"]}
            entry["match_finished"] = final["finished"]
            entry["points_counted"] = {
                "white": sum(p for *_r, c, p in turns_of(result) if c == "white"),
                "yellow": sum(p for *_r, c, p in turns_of(result) if c == "yellow"),
            }
        index.append(entry)
        totals["plays"] += written
        totals["usable"] += traced
        totals["scored"] += entry["scored"]
        print(f"  {event:<12} {written:4d} plays, {traced:4d} usable "
              f"({entry['scored']} scored / {entry['missed']} missed), "
              f"{entry['reprojection_error_mm']:.1f} mm", flush=True)

    with open(os.path.join(DATASET, "index.json"), "w", encoding="utf-8") as handle:
        json.dump({"matches": index, "totals": totals}, handle, indent=1, ensure_ascii=False)
    print(f"\n{len(index)} matches: {totals['plays']} plays detected, "
          f"{totals['usable']} usable, {totals['scored']} of them scoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
