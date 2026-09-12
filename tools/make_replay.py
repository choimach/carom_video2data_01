"""Pack one match's plays into the data file the replay table reads.

Positions go out as whole millimetres at 30 frames a second. The table is
2844 mm long, a ball is 61.5 mm across, and the scan measures to about 2 mm, so
a millimetre is already finer than the data and the second decimal place is
noise that would double the file.

    ~/.venvs/carom/bin/python tools/make_replay.py POWC2026 --out build/replay-data.js
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BALLS = ("white", "yellow", "red")
STRIDE = 2  # 60 fps down to 30


def pack(match, dataset=None):
    dataset = dataset or os.path.join(ROOT, "data", "dataset")
    plays = json.load(open(os.path.join(dataset, f"{match}.json"), encoding="utf-8"))["plays"]
    bundle = np.load(os.path.join(dataset, f"{match}_traj.npz"))
    index = json.loads(str(bundle["index"]))
    fps = float(bundle["fps"])

    by_key = {(p["inning"], p["shot_number"]): p for p in plays}
    out = []
    for entry in index:
        play = by_key.get((entry["inning"], entry["shot_number"]))
        if play is None:
            continue
        paths = {}
        for colour in BALLS:
            xy = bundle[f"{entry['key']}_{colour}"][::STRIDE]
            # A gap in the track is a frame where the ball was not seen. Carrying
            # the last known position across it would invent a ball standing
            # still; -1 lets the table hide it instead.
            flat = np.where(np.isfinite(xy), np.round(xy), -1).astype(int)
            paths[colour] = flat.reshape(-1).tolist()
        out.append({
            "key": entry["key"],
            "inning": entry["inning"],
            "shot": entry["shot_number"],
            "cue": play["cue_ball"],
            "scored": play["success"],
            "source": play.get("verdict_source"),
            "cushions": play.get("cushions_before_second"),
            "first": play.get("first_object_ball"),
            "second": play.get("second_object_ball"),
            "thickness": play.get("thickness"),
            "spinX": play.get("spin_x"),
            "spinY": play.get("spin_y"),
            "spinRail": play.get("spin_rail"),
            "speed": play.get("cue_speed_ms"),
            "complete": play.get("complete"),
            "second_at": round(play.get("start_second", 0.0), 1),
            "paths": paths,
        })
    return {"match": match, "fps": fps / STRIDE, "plays": out}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pack a match for the replay table")
    parser.add_argument("match", help="part of a dataset filename, e.g. POWC2026")
    parser.add_argument("--out", default=os.path.join(ROOT, "build", "replay-data.js"))
    args = parser.parse_args(argv)

    found = [os.path.basename(p)[:-5] for p in
             sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json")))
             if args.match in p and not p.endswith("index.json")]
    if not found:
        print(f"no dataset matching {args.match}", file=sys.stderr)
        return 1

    data = pack(found[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("window.CAROM = ")
        json.dump(data, handle, separators=(",", ":"))
        handle.write(";\n")
    size = os.path.getsize(args.out) / 1024
    print(f"{len(data['plays'])} plays -> {args.out} ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
