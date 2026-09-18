"""Pack what the assistant needs into one file the page can hold.

Every play the model may learn from, already canonicalised into one quarter of
the table, with the features the neighbour search runs on and the cue ball's own
path so an answer can show the plays it rests on rather than only assert them.

Paths are thinned to at most 32 points and rounded to whole millimetres: the
scan measures to about 2 mm and the page draws them a few pixels wide, so the
rest is weight for nothing.

    ~/.venvs/carom/bin/python tools/app_data.py --out build/app-data.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from model_dataset import BALLS, canonical, transform  # noqa: E402
from route_model import OUTCOME_IN_NAME, player_features  # noqa: E402

POINTS = 32


def thin(path):
    seen = [p for p in path if np.isfinite(p).all()]
    if len(seen) <= POINTS:
        return [[int(round(x)), int(round(y))] for x, y in seen]
    step = len(seen) / POINTS
    return [[int(round(seen[int(i * step)][0])), int(round(seen[int(i * step)][1]))]
            for i in range(POINTS)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "model.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "build", "app-data.json"))
    args = parser.parse_args(argv)

    rows = [r for r in json.load(open(args.data, encoding="utf-8"))
            if r["route"] not in OUTCOME_IN_NAME]
    features = np.array([player_features(r) for r in rows])
    middle, spread = features.mean(axis=0), features.std(axis=0)
    spread[spread == 0] = 1.0
    scaled = (features - middle) / spread

    paths = {}
    for name in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*_traj.npz"))):
        match = os.path.basename(name)[:-9]
        paths[match] = np.load(name)

    plays, missing = [], 0
    for row, point in zip(rows, scaled):
        bundle = paths.get(row["match"])
        key = f"i{row['inning']:03d}s{row['shot']:02d}_{row['cue']}"
        if bundle is None or key not in bundle:
            missing += 1
            continue
        # The stored layout is canonical; the path is not, so it is turned the
        # same way before it is kept, or it would be drawn somewhere else
        # entirely from the layout it belongs to.
        raw = np.array(row["layout_mm"][row["cue"]], dtype=float)
        original = bundle[key]
        flip_y = bool(abs(original[0][0] - raw[0]) > 1.0) if len(original) else False
        flip_x = bool(abs(original[0][1] - raw[1]) > 1.0) if len(original) else False
        turned = [transform(p, flip_x, flip_y) for p in original if np.isfinite(p).all()]
        plays.append({
            "f": [round(float(v), 3) for v in point],
            "route": row["route"],
            "scored": row["scored"],
            "cue": [int(round(v)) for v in row["layout_mm"][row["cue"]]],
            "balls": [[int(round(v)) for v in row["layout_mm"][c]]
                      for c in BALLS if c != row["cue"]],
            "path": thin(turned),
            "match": row["match"].replace("soop_", ""),
        })

    out = {"mean": [round(float(v), 4) for v in middle],
           "std": [round(float(v), 4) for v in spread],
           "plays": plays}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), separators=(",", ":"))
    size = os.path.getsize(args.out) / 1024
    print(f"{len(plays)} plays ({missing} without a path), {size:.0f} KB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
