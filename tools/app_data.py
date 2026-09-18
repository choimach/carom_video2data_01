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


def opening(path):
    """The direction and speed the cue ball actually left at, from the path.

    This is the seed the assistant searches around: the professional's own
    aim on a layout like the one in front of the player. The path is already
    turned into the canonical quarter, so the angle is in that frame too and
    has to be turned back with the layout.

    Measured over the first few frames, where the ball is going fastest and
    straightest, and skipping the very first - a ball just struck is the
    hardest thing on the table for the detector to place.
    """
    seen = [p for p in path if np.isfinite(p).all()]
    if len(seen) < 6:
        return None, None
    start, end = np.asarray(seen[1], float), np.asarray(seen[5], float)
    step = end - start
    travelled = float(np.linalg.norm(step))
    if travelled < 5.0:
        return None, None
    degrees = float(np.degrees(np.arctan2(step[1], step[0])) % 360.0)
    # Four frames at 60 fps, and the packer keeps every frame of the original.
    return round(degrees, 1), round(travelled * 15.0)


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
        # Which object ball was struck first, as near or far from the cue -
        # the colours are arbitrary but "the one nearer him" survives every
        # mirror and every swap of who is playing which ball.
        struck = row.get("first_object_ball")
        near_first = None
        if struck and struck != row["cue"] and struck in row["layout_mm"]:
            cue_at = np.array(row["layout_mm"][row["cue"]], dtype=float)
            gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue_at))
                    for c, xy in row["layout_mm"].items() if c != row["cue"]}
            near_first = gaps[struck] == min(gaps.values())
        # Negative offset is the object ball's right face - the same
        # convention the page uses when it names a candidate.
        struck = row.get("struck_side")
        plays.append({
            "f": [round(float(v), 3) for v in point],
            "near": near_first,
            "face": None if struck is None else ("left" if struck > 0 else "right"),
            "route": row["route"],
            "scored": row["scored"],
            "cue": [int(round(v)) for v in row["layout_mm"][row["cue"]]],
            "balls": [[int(round(v)) for v in row["layout_mm"][c]]
                      for c in BALLS if c != row["cue"]],
            "path": thin(turned),
            "aim": opening(turned)[0],
            "speed": opening(turned)[1],
            # 두께는 절반쯤의 플레이에만 있다 (적구가 떠나는 각을 읽을 만큼
            # 깨끗하게 찍힌 경우). 없는 것은 없는 대로 두고, 순위에서는 있는
            # 이웃만 세어 평균을 낸다 - 없는 것을 0으로 채우면 프로가 전부
            # 얇게 친 것처럼 보인다.
            "thick": (None if row.get("thickness") is None
                      else round(float(row["thickness"]), 3)),
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
