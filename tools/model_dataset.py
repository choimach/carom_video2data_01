"""The plays a model may learn from, canonicalised and split by match.

Three filters, and each one throws away something for a stated reason:

* The path has to show what the label says happened. A play the inning's points
  call a point, whose path reconstructs no carom, has a tracking fault
  somewhere in it - and a layout read from a faulty track is not a layout.
* The route has to have a name. 미분류 is the labelling saying it does not know,
  and training on it teaches the model to not know.
* All three balls have to have been seen at the start. A layout missing a ball
  is not a layout.

Then the symmetry. A carom table has four: itself, the mirror in each axis, and
the half turn. A layout and its three images are the same shot, so every one is
turned to face the same way before anything compares them - otherwise the same
shot sits in four places and nothing finds it twice. The transform that was
used is kept, because an answer has to be turned back to face the player.

⚠️ A mirror flips side spin: left becomes right, and the rail names swap with
it. The half turn does not. Anything reading spin has to apply that, which is
why `english` is carried through the transform rather than copied.

    ~/.venvs/carom/bin/python tools/model_dataset.py --out data/model.json
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM  # noqa: E402

BALLS = ("white", "yellow", "red")
# Held out before any model is fitted, and never moved: three matches from three
# different events, a tenth of the plays. Chosen by name rather than at random
# so that the split survives every re-export, and by match rather than by play
# because two plays from one match share a table, a camera and often a layout.
TEST_MATCHES = ("soop_198597927_AKR2026", "soop_196625005_HCMC2026",
                "soop_206415235_LIWC2026")

MIRROR_X = "mirror-long"    # across the long axis: y -> W - y
MIRROR_Y = "mirror-short"   # across the short axis: x -> L - x
HALF_TURN = "half-turn"


def transform(point, flip_x, flip_y):
    x, y = point
    if flip_y:
        x = TABLE_LENGTH_MM - x
    if flip_x:
        y = TABLE_WIDTH_MM - y
    return [x, y]


def canonical(layout, cue):
    """Turn the layout so the cue ball sits in one chosen quarter of the table.

    Returns (layout, name of the transform, whether side spin flips). The
    quarter is the one nearest the origin; which quarter is arbitrary, that it
    is always the same one is not.
    """
    at = layout[cue]
    flip_y = at[0] > TABLE_LENGTH_MM / 2
    flip_x = at[1] > TABLE_WIDTH_MM / 2
    moved = {colour: transform(point, flip_x, flip_y)
             for colour, point in layout.items()}
    if flip_x and flip_y:
        name = HALF_TURN
    elif flip_x:
        name = MIRROR_X
    elif flip_y:
        name = MIRROR_Y
    else:
        name = "none"
    # One mirror flips handedness; two of them turn it back.
    flips_side = flip_x != flip_y
    return moved, name, flips_side


def plays(dataset=None):
    dataset = dataset or os.path.join(ROOT, "data", "dataset")
    for path in sorted(glob.glob(os.path.join(dataset, "soop_*.json"))):
        if path.endswith("_traj.json"):
            continue
        match = os.path.basename(path)[:-5]
        for play in json.load(open(path, encoding="utf-8"))["plays"]:
            yield match, play


def keep(play):
    """Why a play is not fit to learn from, or None if it is."""
    if play.get("rejected_for"):
        return play["rejected_for"][0]
    if play.get("trajectory_agrees") is not True:
        return "path does not show the label"
    route = (play.get("route") or {}).get("route")
    if not route or route == "미분류":
        return "route not named"
    if any(colour not in (play.get("layout_mm") or {}) for colour in BALLS):
        return "layout incomplete"
    if play.get("success") is None:
        return "no verdict"
    return None


def build():
    rows, dropped = [], {}
    for match, play in plays():
        reason = keep(play)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        layout, turned, flips_side = canonical(play["layout_mm"], play["cue_ball"])
        english = play.get("english")
        rows.append({
            "match": match,
            "split": "test" if match in TEST_MATCHES else "train",
            "inning": play["inning"], "shot": play["shot_number"],
            "cue": play["cue_ball"],
            "layout_mm": layout, "turned": turned,
            "route": (play["route"] or {}).get("route"),
            "route_basis": (play["route"] or {}).get("basis"),
            "cushions": play.get("cushions_before_second"),
            "scored": bool(play["success"]),
            "first_object_ball": play.get("first_object_ball"),
            "thickness": play.get("thickness"),
            "speed_ms": play.get("cue_speed_ms"),
            "english": None if english is None or not flips_side else -english,
        })
    return rows, dropped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "model.json"))
    args = parser.parse_args(argv)

    rows, dropped = build()
    json.dump(rows, open(args.out, "w", encoding="utf-8"))
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"{len(rows)} plays to learn from: {len(train)} train, {len(test)} test")
    print(f"  scoring {sum(r['scored'] for r in rows)} "
          f"({sum(r['scored'] for r in rows) / len(rows) * 100:.0f}%)")
    counts = {}
    for row in rows:
        counts[row["route"]] = counts.get(row["route"], 0) + 1
    for route, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        scored = sum(r["scored"] for r in rows if r["route"] == route)
        print(f"  {route:8s} {n:5d}   득점 {scored / n * 100:3.0f}%")
    print("dropped:")
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:34s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
