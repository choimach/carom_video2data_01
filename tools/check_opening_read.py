"""조언판 자료의 겨냥·속도 읽기 — 옛 것(`app_data.opening`)과 새 것(`opening_solved`)을 견준다.

두 읽기로 프로 샷을 `sim.js`에서 다시 쳐서 (당점은 격자에서 첫 두 쿠션이 가장 맞는 것)
1적구가 같은지, 첫 두 쿠션이 프로 자리 300 mm 안인지 센다. 노드 쪽은 `tools/replay_reads.js`.

    ~/.venvs/carom/bin/python tools/check_opening_read.py [판 수=600]
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from app_data import ball_point, opening, opening_solved, rail_points  # noqa: E402
from model_dataset import transform  # noqa: E402


def main(argv):
    limit = int(argv[0]) if argv else 600
    rows = json.load(open(os.path.join(ROOT, "data", "model.json"), encoding="utf-8"))
    step = max(1, len(rows) // limit)
    rows = rows[::step][:limit]
    paths = {os.path.basename(n)[:-9]: np.load(n)
             for n in glob.glob(os.path.join(ROOT, "data", "dataset", "*_traj.npz"))}
    events = {}
    for name in glob.glob(os.path.join(ROOT, "data", "dataset", "*.json")):
        if name.endswith("index.json"):
            continue
        body = json.load(open(name, encoding="utf-8"))
        for play in (body["plays"] if isinstance(body, dict) and "plays" in body else body):
            events[(os.path.basename(name)[:-5], play.get("inning"), play.get("shot_number"))] = \
                play.get("events") or []
    out = []
    for row in rows:
        bundle = paths.get(row["match"])
        key = f"i{row['inning']:03d}s{row['shot']:02d}_{row['cue']}"
        if bundle is None or key not in bundle:
            continue
        original = bundle[key]
        raw = np.array(row["layout_mm"][row["cue"]], dtype=float)
        flip_y = bool(abs(original[0][0] - raw[0]) > 1.0)
        flip_x = bool(abs(original[0][1] - raw[1]) > 1.0)
        turned = [transform(p, flip_x, flip_y) for p in original if np.isfinite(p).all()]
        marks = events.get((row["match"], row["inning"], row["shot"]), [])
        rails = rail_points(original, marks, flip_x, flip_y) or []
        if len(rails) < 2 or not row.get("first_object_ball"):
            continue
        old = opening(turned)
        new = opening_solved(original, marks, row["layout_mm"], row["cue"], flip_x, flip_y)
        if old[0] is None:
            continue
        out.append({"id": f"{row['match']}:{row['inning']}:{row['shot']}", "cue": row["cue"],
                    "layout": row["layout_mm"], "rails": rails, "first": row["first_object_ball"],
                    "near": bool(ball_point(original, marks, flip_x, flip_y)),
                    "old": list(old), "new": list(new) if new[0] is not None else None})
    tmp = os.path.join(ROOT, "build", "_opening_reads.json")
    json.dump(out, open(tmp, "w"))
    got_new = sum(1 for o in out if o["new"])
    print(f"판 {len(out)}개 · 새 읽기가 창을 잡은 판 {got_new}")
    subprocess.run(["node", os.path.join(ROOT, "tools", "replay_reads.js"), tmp], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
