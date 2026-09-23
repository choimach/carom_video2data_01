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


def _at(original, frame, flip_x, flip_y):
    """이벤트의 프레임 번호를 궤적 위의 자리로.

    ⚠️ 두 가지에 걸린 적이 있다. (1) events의 프레임은 **플레이 시작 기준
    상대값**이지 영상 전체의 절대 프레임이 아니다. (2) 그러므로 NaN을 걸러낸
    뒤의 배열에 대고 세면 안 된다 — 거른 만큼 인덱스가 밀린다. 원본 배열에서
    집어내고, 그 점만 캐노니컬 프레임으로 돌린다.
    """
    if frame is None:
        return None
    i = int(frame)
    if i < 0 or i >= len(original):
        return None
    point = original[i]
    if not np.isfinite(point).all():
        return None
    return [int(round(v)) for v in transform(point, flip_x, flip_y)]


def rail_points(original, events, flip_x, flip_y):
    """수구가 쿠션을 맞은 자리들, 순서대로.

    쿠션이 **어디서** 일어났는지는 파이프라인이 이미 알고 있다. 그것을 옮겨
    적지 않으면 읽는 쪽은 32점으로 솎인 길에서 되짚어야 하고 대개 놓친다 —
    점 간격이 200 mm인데 쿠션 띠는 49 mm다.
    """
    out = []
    for item in events:
        if len(item) < 2 or item[1] != "cushion":
            continue
        at = _at(original, item[0], flip_x, flip_y)
        if at:
            out.append(at)
    return out or None


def ball_point(original, events, flip_x, flip_y):
    """수구가 제1적구를 맞은 자리. 반사각을 재는 기준점이다."""
    for item in events:
        if len(item) >= 2 and item[1] == "ball":
            return _at(original, item[0], flip_x, flip_y)
    return None


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

    # 쿠션이 **어디서** 일어났는지는 파이프라인이 이미 알고 있다 (경기별 JSON의
    # events가 프레임으로 들고 있다). 그것을 여기서 옮겨 적지 않으면, 읽는 쪽은
    # 32점으로 솎인 길에서 쿠션 자리를 되짚어야 하고 대개 놓친다.
    # 선수가 정한 궤적 안의 차례 (2026-09-21)에서 2번과 5번이 바로 이 값이다.
    events = {}
    for name in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json"))):
        if name.endswith("index.json"):
            continue
        match = os.path.basename(name)[:-5]
        try:
            body = json.load(open(name, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for play in (body["plays"] if isinstance(body, dict) and "plays" in body else body):
            events[(match, play.get("inning"), play.get("shot_number"))] = \
                play.get("events") or []

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
        marks = events.get((row["match"], row["inning"], row["shot"]), [])
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
        #
        # ★거울 한 번은 손을 바꾼다. 배치는 캐노니컬 구역으로 뒤집어 저장하면서
        # 면은 원본 프레임의 값을 그대로 쓰고 있었다 — 거울이 한 번 걸린 판에서
        # 왼쪽 면이 오른쪽 면으로 저장된 것이다. 절반쯤이 그랬고, 그래서 면이
        # 동전 던지기로 나왔다. 거울 두 번(180도 회전)은 손을 바꾸지 않는다.
        struck = row.get("struck_side")
        if struck is not None and (flip_x != flip_y):
            struck = -struck
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
            # 쿠션과 1적구 접촉이 궤적의 어디였나. 궤적은 이미 캐노니컬 프레임으로
            # 돌려 두었으므로, 프레임 번호로 그 위에서 집어내면 된다.
            "rails": rail_points(original, marks, flip_x, flip_y),
            "hit": ball_point(original, marks, flip_x, flip_y),
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
