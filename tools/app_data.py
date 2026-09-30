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
from route_model import player_features  # noqa: E402

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


def opening_solved(original, marks, layout, cue, flip_x, flip_y, fps=60.0):
    """겨냥과 속도를 **첫 사건 전** 구간에서 읽고, 속도는 시뮬로 되푼다 (2026-10).

    `opening()`은 NaN을 걸러낸 배열의 1·5번째 점을 쓴다 — 공이 언제 출발했는지 안 보고,
    1~5프레임 안에 1적구에 닿는 가까운 공이면 충돌 뒤까지 섞는다. 탐색 구멍 조사에서
    프로의 겨냥·속도로 다시 친 판의 40%가 1적구부터 달랐다. 검증 쪽
    (`validate_simulator.opening()`)은 2026-09-29에 같은 식으로 고쳤다.

    * 출발 = 수구가 처음 5 mm 넘게 움직인 프레임의 앞.
    * 방향 = 출발 + 2프레임부터 첫 사건(공·쿠션) 2프레임 전까지, 길어야 24프레임.
    * 속도 = 그 창 끝에서 조언판과 같은 `spin.py` 모형(상단 2팁)이 실제와 같은 거리를
      가 있도록 이분법으로. 창 평균은 미끄럼 감속이 섞여 낮게 읽힌다.

    창을 못 잡으면 (None, None) — 부르는 쪽이 옛 읽기로 물러난다.

    ⚠️ **재 보니 옛 읽기와 같다 — 연결하지 않았다** (2026-10, `tools/check_opening_read.py`,
    창을 잡은 387판): 1적구 같음 83 → 82%, 첫 두 쿠션 300 mm 안 75 → 76%, 중앙값 75 → 66 mm.
    나머지 196판은 첫 사건이 너무 빨라 창을 못 잡는다 (공이 가깝다) — 어느 읽기든 한계다.
    프로 샷 재현 실패는 겨냥 읽기 탓이 아니다.
    """
    from src.physics.simulator import simulate_with_spin
    pts = np.asarray(original, dtype=float)
    if len(pts) < 8 or not np.isfinite(pts[0]).all():
        return None, None
    moved = np.flatnonzero(np.isfinite(pts).all(axis=1)
                           & (np.hypot(*(pts - pts[0]).T) > 5.0))
    if not len(moved):
        return None, None
    start = max(0, int(moved[0]) - 1)
    if not np.isfinite(pts[start]).all():
        start = 0
    first_event = min((int(m[0]) for m in marks if len(m) >= 2), default=len(pts))
    a = start + 2
    b = min(a + 24, first_event - 2, len(pts) - 1)
    while b > a and not np.isfinite(pts[b]).all():
        b -= 1
    if b - a < 3 or not np.isfinite(pts[a]).all():
        return None, None
    p_a = transform(pts[a], flip_x, flip_y)
    p_b = transform(pts[b], flip_x, flip_y)
    step = np.subtract(p_b, p_a)
    if np.hypot(*step) < 5.0:
        return None, None
    degrees = float(np.degrees(np.arctan2(step[1], step[0])) % 360.0)
    rad = np.radians(degrees)
    heading = np.array([np.cos(rad), np.sin(rad)])
    target = float(np.hypot(*(np.subtract(transform(pts[b], flip_x, flip_y),
                                          transform(pts[start], flip_x, flip_y)))))
    n = b - start
    balls = {c: np.array(xy, dtype=float) for c, xy in layout.items()}

    def gone(speed):
        shot = simulate_with_spin(balls, cue, tuple(heading * speed), tips_vertical=2.0,
                                  fps=fps, max_seconds=(n + 2) / fps)
        path = shot.paths[cue]
        i = min(n, len(path) - 1)
        return float(np.hypot(*(np.asarray(path[i]) - path[0])))

    guess = target / n * fps
    if not np.isfinite(guess) or guess <= 0:
        return None, None
    low, high = 0.5 * guess, 3.0 * guess
    if gone(high) < target:
        return round(degrees, 1), round(high)
    for _ in range(14):
        middle = 0.5 * (low + high)
        if gone(middle) < target:
            low = middle
        else:
            high = middle
    return round(degrees, 1), round(0.5 * (low + high))


def curve_depth(turned, hit, first_rail):
    """1적구 충돌에서 1쿠션까지, 수구가 직선에서 얼마나 벗어났나 (mm).

    **상하 당점에 가장 민감한 관측이다.** 시뮬레이터로 재 보면 좌우 당점을
    −3팁에서 +3팁까지 바꿔도 이 깊이는 5 mm밖에 안 움직이는데, 상하 당점을
    같은 폭으로 바꾸면 **236 mm** 움직인다 (2026-09-25). 천 위에서 공을 휘게
    하는 것은 좌우 회전이 아니라 **충돌 뒤 미끄럼이 구름으로 바뀌는 과정**이고,
    그것을 정하는 것이 상하 당점이기 때문이다.

    ⚠️ 솎기 전의 원본 경로에서 잰다. 32점으로 솎은 경로에는 이 구간에 점이
    네 개뿐이라 곡선이 잡히지 않는다.

    ⚠️ 깊이는 |당점|만 준다 — U자라서 끌어치기와 밀어치기가 둘 다 휜다.
    부호는 `rise`(분리각)가 주는데 그쪽은 약하다. 둘을 같이 써야 한다.
    """
    if hit is None or first_rail is None:
        return None
    way = np.asarray([p for p in turned if np.isfinite(p).all()], dtype=float)
    if len(way) < 8:
        return None
    start = int(np.argmin(np.linalg.norm(way - np.asarray(hit, float), axis=1)))
    stop = int(np.argmin(np.linalg.norm(way - np.asarray(first_rail, float), axis=1)))
    if stop - start < 5:
        return None
    seg = way[start:stop + 1]
    span = seg[-1] - seg[0]
    length = float(np.hypot(*span))
    if length < 200:
        return None
    unit = span / length
    rel = seg - seg[0]
    off = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
    return [int(round(float(off.max()))), int(round(length))]


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

    # 유형을 가리지 않는다. 한때 `OUTCOME_IN_NAME`(대회전·되돌아오기·횡단)을
    # 여기서도 뺐는데, 그 목록의 이유는 **유형 이름 맞히기 실험**에서 "이름이
    # 결과를 담고 있어 채점할 수 없다"는 것이지(`route_model.py`), 이 풀과는
    # 상관이 없다. 이 풀이 하는 일은 "닮은 배치에서 프로가 무엇을 쳤나"를
    # 모으는 것이고, 프로가 대회전을 쳤으면 그건 센다.
    #
    # 2026-09-24에 재 보니 진짜 갈라짐이었다. 학습 쪽(`learn_choices.py`)은
    # 이 목록을 쓰지 않아 세 유형에도 표를 주는데(후보 400판에서 643번),
    # 화면은 **무조건 0명**을 주고 있었다. 후보 중 이웃 0명의 비율도 학습
    # 45.5% 대 화면 50.9%로 달랐다 — `이웃프로수 +0.95`가 배운 분포와 화면이
    # 먹이는 분포가 다르다는 뜻이다. 대회전은 프로가 10.7%나 치는 공략이다.
    rows = json.load(open(args.data, encoding="utf-8"))
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
    events, spin, rise = {}, {}, {}
    for name in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json"))):
        if name.endswith("index.json"):
            continue
        match = os.path.basename(name)[:-5]
        try:
            body = json.load(open(name, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for play in (body["plays"] if isinstance(body, dict) and "plays" in body else body):
            at = (match, play.get("inning"), play.get("shot_number"))
            events[at] = play.get("events") or []
            # 첫 쿠션에서 거울 반사로부터 몇 도 틀어졌나. 회전이 얼마나 걸렸는지를
            # 재는 유일한 값이고, 반사 표를 **회전이 적은 것만 골라** 다시 재려면
            # 이것이 있어야 한다 (2026-09-25). 부호는 거울 뒤집기에서 바뀌지만
            # 쓰는 쪽은 크기만 보므로 그대로 옮긴다.
            spin[at] = play.get("spin_x")
            # 밀어치기/끌어치기. 파이프라인이 재 놓고도 아무 데도 안 쓰던 값이고
            # (`follow_draw`의 주석: UNVALIDATED), ④ 당점에 대한 영상의 유일한
            # 증거다. 2026-09-25에 실어 나르기 시작했다.
            rise[at] = play.get("spin_y")

    # 합동 맞추기로 되찾은 당점. `tools/fit_tip.js --export`가 적어 둔다.
    # 여기서 계산하지 않는 이유: 판당 625번을 쳐야 하는 일이라 파이썬 경로에
    # 넣으면 조언판을 다시 만들 때마다 십수 분이 든다.
    #
    # ④(당점)에 대한 **첫 실측 자료**다. 그전에는 팁이 아예 없어서, 조언판이
    # 이웃 프로의 당점을 옮겨 올 수 없었다.
    tips = {}
    tip_file = os.path.join(ROOT, "data", "tips.json")
    if os.path.exists(tip_file):
        try:
            tips = json.load(open(tip_file, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            tips = {}

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
        #
        # ⚠️ 2026-09-30: 그런데 `model_dataset.py`가 **이미** 거울 한 번이면 뒤집어서
        # 넘긴다 (`flips_side`). 2026-09-23(4b45234)에 여기서 한 번 더 뒤집어, 거울 한 번인
        # 판의 면이 도로 원본 틀로 돌아가 있었다. 옆/뒤돌리기 규칙("맞힌 면 = 도는 방향")에
        # 대 보니 거울 0·2번 판은 10~12%, 1번 판은 88%로 정확히 갈렸다
        # (`tools/ceiling_path.py`). 조언판의 이웃 투표가 절반의 판에서 틀린 면을 셌다.
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
            # 쿠션과 1적구 접촉이 궤적의 어디였나. 궤적은 이미 캐노니컬 프레임으로
            # 돌려 두었으므로, 프레임 번호로 그 위에서 집어내면 된다.
            "rails": rail_points(original, marks, flip_x, flip_y),
            "hit": ball_point(original, marks, flip_x, flip_y),
            # 되찾은 당점 [좌우, 상하] (팁)과 믿을 만한가.
            # `tip_ok`가 거짓이면 맞추기가 ±3팁 끝에 걸린 것이다 — 어떤 물리적
            # 당점으로도 설명되지 않는 판이라 그 값은 모형 오차의 쓰레기통이다.
            # 2026-09-25 실측: 안쪽 판은 2쿠션 오차가 91→83 mm(3.0 표준편차)로
            # 나아지는데, 끝에 걸린 판은 153→135 mm(0.4 표준편차)로 무의미하다.
            "tip": (tips.get(f"{row['match'].replace('soop_', '')}:{row['inning']}:{row['shot']}")
                    or {}).get("tip"),
            "tip_ok": (tips.get(f"{row['match'].replace('soop_', '')}:{row['inning']}:{row['shot']}")
                       or {}).get("ok"),
            # 충돌→1쿠션 구간의 곡선 깊이와 그 구간 길이 [깊이, 길이] (mm).
            # 상하 당점에 대한 가장 강한 관측이다 — curve_depth()의 설명 참조.
            "curve": curve_depth(turned, ball_point(original, marks, flip_x, flip_y),
                                 (rail_points(original, marks, flip_x, flip_y) or [None])[0]),
            "aim": opening(turned)[0],
            "speed": opening(turned)[1],
            # 두께는 절반쯤의 플레이에만 있다 (적구가 떠나는 각을 읽을 만큼
            # 깨끗하게 찍힌 경우). 없는 것은 없는 대로 두고, 순위에서는 있는
            # 이웃만 세어 평균을 낸다 - 없는 것을 0으로 채우면 프로가 전부
            # 얇게 친 것처럼 보인다.
            "thick": (None if row.get("thickness") is None
                      else round(float(row["thickness"]), 3)),
            "match": row["match"].replace("soop_", ""),
            # 판을 경기별 JSON과 다시 이을 수 있게 남긴다. 이것이 없어서
            # 2026-09-25에 "회전이 적은 것만 골라 반사 표를 재기"가 막혔다.
            "inning": row["inning"],
            "shot": row["shot"],
            # 첫 쿠션의 거울 대비 틀어짐(도). 크기만 쓴다.
            "spin": (None if spin.get((row["match"], row["inning"], row["shot"])) is None
                     else round(abs(float(spin[(row["match"], row["inning"], row["shot"])])), 2)),
            # 밀어치기(+) / 끌어치기(−). 거울 뒤집기는 위아래를 바꾸지 않으므로
            # 부호를 그대로 옮긴다.
            "rise": (None if rise.get((row["match"], row["inning"], row["shot"])) is None
                     else round(float(rise[(row["match"], row["inning"], row["shot"])]), 3)),
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
