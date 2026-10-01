"""Score every naming rule against the plays the player named himself.

He is learning the game too, so nothing he says is a rule - it is a hypothesis,
and `data/labels.json` is what decides between hypotheses. This prints the
current rule's score and, beside it, every rule that has been tried and dropped,
so a new idea has to beat them on the record rather than on how good it sounds.

Run it before changing how routes are named, and again after. A rule that scores
worse than the one in place is not an improvement no matter how well it argues.

    ~/.venvs/carom/bin/python tools/rules_check.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LABELS = os.path.join(ROOT, "data", "labels.json")
DATASET = os.path.join(ROOT, "data", "dataset")
SHORT = ("left", "right")


def labelled():
    """The named plays, each with what the pipeline measured for it."""
    named = json.load(open(LABELS, encoding="utf-8"))["labels"]
    wanted = {(row["match"], row["inning"], row["shot"]): row["route"] for row in named}
    out = []
    for key, route in wanted.items():
        match, inning, shot = key
        path = os.path.join(DATASET, f"{match}.json")
        if not os.path.exists(path):
            continue
        for play in json.load(open(path, encoding="utf-8"))["plays"]:
            if play["inning"] == inning and play["shot_number"] == shot:
                play["_match"] = match
                out.append((route, play))
                break
    return out


def by_circuit(play):
    """Dropped 2026-10-02: the face struck against the circuit round the table's middle
    (signed area of the whole path). Wrong on short shots — see by_family."""
    rails = (play.get("route") or {}).get("rails") or []
    if not rails or play.get("struck_side") is None or play.get("circuit") is None:
        return None
    same = (play["struck_side"] < 0) == (play["circuit"] > 0)
    short = rails[0] in SHORT
    return ("앞돌리기" if short else "옆돌리기") if same else ("빗겨치기" if short else "뒤돌리기")


def by_drift(play):
    """Dropped: how far the second cushion carried along the line of the shot."""
    rails = (play.get("route") or {}).get("rails") or []
    away = play.get("away_mm")
    if not rails or away is None:
        return None
    short = rails[0] in SHORT
    if away < 300.0:
        return "앞돌리기" if short else "옆돌리기"
    return "빗겨치기" if short else "뒤돌리기"


def by_rails(play):
    """Dropped: the rail sequence alone."""
    rails = (play.get("route") or {}).get("rails") or []
    if not rails:
        return None
    return "뒤돌리기" if rails[0] in SHORT else "옆돌리기"


_TRAJ = {}


def _points(play):
    """수구 궤적과 사건 자리 — 사건 프레임은 플레이 시작 기준 (app_data._at 참조)."""
    import numpy as np
    match = play.get("_match")
    if match not in _TRAJ:
        f = os.path.join(DATASET, f"{match}_traj.npz")
        _TRAJ[match] = np.load(f) if os.path.exists(f) else None
    bundle = _TRAJ[match]
    key = f"i{play['inning']:03d}s{play['shot_number']:02d}_{play.get('cue_ball')}"
    if bundle is None or key not in bundle:
        return None, None
    path = bundle[key]
    events = [(int(e[0]), e[1], e[2] if len(e) > 2 else None) for e in (play.get("events") or [])]
    return path, events


def _family(play, right):
    rails = (play.get("route") or {}).get("rails") or []
    if not rails or play.get("struck_side") is None or right is None:
        return None
    same = (play["struck_side"] < 0) == right
    short = rails[0] in SHORT
    return ("앞돌리기" if short else "옆돌리기") if same else ("빗겨치기" if short else "뒤돌리기")


def by_area_to_second(play):
    """시험 (2026-10-02): 지금 규칙의 도는 방향을 2적구에 닿을 때까지의 궤적으로만."""
    import numpy as np
    path, events = _points(play)
    if path is None:
        return None
    balls = [e for e in events if e[1] == "ball"]
    end = balls[1][0] if len(balls) >= 2 else len(path) - 1
    seg = path[:end + 1]
    seg = seg[np.isfinite(seg).all(axis=1)]
    if len(seg) < 5:
        return None
    about = seg - np.array([1422.0, 711.0])
    area = float(np.sum(about[:-1, 0] * about[1:, 1] - about[:-1, 1] * about[1:, 0]))
    return _family(play, area > 0)


def by_family(play):
    """The rule in place (2026-10-02): the face struck against the way the path TURNS —
    1적구 → 쿠션들 → 2적구 자리를 이은 선이 꺾이는 방향의 합, 동수면 by_circuit.
    선수가 짚은 짧은 샷 둘("짧게 뒤돌리기", "앞돌리기 짧게")에서 탁자 가운데 둘레 넓이가
    틀렸다. pipeline.turn_sign · route.js turnIsRight와 같다."""
    import numpy as np
    path, events = _points(play)
    if path is None:
        return None
    balls = [e for e in events if e[1] == "ball"]
    if not balls:
        return None
    first = balls[0][0]
    second = balls[1][0] if len(balls) >= 2 else None
    frames = [first] + [e[0] for e in events if e[1] == "cushion" and e[0] > first
                        and (second is None or e[0] < second)]
    if second is not None:
        frames.append(second)
    pts = [path[f] for f in frames if 0 <= f < len(path) and np.isfinite(path[f]).all()]
    if len(pts) < 3:
        return None
    turn = 0.0
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        u, v = b - a, c - b
        turn += np.sign(u[0] * v[1] - u[1] * v[0])
    if turn == 0:
        # 좌우로 꺾인 수가 같으면 판정을 못 한다 — 탁자 가운데 둘레 넓이로 물러난다.
        return by_circuit(play)
    return _family(play, turn > 0)


TURNS = ("뒤돌리기", "옆돌리기", "앞돌리기", "빗겨치기")
RULES = (("맞힌 면 vs 궤적 꺾임 (지금, 2026-10-02)", by_family),
         ("맞힌 면 vs 탁자 가운데 둘레 (버림 2026-10-02)", by_circuit),
         ("2쿠션 진행 거리 (버림)", by_drift),
         ("쿠션 순서 (버림)", by_rails),
         ("둘레 넓이를 2적구까지 (버림)", by_area_to_second))


def main():
    rows = labelled()
    turns = [(route, play) for route, play in rows if route in TURNS]
    print(f"선수가 이름 붙인 플레이 {len(rows)}개 · 그중 네 가지 돌리기 {len(turns)}개\n")
    for name, rule in RULES:
        right = sum(1 for route, play in turns if rule(play) == route)
        print(f"  {name:26s} {right:2d}/{len(turns)}  {right / len(turns) * 100:3.0f}%")

    print("\n지금 규칙이 틀리는 것:")
    for route, play in turns:
        got = by_family(play)
        if got != route:
            rails = "-".join((play.get("route") or {}).get("rails") or [])
            print(f"  {route} → {got}   면 {play.get('struck_side'):.0f} · "
                  f"도는 방향 {play.get('circuit')} · 쿠션 {rails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
