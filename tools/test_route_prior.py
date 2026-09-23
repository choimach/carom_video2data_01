"""유형의 프로 기본 비율을 순위에 넣는 것이 나은가 — 말 바꾸기 전에 잰다.

선수 (2026-09-23): *"뱅크샷 이런 선택은 안함."* 그 판의 후보 12개 중 넷이
뱅크샷이었다. 세어 보니 우리 후보 목록이 프로와 많이 다른 모양이다:

    유형        프로가 친 것   우리 후보
    횡단            0.6%      10.7%   ← 17.8배
    걸어치기         7.0%      17.9%   ← 2.5배
    뱅크샷          12.4%      19.9%   ← 1.6배
    뒤돌리기        26.2%      10.7%   ← 0.4배
    옆돌리기        23.4%       9.0%   ← 0.4배

프로가 치는 절반(뒤돌리기 + 옆돌리기)이 우리 후보에서는 20%뿐이다. 전방위
훑기가 쿠션 먼저 맞는 길을 잘 잡는 대가다.

`이웃프로수`가 이미 비슷한 일을 하지만 **성기다** — 이웃 40명뿐이라 후보의
45%가 0이다. 기본 비율은 촘촘하다. 둘이 겹치는지 보완하는지는 재야 안다.

⚠️ 기본 비율은 **학습 경기에서만** 센다. 전체에서 세면 시험 경기의 답을 미리
본 셈이 된다.

    ~/.venvs/carom/bin/python tools/test_route_prior.py
"""

import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from learn_choices import as_arrays, fit, load, where_it_landed  # noqa: E402
from test_neighbour_term import K  # noqa: E402


def neighbours_of(rounds):
    """이웃 프로 둘을 붙인다 — test_neighbour_term.py와 같은 방식."""
    import json

    from route_model import player_features
    from test_neighbour_term import bucket_of

    model = json.load(open(os.path.join(ROOT, "data", "model.json"), encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    F = np.array([player_features(r) for r in rows], dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    picked = [bucket_of(r) for r in rows]
    match = np.array([r["match"] for r in rows])
    where = {f"{r['match']}:{r['inning']}:{r['shot']}": i for i, r in enumerate(rows)}

    out = []
    for arrays, chose, one in rounds:
        i = where.get(one["id"])
        if i is None:
            continue
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        votes, wins = {}, {}
        for j in np.argsort(gap)[:K]:
            key = picked[j]
            if not key:
                continue
            votes[key] = votes.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + (1 if rows[j].get("scored") else 0)
        seat = one["layout"]
        cue_at = np.array(seat[one["cue"]], dtype=float)
        reach = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue_at))
                 for c, xy in seat.items() if c != one["cue"]}
        add = []
        for branch in one["found"]:
            is_near = reach.get(branch["first"]) == min(reach.values())
            key = f"{branch['route']}|{is_near}|{branch['face']}"
            n = votes.get(key, 0)
            add.append([math.log1p(n), (wins.get(key, 0) + 1) / (n + 2)])
        out.append((np.hstack([arrays, np.array(add, dtype=float)]), chose, one))
    return out


def main():
    kept = neighbours_of(as_arrays(load()))
    print(f"채점할 판 {len(kept)}개 (이웃 {K}명까지 붙인 것)\n")

    matches = sorted({one.get("match") for _a, _c, one in kept})
    folds = min(10, len(matches))
    fold_of = {m: i % folds for i, m in enumerate(matches)}

    places = {"기본 비율 없이": [], "기본 비율 더해서": []}
    for fold in range(folds):
        train = [i for i, (_a, _c, one) in enumerate(kept)
                 if fold_of[one.get("match")] != fold]
        held = [i for i, (_a, _c, one) in enumerate(kept)
                if fold_of[one.get("match")] == fold]
        if not train or not held:
            continue
        # 기본 비율은 **학습 경기에서 프로가 실제로 고른 것**으로만 센다.
        chose = Counter(kept[i][2]["chose"].split("|")[0] for i in train)
        total = sum(chose.values())
        prior = {r: math.log((n + 1) / (total + len(chose))) for r, n in chose.items()}
        floor = math.log(1 / (total + len(chose)))

        def rows_of(i, join):
            if not join:
                return kept[i][0]
            column = np.array([[prior.get(b["route"], floor)]
                               for b in kept[i][2]["found"]], dtype=float)
            return np.hstack([kept[i][0], column])

        for name, join in (("기본 비율 없이", False), ("기본 비율 더해서", True)):
            weight = fit([(rows_of(i, join), kept[i][1], kept[i][2]) for i in train])
            for i in held:
                places[name].append(where_it_landed(rows_of(i, join), kept[i][1], weight))

    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)")
    for name in places:
        p = np.array(places[name], dtype=float)
        print(f"  {name:<14} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
              f" · 자리 중앙값 {np.median(p):4.1f}")

    a = np.array(places["기본 비율 없이"])
    b = np.array(places["기본 비율 더해서"])
    better, worse = int(np.sum(b < a)), int(np.sum(b > a))
    if better + worse:
        away = abs(better - (better + worse) / 2) / math.sqrt((better + worse) * 0.25)
        print(f"\n  올라간 판 {better} · 내려간 판 {worse} · 비긴 판 {len(a) - better - worse}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 넣는 것이 낫다" if away > 2 and better > worse
                 else ("  → 빼는 것이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
