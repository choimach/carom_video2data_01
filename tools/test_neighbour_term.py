"""이웃 프로의 수를 순위에 넣는 것이 나은가 — 말 바꾸기 전에 잰다.

2026-09-23에 이웃 투표를 점수에서 뺐다. 근거는 "닮은 배치에서 프로가 어느 길을
고르는지 맞히는 비율이 28.0%인데 배치와 무관할 때가 26.7%"였다. 그런데 선수가
바로 그날 반대 증거를 냈다 — 프로 0/40명짜리가 1등에 올라왔고, 그가 말한 실제
순위는 프로 수 차례(7명 > 1명 > 0명) 그대로였다.

둘은 **다른 질문**이다:

* 뺄 때 잰 것 — "이웃들이 **무엇을 고를지 맞힐 수 있나**" (거의 못 맞힌다)
* 지금 묻는 것 — "이웃 중 몇 명이 골랐는지가 **후보를 줄 세우는 데 쓸모 있나**"

두 번째가 첫 번째보다 쉬울 수 있다. 어느 길이 가장 많이 뽑히는지는 못 맞혀도,
**한 명도 고르지 않은 길을 아래로 내리는 것**만으로 순위는 좋아질 수 있다.

    ~/.venvs/carom/bin/python tools/test_neighbour_term.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from learn_choices import as_arrays, fit, load, where_it_landed  # noqa: E402
from route_model import player_features  # noqa: E402

K = 40


def bucket_of(row):
    """모델 자료 한 줄이 고른 것 — 조언판의 후보 열쇠와 같은 모양으로."""
    cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
    gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue))
            for c, xy in row["layout_mm"].items() if c != row["cue"]}
    first = row.get("first_object_ball")
    if not first or first not in gaps:
        return None
    near = gaps[first] == min(gaps.values())
    side = row.get("struck_side")
    if side is None:
        return None
    return f"{row['route']}|{near}|{'left' if side > 0 else 'right'}"


def main():
    model = json.load(open(os.path.join(ROOT, "data", "model.json"), encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    F = np.array([player_features(r) for r in rows], dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    picked = [bucket_of(r) for r in rows]
    match = np.array([r["match"] for r in rows])
    where = {f"{r['match']}:{r['inning']}:{r['shot']}": i for i, r in enumerate(rows)}

    data = as_arrays(load())
    print(f"모델 자료 {len(rows)}줄 · 채점할 판 {len(data)}개 · 이웃 {K}명\n")

    # 판마다 후보별 "이웃 중 몇 명이 이 길을 골랐나"와 "그중 몇이 넣었나".
    extra = []
    kept = []
    for arrays, chose, one in data:
        i = where.get(one["id"])
        if i is None:
            continue
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        near = np.argsort(gap)[:K]
        votes, wins = {}, {}
        for j in near:
            key = picked[j]
            if not key:
                continue
            votes[key] = votes.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + (1 if rows[j].get("scored") else 0)
        # 후보의 열쇠는 **공 색깔**로 되어 있는데, 이웃과 견주려면 색깔이 아니라
        # **가까운 공이냐 먼 공이냐**여야 한다 — 이웃의 '빨간공'은 이 배치의
        # '빨간공'과 아무 상관이 없다. 여기서 옮겨 준다.
        seat = one["layout"]
        cue_at = np.array(seat[one["cue"]], dtype=float)
        reach = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue_at))
                 for c, xy in seat.items() if c != one["cue"]}
        add = []
        for branch in one["found"]:
            is_near = reach.get(branch["first"]) == min(reach.values())
            key = f"{branch['route']}|{is_near}|{branch['face']}"
            chosen = votes.get(key, 0)
            add.append([np.log1p(chosen),
                        (wins.get(key, 0) + 1) / (chosen + 2)])
        extra.append(np.array(add, dtype=float))
        kept.append((arrays, chose, one))

    print(f"이웃을 붙일 수 있었던 판 {len(kept)}개")
    counts = np.concatenate([e[:, 0] for e in extra])
    print(f"후보의 {float(np.mean(counts == 0)):.0%}는 이웃 중 아무도 고른 적이 없다\n")

    matches = sorted({one.get("match") for _a, _c, one in kept})
    folds = min(10, len(matches))
    fold_of = {m: i % folds for i, m in enumerate(matches)}

    places = {"이웃 없이": [], "이웃 더해서": []}
    for fold in range(folds):
        train = [i for i, (_a, _c, one) in enumerate(kept)
                 if fold_of[one.get("match")] != fold]
        held = [i for i, (_a, _c, one) in enumerate(kept)
                if fold_of[one.get("match")] == fold]
        if not train or not held:
            continue
        for name, join in (("이웃 없이", False), ("이웃 더해서", True)):
            def rows_of(i):
                return np.hstack([kept[i][0], extra[i]]) if join else kept[i][0]
            weight = fit([(rows_of(i), kept[i][1], kept[i][2]) for i in train])
            for i in held:
                places[name].append(where_it_landed(rows_of(i), kept[i][1], weight))

    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)")
    for name in ("이웃 없이", "이웃 더해서"):
        p = np.array(places[name], dtype=float)
        print(f"  {name:<10} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
              f" · 자리 중앙값 {np.median(p):4.1f}")

    a = np.array(places["이웃 없이"]); b = np.array(places["이웃 더해서"])
    better, worse = int(np.sum(b < a)), int(np.sum(b > a))
    if better + worse:
        away = abs(better - (better + worse) / 2) / np.sqrt((better + worse) * 0.25)
        print(f"\n  이웃을 더해 올라간 판 {better} · 내려간 판 {worse}"
              f" · 비긴 판 {len(a) - better - worse}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 넣는 것이 낫다" if away > 2 and better > worse
                 else ("  → 빼는 것이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
