"""같은 배치에서 프로가 무엇 **대신** 무엇을 골랐는지 배운다.

`tools/enumerate_alternatives.js`가 프로 플레이의 배치마다 그날 실제로 있었던
길을 전부 세워 두었다. 그중 하나가 프로가 친 것이고 나머지가 버린 것이다 —
영상만으로는 얻을 수 없었던 절반이다.

여기서 묻는 것은 하나다: **버린 것과 고른 것을 가르는 것이 무엇인가.**

지금 조언판의 순위는 손으로 정한 식이다 —
`(고른 프로 수 + 0.5) × 득점률 × 두께일치 × 조준여유 ÷ (1 + 1적구이동/4000)`.
그 식이 맞는지 아무도 잰 적이 없다. 이 도구가 잰다: 같은 자로 손 식과 배운
식을 나란히 놓고, **프로가 실제로 고른 길이 목록에서 몇 번째에 오는지** 본다.

    ~/.venvs/carom/bin/python tools/learn_choices.py

train/test는 model.json이 이미 나눠 둔 경기 단위 split을 그대로 쓴다. 같은
경기가 양쪽에 걸치면 같은 선수의 같은 버릇을 외우고 맞혔다고 하게 된다.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FOUND = os.path.join(ROOT, "data", "alternatives.jsonl")
NEIGHBOURS = 40
OUT = os.path.join(ROOT, "data", "choice_weights.json")

# 한 갈래를 설명하는 값들. 전부 그 갈래 자체에서 나오는 것이고, 프로가 무엇을
# 골랐는지는 절대 들어가지 않는다 - 그것이 맞혀야 할 답이다.
def features(branch, layout=None, cue=None):
    side, up = branch["side"], branch["up"]
    tips = math.hypot(side, up)
    return [
        math.log1p(branch["room"]),          # 조준 여유 - 넓을수록 쉽다
        branch["thickness"],                 # 두께
        branch["strength"] / 7.0,            # 세기
        min(branch["rails"], 6) / 6.0,       # 쿠션 수
        math.log1p(branch["pushed"]) / 10.0,  # 1적구가 굴러간 거리
        math.log1p(branch["lines"]) / 5.0,   # 그 갈래가 얼마나 두툼한가
        tips / 3.0,                          # 준 회전의 양
        1.0 if up > 0.3 else 0.0,            # 상단 당점인가
        # 이웃 프로 둘. 2026-09-23에 한 번 뺐다가 되돌렸다 — 뺄 때는 "이웃이
        # 무엇을 고를지 맞힐 수 있나"를 쟀는데(거의 못 맞힌다), 쓸모는 다른 데
        # 있었다. **후보의 45%는 이웃 중 아무도 고른 적이 없고**, 우승자를 맞히지
        # 않아도 그런 길을 가라앉히는 것만으로 순위가 좋아진다.
        # 넣으면 1등 24.7% → 33.4%, 3등 안 49.9% → 60.0% (10.8 표준편차).
        math.log1p(branch.get("chosen", 0)),
        branch.get("rate", 0.5),
        1.0,                                 # 기준선
    ]


NAMES = ["여유", "두께", "세기", "쿠션수", "1적구이동", "줄두께", "회전량", "상단당점",
         "이웃프로수", "이웃득점률", "기준"]


def load(limit=None):
    rounds = []
    if not os.path.exists(FOUND):
        return rounds
    with open(FOUND, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                one = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 우리 탐색이 프로의 길을 못 찾은 판은 쓸 수 없다. 그건 그가 버린
            # 것이 아니라 우리가 놓친 것이라, 버림의 근거가 되지 못한다.
            if not one.get("reached") or len(one.get("found") or []) < 2:
                continue
            rounds.append(one)
            if limit and len(rounds) >= limit:
                break
    return rounds


def with_neighbours(rounds):
    """판마다 후보에 "이웃 중 몇 명이 이 길을 골랐나"를 붙인다.

    열쇠를 공 색깔이 아니라 **가까운 공이냐 먼 공이냐**로 맞춰야 한다 — 이웃의
    '빨간공'은 이 배치의 '빨간공'과 아무 상관이 없다.
    """
    from route_model import player_features
    model = json.load(open(os.path.join(ROOT, "data", "model.json"), encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    F = np.array([player_features(r) for r in rows], dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    match = np.array([r["match"] for r in rows])
    seat = {f"{r['match']}:{r['inning']}:{r['shot']}": i for i, r in enumerate(rows)}

    def bucket(row):
        cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
        gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue))
                for c, xy in row["layout_mm"].items() if c != row["cue"]}
        first, side = row.get("first_object_ball"), row.get("struck_side")
        if not first or first not in gaps or side is None:
            return None
        return f"{row['route']}|{gaps[first] == min(gaps.values())}|{'left' if side > 0 else 'right'}"

    picked = [bucket(r) for r in rows]
    out = []
    for one in rounds:
        i = seat.get(one["id"])
        if i is None:
            continue
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        votes, wins = {}, {}
        for j in np.argsort(gap)[:NEIGHBOURS]:
            key = picked[j]
            if not key:
                continue
            votes[key] = votes.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + (1 if rows[j].get("scored") else 0)
        at = one["layout"]
        cue_at = np.array(at[one["cue"]], dtype=float)
        reach = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue_at))
                 for c, xy in at.items() if c != one["cue"]}
        for branch in one["found"]:
            key = f"{branch['route']}|{reach.get(branch['first']) == min(reach.values())}|{branch['face']}"
            branch["chosen"] = votes.get(key, 0)
            branch["rate"] = (wins.get(key, 0) + 1) / (branch["chosen"] + 2)
        out.append(one)
    return out


def as_arrays(rounds):
    """한 판을 (갈래별 특징, 고른 것의 자리)로."""
    out = []
    for one in rounds:
        rows = [features(b, one["layout"], one["cue"]) for b in one["found"]]
        chose = [i for i, b in enumerate(one["found"]) if b["key"] == one["chose"]]
        if not chose:
            continue
        out.append((np.array(rows, dtype=float), chose[0], one))
    return out


def fit(train, rounds_of_weight=400, step=0.25):
    """고른 것의 점수가 버린 것들보다 높아지도록 가중치를 민다.

    갈래 수가 판마다 다르므로 softmax로 한 판 전체를 한꺼번에 본다 - 순위를
    배우는 데 맞는 모양이고, 갈래가 많은 판이 제멋대로 무거워지지 않는다.
    """
    width = train[0][0].shape[1]
    weight = np.zeros(width)
    for _ in range(rounds_of_weight):
        grad = np.zeros(width)
        for rows, chose, _one in train:
            scores = rows @ weight
            scores -= scores.max()
            share = np.exp(scores)
            share /= share.sum()
            grad += rows[chose] - share @ rows
        weight += step * grad / len(train)
    return weight


def where_it_landed(rows, chose, weight):
    order = np.argsort(-(rows @ weight))
    return int(np.where(order == chose)[0][0]) + 1


def by_hand(one, index):
    """지금 조언판이 쓰는 손 식 - 이웃 수와 득점률은 여기 없으므로 뺀 형태."""
    b = one["found"][index]
    return b["room"] / (1.0 + b["pushed"] / 4000.0)


def report(name, places, total_branches):
    places = np.array(places, dtype=float)
    print(f"  {name:<12} 1등 {np.mean(places == 1):5.0%}"
          f" · 3등 안 {np.mean(places <= 3):5.0%}"
          f" · 자리 중앙값 {np.median(places):4.1f}"
          f" / 갈래 {np.median(total_branches):4.1f}개")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folds", type=int, default=10, help="경기 단위 겹 수")
    args = parser.parse_args()

    rounds = load(args.limit)
    if len(rounds) < 30:
        print(f"쓸 수 있는 판이 {len(rounds)}개뿐입니다 — "
              "tools/enumerate_alternatives.js를 먼저 끝까지 돌리세요.")
        return 1

    data = as_arrays(with_neighbours(rounds))

    # 경기 단위 k-겹. model.json의 고정 split은 test가 전체의 9%뿐이라 95판이
    # 되고, 그 위에서 잰 차이는 2 표준오차도 안 된다. 겹으로 나누면 **모든 판이
    # 한 번씩 test가 되므로** 같은 자료로 열 배 단단한 숫자가 나온다.
    #
    # 나누는 단위는 반드시 경기다. 같은 경기가 양쪽에 걸치면 같은 선수의 같은
    # 버릇을 외운 다음 맞혔다고 하게 된다.
    matches = sorted({one.get("match") for _r, _c, one in data})
    folds = min(args.folds, len(matches))
    where = {m: i % folds for i, m in enumerate(matches)}
    print(f"경기 {len(matches)}개를 {folds}겹으로 나눕니다 (겹마다 전부 한 번씩 test)\n")

    branches = [len(one["found"]) for _r, _c, one in data]
    print(f"쓸 수 있는 판 {len(data)}개 (탐색이 프로의 길을 찾아낸 것만)")
    print(f"  배치당 갈래 중앙값 {np.median(branches):.0f}개"
          f" · 버린 길 {sum(branches) - len(data):,}개")

    rng = np.random.default_rng(0)
    places = {"아무렇게나": [], "손 식": [], "배운 식": []}
    sizes = []
    for fold in range(folds):
        train = [d for d in data if where[d[2].get("match")] != fold]
        held = [d for d in data if where[d[2].get("match")] == fold]
        if not train or not held:
            continue
        weight = fit(train)
        for rows, chose, one in held:
            sizes.append(len(rows))
            places["아무렇게나"].append(int(rng.integers(1, len(rows) + 1)))
            places["손 식"].append(
                sorted(range(len(one["found"])), key=lambda i: -by_hand(one, i)).index(chose) + 1)
            places["배운 식"].append(where_it_landed(rows, chose, weight))

    print(f"프로가 실제로 고른 길이 몇 번째에 오는가 ({len(sizes)}판, 전부 한 번씩 test)")
    for name in ("아무렇게나", "손 식", "배운 식"):
        report(name, places[name], sizes)

    # 이겼다고 말해도 되는가. 같은 판을 두 식이 나란히 풀었으므로 짝지어 센다.
    hand = np.array(places["손 식"]); learned = np.array(places["배운 식"])
    better, worse = int(np.sum(learned < hand)), int(np.sum(learned > hand))
    if better + worse:
        # 부호검정: 비긴 판은 버리고, 나머지가 반반일 확률을 정규근사로 본다.
        spread = np.sqrt((better + worse) * 0.25)
        away = abs(better - (better + worse) / 2) / spread
        print(f"\n  배운 식이 더 위로 올린 판 {better} · 더 내린 판 {worse}"
              f" · 비긴 판 {len(hand) - better - worse}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 차이라고 불러도 된다" if away > 2 else "  → 아직 차이라고 부르지 않는다"))

    weight = fit(data)   # 적어 둘 가중치는 전부로 맞춘 것
    print("\n무엇이 고른 것과 버린 것을 갈랐나 (+면 고르는 쪽)")
    for name, value in sorted(zip(NAMES, weight), key=lambda kv: -abs(kv[1])):
        if name == "기준":
            continue
        print(f"  {name:<10} {value:+7.2f}")

    json.dump({"note": "프로가 같은 배치에서 무엇 대신 무엇을 골랐는지로 맞춘 가중치. "
                       "tools/learn_choices.py가 만든다.",
               "names": NAMES, "weights": [round(float(v), 4) for v in weight],
               "rounds": len(data), "folds": folds},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
