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
from collections import Counter

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
        # 이 유형을 프로가 얼마나 자주 치는가 (로그 비율). 이웃 프로 수와 겹칠까
        # 봐 재 봤는데 겹치지 않는다 — 이웃은 40명뿐이라 후보의 45%가 0이고,
        # 기본 비율은 촘촘하다. 넣으면 1등 33.4% → 34.4% (4.1 표준편차,
        # tools/test_route_prior.py).
        #
        # 선수가 먼저 말했다 (2026-09-23): "뱅크샷 이런 선택은 안함." 세어 보니
        # 우리 후보 목록이 프로와 많이 달랐다 — 횡단 17.8배, 걸어치기 2.5배,
        # 뱅크샷 1.6배로 넘치고, 프로가 치는 절반(뒤돌리기 26% + 옆돌리기 23%)이
        # 후보에서는 20%뿐이었다. 전방위 훑기가 쿠션 먼저 맞는 길을 잘 잡는
        # 대가다. 손으로 유형을 깎지 않고 가중치가 정하게 둔다.
        branch.get("prior", LOG_FLOOR),
        # ★이웃 프로의 **길**(가까운공|면|첫 2쿠션 레일)로 센 표 (2026-09-30). 이름 표와
        # 같이 쓴다. 이름 표만 쓸 때와 짝지어: 1등 후보가 프로 길의 300 mm 안으로 들어온
        # 판 102 · 빠진 판 51 (4.1 표준편차), 프로 후보 자리 3.2 표준편차.
        math.log1p(branch.get("chosen_path", 0)),
        branch.get("rate_path", 0.5),
        # ★이웃 프로 길과의 **연속** 닮음 (2026-10). 이웃 40명의 첫 두 쿠션 자리와 이 후보의
        # 첫 두 쿠션 자리 평균 거리 d로 exp(−½(d/300)²)를 더한다. 공·면은 안 맞춘다 (맞추면
        # 더 약했다). 이름 표만으로는 구석 근처에서 레일 이름이 갈려 표를 잃는다.
        # 짝지어: 1등 후보가 프로 길 300 mm 안으로 +47 / −17판 (3.8 SD). σ 150·600은 더 약했다.
        math.log1p(branch.get("path_sim", 0.0)),
    ] + ([math.log1p(branch.get("path_sim3", 0.0))] if SIM3 else []) + bank_terms(branch, layout, cue) + [
        1.0,                                 # 기준선
    ]


BANK_NEAR_MM, BANK_FAR_MM = 250.0, 600.0


def bank_terms(branch, layout=None, cue=None):
    """뱅크샷 × 두 적구가 모였나 / 멀리 떨어졌나 (2026-10-06).

    선수: "프로들은 가능하면 제1적구를 맞히는 걸 선호해. 그러나 공 2개가 모여 있거나 뱅크샷으로
    득점할 확률이 더 높은 경우에, 또는 safety해야 할 경우에는 뱅크샷을 선택." 프로 2,788판:
    두 적구 사이 150 mm 안이면 뱅크샷을 81% 고르고(득점 77%, 다른 길 33%), 400 mm 넘으면 10~14%.
    모양을 넷 견줬다 (exp(−d/250) · 구간 · log 거리 · 문). 전체 적중은 같고(3등 안 56.6~57.1%),
    **문**만이 모인 배치의 뱅크샷을 지켰다 — 프로가 뱅크샷을 친 판에서 3등 안: 250 mm 안 28% (특징 없을 때 30%,
    exp 모양 18%), 250~600 mm 11% (11%, 4%). 나머지 모양은 뱅크샷 전체를 깎았다.
    거리는 `apart`가 있으면 그것, 없으면 배치에서.
    """
    bank = 1.0 if branch.get("route") == "뱅크샷" else 0.0
    apart = branch.get("apart")
    if apart is None and layout and cue:
        others = [layout[c] for c in layout if c != cue]
        apart = math.dist(others[0], others[1]) if len(others) == 2 else None
    if apart is None:
        return [0.0, 0.0]
    return [bank * (apart < BANK_NEAR_MM), bank * (apart >= BANK_FAR_MM)]


NAMES = ["여유", "두께", "세기", "쿠션수", "1적구이동", "줄두께", "회전량", "상단당점",
         "이웃프로수", "이웃득점률", "유형빈도", "이웃길수", "이웃길득점률", "이웃길닮음",
         "뱅크샷<250", "뱅크샷>=600", "기준"]

# 유형을 한 번도 못 본 경우의 바닥값. 2,000판쯤에서 "한 번 봤다"보다 낮다.
LOG_FLOOR = math.log(1 / 2000.0)


def route_prior(routes):
    """유형마다 log((센 수 + 1) / (전체 + 유형 수)) — 라플라스로 눌러 둔다."""
    total = sum(routes.values())
    kinds = max(len(routes), 1)
    return {r: math.log((n + 1) / (total + kinds)) for r, n in routes.items()}


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


# 이웃 투표의 열쇠 (2026-09-30 실험). "name" = 유형|가까운공|면 (지금까지),
# "pathN" = 가까운공|면|첫 N쿠션 레일 — 이름을 거치지 않고 **길**로 센다.
VOTES = "both"
# 실험 (2026-10): 이웃 길과 후보 길의 첫 두 쿠션 자리 거리로 준 연속 표. None이면 끔.
SIGMA = 300.0
SIM_SAME_FACE = False
SIM3 = False   # 실험: 첫 세 쿠션으로 잰 닮음을 하나 더
RAILS_OF = []
TABLE_L, TABLE_W = 2844.0, 1422.0


def rail_tokens(points, n):
    """쿠션 자리에서 가장 가까운 벽 이름 n개. 시뮬과 영상은 레일 이름 약속이 달라
    (시뮬은 y≈1391이 bottom) 이름을 쓰지 않고 **자리**에서 다시 계산한다."""
    out = []
    for x, y in (points or [])[:n]:
        out.append(min((("L", x), ("R", TABLE_L - x), ("B", y), ("T", TABLE_W - y)),
                       key=lambda t: t[1])[0])
    return "-".join(out) if len(out) == n else None


def with_neighbours(rounds):
    if VOTES == "both":
        return _both(rounds)
    return _with_neighbours(rounds)


def _both(rounds):
    """이름 투표와 길(첫 2쿠션) 투표를 둘 다 붙인다."""
    global VOTES
    VOTES = "path2"
    _with_neighbours(rounds)
    for one in rounds:
        for b in one["found"]:
            b["chosen_path"], b["rate_path"] = b.get("chosen", 0), b.get("rate", 0.5)
    VOTES = "name"
    out = _with_neighbours(rounds)
    VOTES = "both"
    return out


def _with_neighbours(rounds):
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
    global RAILS_OF
    _rails = pro_rails()
    RAILS_OF = [_rails.get(f"{r['match']}:{r['inning']}:{r['shot']}", []) for r in rows]
    if VOTES.startswith("path"):
        n = int(VOTES[4:])
        rails = pro_rails()
        def path_key(row, name_key):
            if not name_key:
                return None
            _route, near, face = name_key.split("|")
            tokens = rail_tokens(rails.get(f"{row['match']}:{row['inning']}:{row['shot']}"), n)
            return None if tokens is None else f"{near}|{face}|{tokens}"
        picked = [path_key(r, k) for r, k in zip(rows, picked)]
    out = []
    for one in rounds:
        i = seat.get(one["id"])
        if i is None:
            continue
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        votes, wins = {}, {}
        nearest = np.argsort(gap)[:NEIGHBOURS]
        for j in nearest:
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
            if VOTES.startswith("path"):
                tokens = rail_tokens(branch.get("cush"), int(VOTES[4:]))
                key = None if tokens is None else "|".join(key.split("|")[1:] + [tokens])
            branch["chosen"] = votes.get(key, 0)
            branch["rate"] = (wins.get(key, 0) + 1) / (branch["chosen"] + 2)
            if SIGMA and VOTES == "name":
                mine = branch.get("cush") or []
                near_face = "|".join(key.split("|")[1:3])
                total = total3 = 0.0
                for j in nearest:
                    theirs = RAILS_OF[j]
                    if len(mine) < 2 or len(theirs) < 2 or not picked[j]:
                        continue
                    if SIM_SAME_FACE and "|".join(picked[j].split("|")[1:3]) != near_face:
                        continue
                    d = np.mean([math.hypot(mine[k][0] - theirs[k][0], mine[k][1] - theirs[k][1])
                                 for k in (0, 1)])
                    total += math.exp(-0.5 * (d / SIGMA) ** 2)
                    if SIM3 and len(mine) >= 3 and len(theirs) >= 3:
                        d3 = np.mean([math.hypot(mine[k][0] - theirs[k][0], mine[k][1] - theirs[k][1])
                                      for k in (0, 1, 2)])
                        total3 += math.exp(-0.5 * (d3 / SIGMA) ** 2)
                branch["path_sim"] = total
                branch["path_sim3"] = total3
        out.append(one)
    return out


def with_prior(rounds, routes=None):
    """후보마다 그 유형을 프로가 얼마나 자주 치는지를 붙인다.

    세는 곳은 **프로가 실제로 고른 것**(각 판의 chose)이지 우리 후보 목록이
    아니다 — 후보 목록의 치우침이 바로 고치려는 대상이다.

    `routes`를 주면 그것으로 센다. 교차검증에서 학습 겹만으로 세려고 있는
    구멍이다 (tools/test_route_prior.py).
    """
    if routes is None:
        routes = Counter(one["chose"].split("|")[0] for one in rounds)
    prior = route_prior(routes)
    for one in rounds:
        for branch in one["found"]:
            branch["prior"] = prior.get(branch["route"], LOG_FLOOR)
    return rounds


def pro_rails():
    """프로가 실제로 간 길의 쿠션 자리 (app-data, 열거와 같은 틀)."""
    path = os.path.join(ROOT, "build", "app-data.json")
    plays = json.load(open(path, encoding="utf-8"))["plays"]
    return {f"soop_{p['match']}:{p['inning']}:{p['shot']}": p.get("rails") or [] for p in plays}


def path_gap(branch, rails):
    """후보와 프로의 첫 두 쿠션 자리 평균 거리 (mm). 못 재면 None."""
    mine = branch.get("cush") or []
    if len(mine) < 2 or len(rails) < 2:
        return None
    return float(np.mean([np.hypot(mine[k][0] - rails[k][0], mine[k][1] - rails[k][1])
                          for k in (0, 1)]))


_RAILS = None
# 프로의 후보를 무엇으로 정하나. "path" = 이름과 상관없이 프로 길에 가장 가까운 후보가
# PATH_MATCH_MM 안이면 그것, 아니면 같은 이름 중 가장 가까운 것. "name" = 같은 이름만.
# 2026-10: 이름이 다른 후보만 프로 길에 닿는 판이 22% — 이름으로 정답을 정하면 그 판에서
# 프로 길과 먼 후보를 정답이라고 가르친다 (CLAUDE.md §0: 이름은 맨 끝이다).
# ★재 보니 (2026-10) 길로 정답을 정해도 1등 후보가 프로 길에 닿는 비율은 그대로(37%)였고
# 이름 1등만 27%로 떨어졌다 — 지금 특징으로는 이름이 다른 쪽 정답을 올리지 못한다. 그래서 이름.
LABEL = "name"
PATH_MATCH_MM = 300.0


def pro_branch(one):
    """프로의 후보. 같은 이름(`key == chose`)이 여럿이면 첫 두 쿠션이 프로의 실제
    쿠션 자리에 가장 가까운 것 — 2026-09-30부터 열거가 이름 안에서도 길마다 후보를 둔다.
    다른 도구들이 `next(b for b ... if b["key"] == chose)`로 첫 번째를 집던 것을 바꾼다."""
    global _RAILS
    if _RAILS is None:
        _RAILS = pro_rails()
    rails = _RAILS.get(one["id"], [])
    if LABEL == "path":
        gaps = [(g, b) for b in one["found"] if (g := path_gap(b, rails)) is not None]
        if gaps:
            g, b = min(gaps, key=lambda t: t[0])
            if g <= PATH_MATCH_MM:
                return b
    same = [b for b in one["found"] if b["key"] == one["chose"]]
    if len(same) < 2:
        return same[0] if same else None
    gaps = [path_gap(b, rails) for b in same]
    if all(g is None for g in gaps):
        return same[0]
    return min(zip(gaps, same), key=lambda t: np.inf if t[0] is None else t[0])[1]


def as_arrays(rounds):
    """한 판을 (갈래별 특징, 고른 것의 자리)로.

    ★2026-09-30부터 열거는 이름 안에서도 길마다 후보를 따로 둔다. 같은 이름이 여럿이면
    **첫 두 쿠션 자리가 프로의 것에 가장 가까운** 후보가 프로의 것이다 (예전엔 첫 번째).
    """
    rails = pro_rails()
    out = []
    for one in rounds:
        rows = [features(b, one["layout"], one["cue"]) for b in one["found"]]
        chose = [i for i, b in enumerate(one["found"]) if b["key"] == one["chose"]]
        if not chose:
            continue
        one["pro_rails"] = rails.get(one["id"], [])
        pro = pro_branch(one)
        chose = [next(i for i, b in enumerate(one["found"]) if b is pro)]
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
    parser.add_argument("--votes", default="both", help="both | name | path1 | path2 | path3")
    parser.add_argument("--dry", action="store_true", help="가중치를 쓰지 않는다")
    parser.add_argument("--label", default="name", help="name | path — 프로의 후보를 무엇으로 정하나")
    parser.add_argument("--sigma", type=float, default=300.0, help="길 닮음 표의 폭 (mm)")
    parser.add_argument("--same-face", action="store_true", help="실험: 길 닮음에 공·면을 맞춘다")
    parser.add_argument("--sim3", action="store_true", help="실험: 첫 세 쿠션 닮음을 하나 더")
    parser.add_argument("--dump", default=None, help="판별 (1등 자리, 길 거리)를 JSON으로 — 두 식을 짝지어 견줄 때")
    args = parser.parse_args()
    global VOTES, SIGMA, SIM_SAME_FACE, LABEL
    LABEL = args.label
    VOTES = args.votes
    SIGMA = args.sigma
    SIM_SAME_FACE = args.same_face
    global SIM3
    SIM3 = args.sim3

    rounds = load(args.limit)
    if len(rounds) < 30:
        print(f"쓸 수 있는 판이 {len(rounds)}개뿐입니다 — "
              "tools/enumerate_alternatives.js를 먼저 끝까지 돌리세요.")
        return 1

    data = as_arrays(with_prior(with_neighbours(rounds)))

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
    # ★길로 채점 (2026-09-30): 1등 후보의 첫 두 쿠션이 프로의 것과 몇 mm인가.
    # 이름이 같아도 길이 다르면 틀린 것, 이름이 달라도 길이 같으면 맞은 것으로 센다.
    per_play = {}
    gap_of = {"아무렇게나": [], "손 식": [], "배운 식": [], "배운 식 3등 안 중 가장 가까운 것": [],
              "정답으로 삼은 후보": []}
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
            found, pro = one["found"], one["pro_rails"]
            picks = {"아무렇게나": int(rng.integers(0, len(rows))),
                     "손 식": max(range(len(found)), key=lambda i: by_hand(one, i)),
                     "배운 식": int(np.argmax(rows @ weight)),
                     "정답으로 삼은 후보": chose}
            for name, i in picks.items():
                gap_of[name].append(path_gap(found[i], pro))
            # 화면은 궤적 1개 + 2개다 (PURPOSE.md) — 셋 중 하나라도 프로 길 근처면 보여 준 것이다.
            top3 = [path_gap(found[i], pro) for i in np.argsort(-(rows @ weight))[:3]]
            top3 = [g for g in top3 if g is not None]
            gap_of["배운 식 3등 안 중 가장 가까운 것"].append(min(top3) if top3 else None)
            per_play[one["id"]] = (places["배운 식"][-1], gap_of["배운 식"][-1])

    print(f"프로가 실제로 고른 길이 몇 번째에 오는가 ({len(sizes)}판, 전부 한 번씩 test)")
    for name in ("아무렇게나", "손 식", "배운 식"):
        report(name, places[name], sizes)

    if args.dump:
        json.dump(per_play, open(args.dump, "w"))
    print("\n1등 후보의 길이 프로의 길과 얼마나 가까운가 (첫 두 쿠션 자리 평균, mm)")
    for name, values in gap_of.items():
        v = np.array([x for x in values if x is not None], float)
        if len(v):
            print(f"  {name:<18} 중앙값 {np.median(v):5.0f} · 200 안 {np.mean(v <= 200):4.0%}"
                  f" · 300 안 {np.mean(v <= 300):4.0%} · 500 안 {np.mean(v <= 500):4.0%}  (n={len(v)})")

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

    if args.dry:
        return 0
    json.dump({"note": "프로가 같은 배치에서 무엇 대신 무엇을 골랐는지로 맞춘 가중치. "
                       "tools/learn_choices.py가 만든다.",
               "names": NAMES, "weights": [round(float(v), 4) for v in weight],
               "rounds": len(data), "folds": folds},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
