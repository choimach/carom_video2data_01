"""닮음의 기준을 **생긴 모양**에서 **결과**로 바꿔 본다.

2026-09-25. 이 날 "닮았다"의 정의를 네 번 손봤고 네 번 다 졌다 — NCA 3~4축,
극좌표 6축, 무게 학습, 코너 4축. 뒤늦게 바깥을 보니 (`ref/prior_art.md`),

  · **당구 쪽에는 선례가 없다.** PickPocket(AAAI 2006)도 Park&Park(2022)도
    물리 계산·탐색이고 프로 자료로 이웃을 찾지 않는다.
  · 진짜 선례는 **체스 국면 검색**이다. Ganguly&Leveling(SIGIR 2014)이 손으로
    짠 기하 특징으로 **우리 열두 축과 똑같은 자리에서 멈췄고**, 그 다음 세대는
    국면을 **엔진 평가값이 비슷한 것끼리** 묻어 넣어서 뚫었다.

★우리가 진 네 번은 **전부 기하를 다시 적은 것**이다. 목표는 한 번도 안 바꿨다.

체스가 엔진 평가값을 쓴 자리에 우리는 **시뮬레이터**가 있고, 그 결과가 이미
파일로 있다 — `data/alternatives.jsonl`. 판마다 그날 실제로 있었던 길을 통으로
묶고 통마다 **여유**가 붙어 있다.

> **배치 하나 = 여유 벡터.** 두 배치가 닮았다는 건 *같은 샷들이 같은 정도로
> 열려 있다*는 뜻이지, 공이 비슷한 자리에 있다는 뜻이 아니다.

⚠️ **순환 조심 둘.**
① 서명은 **우리 물리**에서 나온다. 물리가 틀리면 이웃이 그쪽으로 쏠린다.
   그래서 채점은 반드시 **프로가 실제로 고른 것**으로 한다 (경기 단위 10겹).
② 이웃은 **같은 경기에서 뽑지 않는다** (`with_neighbours`와 같은 규칙).
   자기 판의 서명으로 자기 판의 답을 찾으면 안 된다.

⚠️ **판돈을 맞춘다.** 서명이 있는 판은 2,095/2,466(85%)뿐이다. 서명 쪽만 이웃
   후보가 적으면 불리하므로, **세 방식 모두 같은 2,095판 안에서만** 이웃을
   찾는다. 그래야 달라지는 것이 잣대 하나뿐이다.

★★ **쟀다. 진다. 그것도 크게.** (2026-09-25, 프로 2,095판 · 경기 단위 10겹)

        지금 (열두 축)        1등 33.7% · 3등 안 60.0%      —
        더하되 무게 반반       1등 31.9% · 3등 안 58.4%   3.3 SD 나쁨
        열두 축 + 서명 (60)   1등 30.5% · 3등 안 57.2%   5.5 SD 나쁨
        결과 서명만 (48)      1등 28.9% · 3등 안 54.6%   8.5 SD 나쁨

    **서명에 무게를 줄수록 나빠진다** — 잡음이 아니라 용량-반응이다. 이유도 쟀다:

                    이웃표–여유 상관   이웃프로수 무게   이웃득점률 무게
        열두 축          +0.222          +0.923         +0.684
        결과 서명        +0.400          +0.344         +0.489

    **이웃의 표가 여유의 메아리가 된다.** 여유는 이미 순위가 직접 쓰는 축이라
    같은 말을 두 번 듣는 셈이 되고, 모형은 이웃을 **믿지 않게 된다**
    (0.92 → 0.34). 그런데 이웃은 **프로 자료가 들어오는 유일한 통로**다.

    ★**이웃은 여유가 말해 주지 않는 것을 가져와야 쓸모가 있다.** 기하가 이기는
    이유가 이것이다 — 기하는 물리와 **다른 말**을 한다. 목표를 바꾸려면 물리에서
    나오지 않은 목표여야 하는데, 프로의 선택 자체는 NCA가 이미 졌다 (09-23).

    남겨 두는 것은 다시 이 생각을 할 사람을 위해서다.

    ~/.venvs/carom/bin/python tools/test_outcome_signature.py
"""

import collections
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import learn_choices  # noqa: E402
from learn_choices import (NEIGHBOURS, as_arrays, fit,  # noqa: E402
                           load, where_it_landed, with_prior)
from route_model import player_features  # noqa: E402

FOUND = os.path.join(ROOT, "data", "alternatives.jsonl")
MODEL = os.path.join(ROOT, "data", "model.json")
MIN_PLAYS = 40          # 이보다 드문 통은 서명에 넣지 않는다 (잡음만 된다)
GEO = 12                # player_features의 축 수


def signatures():
    """판 id → {통 이름: 여유}. 열거가 끝난 판만.

    여유는 꼬리가 길다 (중앙값 0.13, 최대 6.4). `log1p`로 눌러서, "열려 있나
    닫혀 있나"가 "얼마나 활짝 열렸나"에 묻히지 않게 한다.
    """
    seen = collections.Counter()
    table = {}
    with open(FOUND, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            one = json.loads(line)
            if not one.get("reached") or len(one.get("found") or []) < 2:
                continue
            room = {}
            for branch in one["found"]:
                room[branch["key"]] = max(room.get(branch["key"], 0.0),
                                          float(branch.get("room") or 0.0))
            table[one["id"]] = room
            for key in room:
                seen[key] += 1
    vocab = sorted(k for k, n in seen.items() if n >= MIN_PLAYS)
    index = {k: i for i, k in enumerate(vocab)}
    out = {}
    for play, room in table.items():
        vec = np.zeros(len(vocab), dtype=float)
        for key, value in room.items():
            if key in index:
                vec[index[key]] = math.log1p(value)
        out[play] = vec
    return out, vocab


SIGN, VOCAB = signatures()


def signature_features(row):
    return SIGN[f"{row['match']}:{row['inning']}:{row['shot']}"]


def both(row):
    return np.concatenate([player_features(row), signature_features(row)])


def half_and_half():
    """더할 때 **칸 수가 곧 무게**가 되는 것을 막는 열 배율.

    표준화한 뒤 유클리드 거리를 재면 칸마다 무게가 같다. 그냥 이어 붙이면
    48 대 12로 **서명이 기하를 삼킨다** — 오늘 극좌표에서 배운 셈법 그대로다
    (중복이 곧 가중치였다). 각 덩이를 √칸수로 나눠 반반으로 만든다.

    ⚠️ 표준화 **뒤에** 곱해야 한다. 앞에서 나누면 표준화가 도로 지운다.
    """
    return np.concatenate([np.full(GEO, 1.0 / math.sqrt(GEO)),
                           np.full(len(VOCAB), 1.0 / math.sqrt(len(VOCAB)))])


def with_neighbours(rounds, builder, colscale=None):
    """`learn_choices.with_neighbours`와 같되, **잣대를 갈아 끼울 수 있고**
    이웃 후보를 서명 있는 판으로 못박는다 (판돈 맞추기).

    `colscale`은 표준화 **뒤에** 열마다 곱하는 배율이다 (덩이 무게 맞추기).
    """
    model = json.load(open(MODEL, encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    rows = [r for r in rows
            if f"{r['match']}:{r['inning']}:{r['shot']}" in SIGN]      # ← 같은 판돈
    F = np.array([builder(r) for r in rows], dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    if colscale is not None:
        F = F * np.asarray(colscale, dtype=float)
    match = np.array([r["match"] for r in rows])
    seat = {f"{r['match']}:{r['inning']}:{r['shot']}": i for i, r in enumerate(rows)}

    def bucket(row):
        cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
        gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue))
                for c, xy in row["layout_mm"].items() if c != row["cue"]}
        first, side = row.get("first_object_ball"), row.get("struck_side")
        if not first or first not in gaps or side is None:
            return None
        return (f"{row['route']}|{gaps[first] == min(gaps.values())}"
                f"|{'left' if side > 0 else 'right'}")

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
            key = (f"{branch['route']}"
                   f"|{reach.get(branch['first']) == min(reach.values())}"
                   f"|{branch['face']}")
            branch["chosen"] = votes.get(key, 0)
            branch["rate"] = (wins.get(key, 0) + 1) / (branch["chosen"] + 2)
        out.append(one)
    return out


def score(builder, title, rounds, colscale=None):
    data = as_arrays(with_prior(with_neighbours(rounds, builder, colscale)))
    folds = 10
    ms = sorted({one.get("match") for _r, _c, one in data})
    where = {m: i % folds for i, m in enumerate(ms)}
    place = []
    for fold in range(folds):
        tr = [d for d in data if where[d[2].get("match")] != fold]
        te = [d for d in data if where[d[2].get("match")] == fold]
        if not tr or not te:
            continue
        w = fit(tr)
        for rows, chose, _one in te:
            place.append(where_it_landed(rows, chose, w))
    p = np.array(place, float)
    print(f"  {title:<24} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
          f" · 자리 중앙값 {np.median(p):4.1f}   (n={len(p)})")
    return p


def sign_test(a, b, label):
    n = min(len(a), len(b))
    a2, b2 = a[:n], b[:n]
    up, down = int(np.sum(b2 < a2)), int(np.sum(b2 > a2))
    if not (up + down):
        return
    away = abs(up - (up + down) / 2) / math.sqrt((up + down) * 0.25)
    verdict = ("  → 새 잣대가 낫다" if away > 2 and up > down
               else ("  → 지금이 낫다" if away > 2 else "  → 가릴 수 없다"))
    print(f"\n  [{label}] 올라간 판 {up} · 내려간 판 {down} · 비긴 판 {n - up - down}")
    print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차{verdict}")


def main():
    print(f"결과 서명 {len(VOCAB)}차원 · 서명 있는 판 {len(SIGN)}\n")
    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)\n")
    rounds = load()
    a = score(player_features, "지금 (열두 축)", rounds)
    b = score(signature_features, "결과 서명만", rounds)
    c = score(both, "열두 축 + 결과 서명", rounds)
    d = score(both, "더하되 무게 반반", rounds, half_and_half())
    sign_test(a, b, "서명만")
    sign_test(a, c, "더해서")
    sign_test(a, d, "반반")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
