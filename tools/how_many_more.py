"""경기를 몇 개 더 받아야 순위가 얼마나 좋아지나 — 추정하지 말고 곡선을 그린다.

선수가 세 번 물었다 (2026-09-25, 2026-09-26): *"몇 경기나 더 download해야
model이 좋아져?"* 지금까지의 답은 **다른 지표로 5일 전에 잰 곡선**
(`tools/learning_curve.py`, 이웃 찾기의 유형 정확도, 두 배마다 +2점)에서
끌어온 것이었다. 여기서는 **지금 쓰는 지표로 직접** 잰다.

방법: 채점하는 판은 **고정**하고, **이웃을 뽑아 오는 못**만 줄인다. 그래야
달라지는 것이 "자료가 얼마나 많은가" 하나뿐이다. 못을 1/8, 1/4, 1/2, 1로
줄여 가며 프로의 선택이 몇 등에 오는지 본다.

⚠️ 못을 줄일 때 **경기 단위로** 줄인다. 판 단위로 줄이면 같은 경기의 다른 판이
   남아 이웃 제외 규칙(같은 경기는 빼기)이 헐거워진다.

★천장을 같이 그린다. 닮은 배치에서 **프로들끼리** 얼마나 일치하는지가 상한이고
(2026-09-23에 쟀다: 공+면 40.1%), 그 위로는 올라갈 자리가 없다. 곡선의 기울기와
천장까지의 거리를 알면 **몇 배가 필요한지**가 나눗셈으로 나온다.

    ~/.venvs/carom/bin/python tools/how_many_more.py
"""

import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from learn_choices import (NEIGHBOURS, as_arrays, fit, load,  # noqa: E402
                           where_it_landed, with_prior)
from route_model import player_features  # noqa: E402

MODEL = os.path.join(ROOT, "data", "model.json")
CEILING = 0.401      # 공+면 상한, 2026-09-23 (HANDOFF "①번의 천장")
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
REPEATS = 3          # 못을 어떻게 고르냐에 따라 흔들리므로 여러 번


def pool(fraction, seed):
    """이웃을 뽑아 올 판들 — 경기 단위로 줄인다."""
    model = json.load(open(MODEL, encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    matches = sorted({r["match"] for r in rows})
    if fraction < 1.0:
        keep = set(random.Random(seed).sample(matches, max(2, round(len(matches) * fraction))))
        rows = [r for r in rows if r["match"] in keep]
    return rows


def with_neighbours(rounds, rows):
    """`learn_choices.with_neighbours`와 같되 못을 밖에서 받는다."""
    F = np.array([player_features(r) for r in rows], dtype=float)
    # ⚠️ 채점하는 판도 **같은 자로** 표준화해야 한다. 못만 표준화하고 기준 판은
    # 원값으로 두면 거리가 단위가 큰 축(mm)에 끌려간다 — 처음에 그렇게 짜서
    # 최근접 거리가 2,487로 찍혔다 (제대로면 1.2 언저리다).
    mean, spread = F.mean(0), F.std(0) + 1e-9
    F = (F - mean) / spread
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
    out, nearest = [], []
    for one in rounds:
        # 채점하는 판은 못에서 빠졌을 수도 있다. 그래도 채점은 해야 하므로
        # 자기 자리를 못에서 찾는 대신 **특징을 직접 만들어** 거리를 잰다.
        here = one.get("_features")
        if here is None:
            continue
        here = (here - mean) / spread
        gap = np.linalg.norm(F - here, axis=1)
        gap[match == one["match"]] = np.inf
        order = np.argsort(gap)[:NEIGHBOURS]
        if not np.isfinite(gap[order[0]]):
            continue
        nearest.append(float(gap[order[0]]))
        votes, wins = {}, {}
        for j in order:
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
    return out, (float(np.median(nearest)) if nearest else float("nan"))


def attach_features(rounds):
    """채점하는 판의 특징을 미리 붙여 둔다 (못이 줄어도 변하지 않게)."""
    model = json.load(open(MODEL, encoding="utf-8"))
    rows = model if isinstance(model, list) else model.get("plays", model.get("rows"))
    seat = {f"{r['match']}:{r['inning']}:{r['shot']}": r
            for r in rows if r.get("route") and r.get("layout_mm")}
    kept = []
    for one in rounds:
        row = seat.get(one["id"])
        if row is None:
            continue
        one["_features"] = player_features(row)
        one["match"] = row["match"]
        kept.append(one)
    # 표준화는 못마다 다르므로, 여기서는 원값만 붙이고 with_neighbours에서 맞춘다
    return kept


def score(rounds, rows):
    ready, nearest = with_neighbours(rounds, rows)
    if len(ready) < 50:
        return None
    data = as_arrays(with_prior(ready))
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
        for branch_rows, chose, _one in te:
            place.append(where_it_landed(branch_rows, chose, w))
    p = np.array(place, float)
    return (float(np.mean(p == 1)), float(np.mean(p <= 3)), nearest,
            len(rows), len({r["match"] for r in rows}))


def main():
    rounds = attach_features(load())
    print(f"채점하는 판 {len(rounds)}개 — 못만 줄인다 (경기 단위)\n")
    print(f"{'못':<10}{'경기':>6}{'판':>7}{'1등':>8}{'3등 안':>8}{'최근접 거리':>12}")
    curve = []
    for fraction in FRACTIONS:
        got = []
        for repeat in range(REPEATS if fraction < 1.0 else 1):
            rows = pool(fraction, 1000 + repeat)
            out = score(rounds, rows)
            if out:
                got.append(out)
        if not got:
            continue
        first = float(np.mean([g[0] for g in got]))
        top3 = float(np.mean([g[1] for g in got]))
        near = float(np.mean([g[2] for g in got]))
        plays = int(np.mean([g[3] for g in got]))
        matches = int(np.mean([g[4] for g in got]))
        curve.append((plays, first))
        print(f"{fraction:<10.3f}{matches:>6}{plays:>7}{first:>8.1%}{top3:>8.1%}{near:>12.3f}")

    if len(curve) < 2:
        return 1
    # 판수의 로그에 직선을 맞춘다 — 두 배마다 몇 점인가
    x = np.log2([c[0] for c in curve])
    y = np.array([c[1] for c in curve])
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\n두 배마다 {slope * 100:+.2f}점 (1등 기준)")
    now_plays, now_score = curve[-1]
    per_match = now_plays / 53.0          # 지금 53경기에서 나온 판
    if slope <= 0:
        print("★기울기가 0 이하다 — 자료를 늘려도 안 좋아진다.")
        return 0
    doublings = (CEILING - now_score) / slope
    print(f"천장 {CEILING:.1%}까지 {CEILING - now_score:+.1%} 남았다"
          f" → 두 배를 {doublings:.1f}번 = 자료 {2 ** doublings:,.0f}배")
    print(f"  필요한 판 {now_plays * 2 ** doublings:,.0f} · "
          f"경기 {now_plays * 2 ** doublings  / per_match:,.0f}개 "
          f"(지금 {now_plays  / per_match:.0f}개) · "
          f"내려받기 {now_plays * 2 ** doublings  / per_match * 9 / 1000:,.1f} TB")
    for step in (2, 4, 8):
        gain = slope * math.log2(step)
        need = (step - 1) * now_plays  / per_match
        print(f"  경기 {need:,.0f}개를 더 받으면 ({step}배) {gain * 100:+.1f}점"
              f" → {now_score + gain:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
