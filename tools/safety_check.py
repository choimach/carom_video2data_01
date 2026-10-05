"""프로는 어려운 배치에서 수비(세이프티)를 하는가 — 실패한 다음 상대에게 무엇이 남았나.

선수 (2026-10-05): "프로의 선택 중 득점 확률이 낮은 배치가 왔을 때 세이프티를 고려한 공략을
한다." 수비라면 득점보다 **실패했을 때 상대에게 어려운 배치를 남기는 것**을 고른다. 그러면 자료에
두 흔적이 남아야 한다:

  A. 어려운 배치에서 실패했을 때 상대가 받는 배치가, 쉬운 배치에서 (어쩌다) 실패했을 때보다 어렵다.
  B. 어려운 배치에서 프로는 확률이 가장 높은 길을 덜 고른다 — 그리고 덜 고른 판일수록 A가 뚜렷하다.

배치의 어려움 = 그 배치의 후보 중 **가장 높은 득점 확률** (score_probability.py의 식, 조언판의 "득점 확률").
상대가 받은 배치 = 실패한 판(match:inning:shot)의 다음 이닝 첫 판 (match:inning+1:1). 자료에 있는 것만.

    ~/.venvs/carom/bin/python tools/safety_check.py

⚠️ 이 식은 AUC 0.64다 — 잘 가르는 자가 아니다. 그래서 상대가 **실제로 들어갔는가**도 같이 본다.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learn_choices import load, with_neighbours, with_prior  # noqa: E402
from score_probability import features, pro_branch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


L, W = 2844.0, 1422.0


def geometry(one):
    """배치를 공 위치로: 수구-쿠션 거리, 두 적구 사이 거리, 수구-가까운 적구 거리 (mm). (2026-10-05)
    수비가 노리는 것 — 상대 수구를 쿠션에 붙이기, 공을 흩어 놓기 — 을 확률 식보다 곧장 잰다."""
    lay, cue = one.get("layout") or {}, one.get("cue")
    if cue not in lay or len(lay) < 3:
        return None
    c = lay[cue]
    others = [lay[k] for k in lay if k != cue]
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    return {"cushion": min(c[0], L - c[0], c[1], W - c[1]),
            "apart": d(others[0], others[1]),
            "near": min(d(c, o) for o in others)}


def main():
    w = json.load(open(os.path.join(ROOT, "data", "probability_weights.json"), encoding="utf-8"))
    mean, spread = np.array(w["mean"]), np.array(w["spread"])
    weights, bias = np.array(w["weights"]), w["bias"]

    def chance(branch):
        x = (features(branch) - mean) / spread
        return 1.0 / (1.0 + math.exp(-(x @ weights + bias)))

    rounds = with_prior(with_neighbours(load()))
    info = {}
    for one in rounds:
        found = one.get("found") or []
        if not found:
            continue
        best = max(found, key=chance)
        pro = pro_branch(one)
        info[one["id"]] = {
            "scored": one.get("scored"),
            "best": chance(best),
            "pro": chance(pro) if pro else None,
            "followed": bool(pro) and pro.get("key") == best.get("key") and pro.get("path") == best.get("path"),
            "pro_route": pro["route"] if pro else None,
            "geo": geometry(one),
        }

    def next_of(pid):
        match, inning, _shot = pid.rsplit(":", 2)
        return f"{match}:{int(inning) + 1}:1"

    misses = [(pid, v) for pid, v in info.items() if v["scored"] is False and next_of(pid) in info]
    print(f"판 {len(info)} · 실패한 판 중 다음 상대 판이 있는 것 {len(misses)}")
    if len(misses) < 60:
        print("너무 적습니다."); return 1

    all_best = np.array([v["best"] for v in info.values()])
    print(f"모든 판의 '가장 높은 득점 확률' — 중앙값 {np.median(all_best):.2f}, "
          f"사분위 {np.percentile(all_best, 25):.2f} / {np.percentile(all_best, 75):.2f}")
    base_next = [info[next_of(p)] for p, _ in misses]
    print(f"상대가 받은 배치 전체 — 확률 중앙값 {np.median([n['best'] for n in base_next]):.2f}, "
          f"실제 득점 {np.mean([1.0 if n['scored'] else 0.0 for n in base_next if n['scored'] is not None]):.0%}")

    # A. 실패한 판의 어려움(사분위)별로 상대가 받은 배치
    qs = np.percentile([v["best"] for _, v in misses], [25, 50, 75])
    def bucket(x):
        return int(np.searchsorted(qs, x, side="right"))
    rows = {k: [] for k in range(4)}
    for pid, v in misses:
        rows[bucket(v["best"])].append(info[next_of(pid)])
    print("\nA. 실패한 판의 배치 어려움별 — 상대가 받은 배치 (확률이 낮을수록 상대에게 어렵다)")
    print("   내 배치 (가장 높은 확률)   판수   상대 배치 확률 중앙값   상대 실제 득점")
    names = ["가장 어려움 (하위 25%)", "어려움", "쉬움", "가장 쉬움 (상위 25%)"]
    for k in range(4):
        r = rows[k]
        sc = [1.0 if n["scored"] else 0.0 for n in r if n["scored"] is not None]
        print(f"   {names[k]:<22}{len(r):>5}   {np.median([n['best'] for n in r]):>14.2f}   "
              f"{np.mean(sc):>14.0%}  ({len(sc)}판)")

    # 효과 크기: 가장 어려움 vs 가장 쉬움 — 상대 실제 득점의 차와 그 표준오차
    def rate(r):
        sc = [1.0 if n["scored"] else 0.0 for n in r if n["scored"] is not None]
        return np.mean(sc), len(sc)
    (p0, n0), (p3, n3) = rate(rows[0]), rate(rows[3])
    se = math.sqrt(p0 * (1 - p0) / max(n0, 1) + p3 * (1 - p3) / max(n3, 1))
    print(f"   상대 득점 차 (어려움 − 쉬움): {100 * (p0 - p3):+.0f}%p (표준오차 {100 * se:.0f}%p)")

    # B. 어려운 배치에서 프로가 최고 확률 길을 따랐나, 따르지 않았을 때 상대에게 남긴 것
    print("\nB. 프로가 '확률이 가장 높은 길'을 골랐나 — 배치 어려움별 (모든 판)")
    allq = np.percentile(all_best, [25, 50, 75])
    for k in range(4):
        sel = [v for v in info.values() if int(np.searchsorted(allq, v["best"], side="right")) == k and v["pro"] is not None]
        if not sel:
            continue
        follow = np.mean([1.0 if v["followed"] else 0.0 for v in sel])
        gap = np.median([v["best"] - v["pro"] for v in sel])
        print(f"   {names[k]:<22}{len(sel):>5}판 · 최고 확률 길을 고름 {follow:.0%} · 고른 길과 최고의 확률 차 중앙값 {gap:.2f}")

    # C. 공 위치로 (2026-10-05, 선수가 1번을 고름): 먼저 이 값들이 어려움의 척도인가 — 모든 판에서 득점과의 관계
    print("\nC. 공 위치로 잰 어려움 — 먼저, 모든 판에서 이 값이 득점과 관련 있나 (사분위별 득점률)")
    geo_all = [(v["geo"], v["scored"]) for v in info.values() if v["geo"] and v["scored"] is not None]
    labels = {"cushion": "수구-쿠션 거리", "apart": "두 적구 사이", "near": "수구-가까운 적구"}
    for key in ("cushion", "apart", "near"):
        vals = np.array([g[key] for g, _ in geo_all]); hit = np.array([1.0 if s else 0.0 for _, s in geo_all])
        q = np.percentile(vals, [25, 50, 75])
        parts = []
        for k in range(4):
            m = np.searchsorted(q, vals, side="right") == k
            parts.append(f"{hit[m].mean():.0%}")
        print(f"   {labels[key]:<14} 사분위 경계 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} mm · 득점률 (짧음→김) {' · '.join(parts)}")
    # 상대가 받은 배치의 공 위치 — 내 배치 어려움별, 그리고 모든 판과 견줘
    print("\n   상대가 받은 배치의 공 위치 (중앙값, mm) — 내 배치 어려움별")
    print(f"   {'모든 판':<22}{'':>5}   " + " · ".join(f"{labels[k]} {np.median([g[k] for g, _ in geo_all]):.0f}" for k in labels))
    for k in range(4):
        gs = [info[next_of(p)]["geo"] for p, v in misses if bucket(v["best"]) == k and info[next_of(p)]["geo"]]
        print(f"   {names[k]:<22}{len(gs):>5}   " + " · ".join(f"{labels[x]} {np.median([g[x] for g in gs]):.0f}" for x in labels))
    print("   어려운 배치(하위 50%)에서 실패 — 최고 확률 길을 따랐나에 따라")
    for flag, title in ((True, "따랐다"), (False, "다른 길을 골랐다")):
        gs = [info[next_of(p)]["geo"] for p, v in misses if bucket(v["best"]) <= 1 and v["followed"] is flag and info[next_of(p)]["geo"]]
        if gs:
            print(f"   {title:<22}{len(gs):>5}   " + " · ".join(f"{labels[x]} {np.median([g[x] for g in gs]):.0f}" for x in labels))

    print("\n   어려운 배치(하위 50%)에서 실패했을 때 — 최고 확률 길을 따랐나에 따라 상대에게 남긴 것")
    hard = [(p, v) for p, v in misses if bucket(v["best"]) <= 1]
    for flag, title in ((True, "따랐다"), (False, "다른 길을 골랐다")):
        r = [info[next_of(p)] for p, v in hard if v["followed"] is flag]
        if not r:
            continue
        sc = [1.0 if n["scored"] else 0.0 for n in r if n["scored"] is not None]
        print(f"   {title:<12}{len(r):>4}판 · 상대 배치 확률 중앙값 {np.median([n['best'] for n in r]):.2f}"
              f" · 상대 실제 득점 {np.mean(sc):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
