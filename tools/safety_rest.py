"""수비(세이프티)의 흔적 — 실패한 판이 상대에게 **실제로 남긴** 배치로 (2026-10-05).

`tools/safety_check.py`(A~C)는 "다음 이닝 첫 판"을 상대가 받은 배치로 썼다. 그 사이에 놓친 판이 끼어 있어도
몰랐다. 여기서는 실패한 판의 `rest_mm`(공이 정말 멈춘 자리)과 다음 이닝 첫 판의 시작 배치가 공 셋 모두
25 mm 안에서 같을 때만 짝으로 쓴다 — 상대가 받은 배치와 그 결과가 둘 다 확실하다.

바깥 자료(`ref/safety.md`)의 가설을 하나씩 잰다. 잣대는 **상대가 실제로 득점했는가**:
  H1. 상대 수구를 한쪽 끝에, 나머지 둘을 반대쪽 끝에 (장축 거리)
  H2. 상대 수구를 1적구로 친 실패 (상대 공을 빨간 공에서 떼어 놓는다)
  H3. 상대 수구가 쿠션에 가깝다 (프로즌)
  H4. 공을 흩어 놓는다 (세 공 사이 거리)

    ~/.venvs/carom/bin/python tools/safety_rest.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLOURS = ("white", "yellow", "red")
L, W = 2844.0, 1422.0
MATCH_MM = 25.0


def pairs():
    """(실패한 판, 상대가 받은 판) — 멈춘 자리와 다음 시작이 맞는 것만."""
    out, base = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json"))):
        if path.endswith("index.json"):
            continue
        plays = json.load(open(path, encoding="utf-8"))["plays"]
        by = {(int(p["inning"]), int(p["shot_number"])): p for p in plays}
        for p in plays:
            if str(p.get("inferred")) != "True" and not p.get("rejected_for") and p.get("success") is not None:
                base.append(str(p["success"]) == "True")
            if str(p.get("success")) != "False" or not p.get("rest_mm"):
                continue
            q = by.get((int(p["inning"]) + 1, 1))
            if not q or str(q.get("inferred")) == "True" or q.get("success") is None:
                continue
            lay = q.get("layout_mm") or {}
            if not all(c in lay for c in COLOURS):
                continue
            if max(math.dist(p["rest_mm"][c], lay[c]) for c in COLOURS) <= MATCH_MM:
                out.append((p, q))
    return out, np.array(base)


def leave(q):
    """상대(q의 수구) 쪽에서 본 배치."""
    lay, oc = q["layout_mm"], q["cue_ball"]
    others = [c for c in COLOURS if c != oc]
    o = np.array(lay[oc])
    a, b = np.array(lay[others[0]]), np.array(lay[others[1]])
    return {
        # H1: 두 공이 같은 쪽에 있고 상대 수구가 장축으로 떨어져 있는 정도 (두 공 중 가까운 것까지의 x 거리)
        "ends": min(abs(o[0] - a[0]), abs(o[0] - b[0])),
        # H3
        "rail": min(o[0], L - o[0], o[1], W - o[1]),
        # H4
        "near": min(np.linalg.norm(o - a), np.linalg.norm(o - b)),
        "apart": float(np.linalg.norm(a - b)),
    }


def rate(xs):
    xs = np.asarray(xs, dtype=float)
    if not len(xs):
        return "  -  "
    se = math.sqrt(max(xs.mean() * (1 - xs.mean()), 1e-9) / len(xs))
    return f"{xs.mean():4.0%} ±{se:3.0%} ({len(xs):3d})"


def main():
    ps, base = pairs()
    print(f"짝 {len(ps)} — 실패한 판의 멈춘 자리 = 상대 첫 판의 시작 (공 셋 {MATCH_MM:.0f} mm 안)")
    opp = np.array([str(q["success"]) == "True" for _, q in ps])
    print(f"모든 쓸 수 있는 판의 득점률 {base.mean():.0%} ({len(base)}판) · 실패 뒤 상대 득점률 {rate(opp)}")
    print("  (실패 뒤가 더 낮으면, 실패가 대체로 상대에게 어려운 배치를 남긴다는 뜻)")

    rows = [(p, q, leave(q), str(q["success"]) == "True") for p, q in ps]

    def by_quartile(key, title, unit="mm"):
        v = np.array([r[2][key] for r in rows])
        qs = np.percentile(v, [25, 50, 75])
        print(f"\n{title} — 사분위 경계 {qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} {unit}")
        for k, name in enumerate(("짧음", "", "", "김")):
            m = np.searchsorted(qs, v, side="right") == k
            print(f"   {k + 1}분위 {name:<4} 상대 득점 {rate([r[3] for r, s in zip(rows, m) if s])}")

    by_quartile("ends", "H1. 상대 수구가 두 공에서 장축으로 떨어진 거리 (둘 중 가까운 것)")
    by_quartile("rail", "H3. 상대 수구-쿠션 거리")
    by_quartile("near", "H4a. 상대 수구-가까운 공 거리")
    by_quartile("apart", "H4b. 두 적구 사이 거리 (상대에게)")

    print("\nH2. 실패한 판의 1적구가 무엇이었나 — 상대 득점")
    for title, sel in (("상대 수구를 먼저", lambda p, q: p.get("first_object_ball") == q["cue_ball"]),
                       ("빨간 공을 먼저", lambda p, q: p.get("first_object_ball") == "red"),
                       ("아무것도 / 모름", lambda p, q: p.get("first_object_ball") not in (q["cue_ball"], "red"))):
        sub = [r for r in rows if sel(r[0], r[1])]
        if sub:
            e = np.median([r[2]["ends"] for r in sub])
            print(f"   {title:<14} 상대 득점 {rate([r[3] for r in sub])} · 상대 수구 장축 거리 중앙값 {e:.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
