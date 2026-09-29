"""시뮬이 공을 어디 세우나 — 프로가 친 줄의 끝 자리를 영상의 다음 배치와 견준다.

득점한 판의 다음 샷(같은 이닝, 같은 수구)의 `layout_mm`이 곧 이 판의 실제 끝
자리다. 열거가 적은 `rest`(프로가 고른 이름의 줄)와 공마다 거리를 잰다.

두 틀은 각자 정규화돼 있어 거울 넷 중 가장 가까운 것을 쓴다 — 그러므로 오차는
**낙관적**이다 (실제는 이보다 크다).

잣대 둘: 공이 하나도 안 움직였다고 치기, 프로가 버린 줄들의 끝 자리.

    ~/.venvs/carom/bin/python tools/rest_error.py
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 2844, 1422
ORDER = ("white", "yellow", "red")
MIRRORS = (lambda p: p, lambda p: [W - p[0], p[1]],
           lambda p: [p[0], H - p[1]], lambda p: [W - p[0], H - p[1]])


def error(rest, layout):
    return min(np.mean([np.hypot(*np.subtract(f(rest[i]), layout[c]))
                        for i, c in enumerate(ORDER)]) for f in MIRRORS)


def main():
    rows = json.load(open(os.path.join(ROOT, "data", "model.json"), encoding="utf-8"))
    after = {(r["match"], r["inning"], r["shot"]): r for r in rows}
    sim, stay, other = [], [], []
    for line in open(os.path.join(ROOT, "data", "alternatives.jsonl"), encoding="utf-8"):
        one = json.loads(line)
        if not one["scored"]:
            continue
        match, inning, shot = one["id"].rsplit(":", 2)
        nxt = after.get((match, int(inning), int(shot) + 1))
        if not nxt or nxt["cue"] != one["cue"]:
            continue
        mine = [b for b in one["found"] if b["key"] == one["chose"] and b.get("rest")]
        if not mine:
            continue
        real = nxt["layout_mm"]
        sim.append(min(error(b["rest"], real) for b in mine))
        stay.append(error([one["layout"][c] for c in ORDER], real))
        rest = [b for b in one["found"] if b["key"] != one["chose"] and b.get("rest")]
        if rest:
            other.append(np.mean([error(b["rest"], real) for b in rest]))
    for name, v in (("프로 줄의 시뮬 끝 자리", sim), ("공이 안 움직였다고 치면", stay),
                    ("버린 줄들의 끝 자리", other)):
        v = np.array(v)
        print(f"{name:14s} n={len(v)}  공당 오차 중앙값 {np.median(v):.0f} mm"
              f"  (25% {np.percentile(v, 25):.0f} · 75% {np.percentile(v, 75):.0f})")


if __name__ == "__main__":
    main()
