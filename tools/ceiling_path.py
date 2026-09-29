"""샷을 **이름 대신 길로** 정의하면 프로들끼리 얼마나 일치하는가 (2026-09-30).

`tools/ceiling.py`는 "같은 선택"을 공·면·유형으로 셌다 — 천장 40.1%(공+면).
CLAUDE.md §0: ①은 "어떤 샷이냐"이지 "무슨 유형이냐"가 아니다. 그래서 길을 더한다:

    공+면+1쿠션 레일 · 공+면+2쿠션 레일 · 공+면+3쿠션 레일   (van Balen식, `ref/prior_art.md`)

쿠션 **자리**(mm)는 배치가 다르면 뜻이 없어 이웃끼리 견줄 수 없다. **레일 이름**은
배치가 달라도 뜻이 통한다. 자리(mm)는 같은 배치 안의 채점에 쓴다 (다음 단계).

★정의를 잘게 쪼갤수록 일치도는 **기계적으로** 내려간다. 그러니 볼 것은 천장 자체보다:

* **이름 안의 갈래** — 같은 이름 안에 길이 몇 갈래로 갈리나. 많으면 이름이 서로 다른
  샷을 뭉개고 있고, 지금 열거가 이름마다 하나만 남기며 그 갈래를 버린다.
* **이름을 가로지르는 같은 길** — 같은 길인데 이름이 다른 판. 이름이 같은 샷을 가른다.
* **가까운 배치에서 일치가 오르는가** — 오르면 배치가 길을 정한다 (배울 수 있다).

    ~/.venvs/carom/bin/python tools/ceiling_path.py
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "build", "app-data.json")
L, W = 2844.0, 1422.0


def rail(point):
    """쿠션 자리에서 가장 가까운 벽 (정규화된 틀 안의 이름)."""
    x, y = point
    return min((("left", x), ("right", L - x), ("bottom", y), ("top", W - y)),
               key=lambda t: t[1])[0]


def choice(play, how):
    base = (bool(play["near"]), play["face"])
    rails = tuple(rail(p) for p in play["rails"])
    if how == "공+면":
        return base
    if how.startswith("공+면+"):
        n = int(how[4])
        return base + rails[:n]
    return play["route"]


HOWS = ("공+면", "공+면+1쿠션", "공+면+2쿠션", "공+면+3쿠션", "유형")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--min-rails", type=int, default=3)
    parser.add_argument("--no-corners", type=float, default=0.0,
                        help="첫 3쿠션 중 구석 이 mm 안인 것이 있으면 뺀다 (어느 벽인지 애매)")
    args = parser.parse_args()

    plays = [p for p in json.load(open(DATA, encoding="utf-8"))["plays"]
             if p.get("face") is not None and len(p.get("rails") or []) >= args.min_rails]
    if args.no_corners:
        cut = args.no_corners
        plays = [p for p in plays if not any(min(x, L - x) < cut and min(y, W - y) < cut
                                             for x, y in p["rails"][:3])]
    F = np.array([p["f"] for p in plays], dtype=float)
    match = np.array([p["match"] for p in plays])
    print(f"쿠션이 {args.min_rails}개 넘게 기록된 판 {len(plays)}개 · 이웃 {args.k}명 (같은 경기 빼고)\n")

    order = []
    for i in range(len(plays)):
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        order.append(np.argsort(gap)[:args.k])

    print(f"  {'무엇이 같으면 같은 샷':<16} {'가짓수':>5} {'최빈값':>7} {'★천장':>7} {'가까운 5명':>9}")
    for how in HOWS:
        mine = [choice(p, how) for p in plays]
        ceiling, near5 = [], []
        for i, near in enumerate(order):
            votes = Counter(mine[j] for j in near)
            ceiling.append(votes.most_common(1)[0][1] / len(near))
            near5.append(sum(mine[j] == mine[i] for j in near[:5]) / 5)
        common = Counter(mine).most_common(1)[0][1] / len(mine)
        print(f"  {how:<18} {len(set(mine)):5d} {common:7.1%} {np.mean(ceiling):7.1%} {np.mean(near5):9.1%}")

    # 이름 안의 갈래, 이름을 가로지르는 같은 길
    path3 = [choice(p, "공+면+3쿠션") for p in plays]
    names = [p["route"] for p in plays]
    by_name = defaultdict(Counter)
    by_path = defaultdict(Counter)
    for n, q in zip(names, path3):
        by_name[n][q] += 1
        by_path[q][n] += 1

    print("\n이름 하나 안에 길(공+면+3쿠션)이 몇 갈래인가 — 판의 80%를 덮는 갈래 수")
    for n, c in sorted(by_name.items(), key=lambda t: -sum(t[1].values())):
        total = sum(c.values())
        if total < 30:
            continue
        run, cover = 0, 0
        for _q, k in c.most_common():
            run += k
            cover += 1
            if run >= 0.8 * total:
                break
        top = c.most_common(1)[0][1] / total
        print(f"  {n:<8} {total:5d}판  갈래 {len(c):3d}  80%에 {cover:3d}갈래  제일 큰 갈래 {top:5.1%}")

    mixed = sum(sum(c.values()) for c in by_path.values() if len(c) > 1)
    share = sum(sum(c.values()) - c.most_common(1)[0][1] for c in by_path.values())
    print(f"\n같은 길인데 이름이 둘 이상인 길에 속한 판 {mixed}/{len(plays)} ({mixed / len(plays):.0%})")
    print(f"  그 길의 다수 이름과 다른 이름이 붙은 판 {share} ({share / len(plays):.0%})")
    worst = sorted(by_path.items(), key=lambda t: -(sum(t[1].values()) - t[1].most_common(1)[0][1]))[:5]
    for q, c in worst:
        print(f"    {q}: {dict(c.most_common(4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
