"""프로들끼리 얼마나 일치하는가 — 곧 우리 점수의 상한.

선수가 ①번을 이렇게 정했다 (2026-09-21): *"프로가 실제로 친 게 아니고, 프로
데이터를 기반으로 임의 배치에서 프로가 선택을 할 **확률이 높은** 궤적."*

목표가 확률이면, **채점은 하한만 준다.** 어떤 배치에서 프로의 40%가 A를 30%가
B를 고른다면, A를 1등에 놓아도 가려 둔 판이 B였으면 틀린 것으로 센다. 그러니
"25%가 잘한 건가"를 알려면 먼저 **아무리 잘해도 몇 %인가**를 알아야 한다.

두 가지를 잰다:

* **상한** — 배치마다 이웃들의 선택 분포에서 **가장 많은 쪽의 몫**. 참 분포를
  안다 해도 그보다 잘 맞힐 수는 없다.
* **거리에 따른 일치도** — 배치가 가까울수록 프로들이 더 일치하는가. 가까운
  배치에서 일치도가 1에 가까워지면 **배치가 선택을 정한다**는 뜻이고, 우리
  특징이 그 정보를 못 담고 있는 것이다 (고칠 수 있다). 가까워져도 반반이면
  **같은 배치에서도 사람마다 다르게 고른다**는 뜻이고, 상한이 낮다.

무엇을 "같은 선택"으로 볼지는 선수가 정한 차례를 따른다 — 경로 이름(5번)이
아니라 **어느 공을 어느 면으로**(1·2번을 정하는 것)이다.

    ~/.venvs/carom/bin/python tools/ceiling.py
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "build", "app-data.json")


def choice_of(play, how):
    if how == "면":
        return play["face"]
    if how == "공":
        return bool(play["near"])
    if how == "공+면":
        return (bool(play["near"]), play["face"])
    return play["route"]                      # 5번 — 견주기 위해서만


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=40, help="이웃 수")
    args = parser.parse_args()

    plays = [p for p in json.load(open(DATA, encoding="utf-8"))["plays"]
             if p.get("face") is not None]
    F = np.array([p["f"] for p in plays], dtype=float)
    match = np.array([p["match"] for p in plays])
    print(f"플레이 {len(plays)}판 · 경기 {len(set(match))}개 · 이웃 {args.k}명"
          " (같은 경기는 빼고)\n")

    for how in ("공", "면", "공+면", "유형"):
        mine = [choice_of(p, how) for p in plays]
        ceiling, agree, base = [], [], Counter(mine)
        bands = {0: [], 1: [], 2: [], 3: []}
        for i in range(len(plays)):
            gap = np.linalg.norm(F - F[i], axis=1)
            gap[match == match[i]] = np.inf
            near = np.argsort(gap)[:args.k]
            votes = Counter(mine[j] for j in near)
            total = sum(votes.values())
            if not total:
                continue
            ceiling.append(votes.most_common(1)[0][1] / total)
            agree.append(votes[mine[i]] / total)
            # 가장 가까운 다섯만 따로 — 배치가 닮을수록 더 일치하는지 보려고.
            close = np.argsort(gap)[:5]
            band = min(3, int(np.median(gap[close]) // 1.0))
            bands.setdefault(band, []).append(
                Counter(mine[j] for j in close)[mine[i]] / len(close))

        common = base.most_common(1)[0][1] / len(mine)
        print(f"  ── {how} ──")
        print(f"     아무 생각 없이 최빈값만        {common:5.1%}")
        print(f"     가려 둔 판과 이웃이 일치       {np.mean(agree):5.1%}")
        print(f"     ★상한 (이웃 중 최다의 몫)      {np.mean(ceiling):5.1%}")
        near5 = [v for b in sorted(bands) for v in bands[b]]
        if near5:
            print(f"     가장 가까운 5명과의 일치      {np.mean(near5):5.1%}")
        print()

    print("★읽는 법: 상한이 낮으면 같은 배치에서도 프로가 갈린다는 뜻이고,")
    print("  그때는 정답을 하나로 맞히려 애쓸 것이 아니라 **확률로 보여주는** 것이 맞다.")
    print("  (선수가 궤적 1개 + 2개를 원한 것이 바로 그 모양이다.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
