"""조준 여유를 정밀하게 재면 순위가 나아지는가 — 같은 자료로 가른다.

0.25도 눈금에서는 통의 **95%가 같은 값에 묶였다** (프로 배치 20개, 통 410개).
가장자리를 이분법으로 찾으면 7%로 떨어진다. 그런데 "덜 묶인다"가 곧 "순위가
나아진다"는 아니다.

열거가 통마다 두 값을 다 저장한다 — `room`(이분법)과 `rough`(0.25도 눈금).
같은 판, 같은 후보, 같은 나머지 특징에 **여유만 바꿔** 끼우고 잰다.

    ~/.venvs/carom/bin/python tools/test_fine_room.py
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from learn_choices import (NAMES, as_arrays, fit, load, where_it_landed,  # noqa: E402
                           with_neighbours, with_prior)

ROOM = NAMES.index("여유")


def main():
    rounds = with_prior(with_neighbours(load()))
    data = as_arrays(rounds)
    print(f"채점할 판 {len(data)}개\n")

    matches = sorted({one.get("match") for _r, _c, one in data})
    folds = min(10, len(matches))
    where = {m: i % folds for i, m in enumerate(matches)}

    # 굵은 눈금 판: 여유 칸만 rough로 갈아 끼운다.
    coarse = []
    for rows, chose, one in data:
        swapped = rows.copy()
        swapped[:, ROOM] = [math.log1p(b.get("rough", b["room"])) for b in one["found"]]
        coarse.append((swapped, chose, one))

    places = {"굵은 눈금 0.25도": [], "이분법 0.006도": []}
    for fold in range(folds):
        for name, pack in (("굵은 눈금 0.25도", coarse), ("이분법 0.006도", data)):
            train = [d for d in pack if where[d[2].get("match")] != fold]
            held = [d for d in pack if where[d[2].get("match")] == fold]
            if not train or not held:
                continue
            weight = fit(train)
            for rows, chose, _one in held:
                places[name].append(where_it_landed(rows, chose, weight))

    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)")
    for name in places:
        p = np.array(places[name], dtype=float)
        print(f"  {name:<16} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
              f" · 자리 중앙값 {np.median(p):4.1f}")

    a = np.array(places["굵은 눈금 0.25도"])
    b = np.array(places["이분법 0.006도"])
    better, worse = int(np.sum(b < a)), int(np.sum(b > a))
    if better + worse:
        away = abs(better - (better + worse) / 2) / math.sqrt((better + worse) * 0.25)
        print(f"\n  정밀하게 재서 올라간 판 {better} · 내려간 판 {worse}"
              f" · 비긴 판 {len(a) - better - worse}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 정밀하게 재는 것이 낫다" if away > 2 and better > worse
                 else ("  → 굵은 눈금이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
