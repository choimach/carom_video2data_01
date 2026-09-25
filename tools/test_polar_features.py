"""배치를 열두 축 대신 **겹침 없는 극좌표 여섯 축**으로 적으면 순위가 나아지나.

선수가 물었다 (2026-09-25): *"이 변수들을 극좌표로 바꾸면 변수가 줄어들지는
않나. 다른 장점은 없나?"*

재 보니 열두 축 중 여섯이 나머지에서 **그대로 계산된다** (차이 0.000). 배치는
여섯 자유도뿐이다. 중복은 이웃 거리에서 같은 사실을 두 번 세게 만든다.

여기서 재는 것은 **순위**다. 2026-09-23에 NCA로 3~4축까지 줄여 이웃을 6배
가깝게 만든 적이 있는데, 그때 잰 것은 "이웃이 무엇을 고를지 맞히나"였고 거의
안 움직였다. **순위에 쓸모가 있나는 다른 질문이고 그건 안 재 봤다.**

    ~/.venvs/carom/bin/python tools/test_polar_features.py
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import learn_choices  # noqa: E402
from learn_choices import (as_arrays, fit, load, where_it_landed,  # noqa: E402
                           with_neighbours, with_prior)
from route_model import player_features, polar_features  # noqa: E402


def score(builder, folds=10):
    learn_choices.player_features = builder          # with_neighbours가 쓰는 것
    import route_model
    route_model.player_features = builder
    data = as_arrays(with_prior(with_neighbours(load())))
    matches = sorted({one.get("match") for _r, _c, one in data})
    where = {m: i % folds for i, m in enumerate(matches)}
    place = []
    for fold in range(folds):
        train = [d for d in data if where[d[2].get("match")] != fold]
        held = [d for d in data if where[d[2].get("match")] == fold]
        if not train or not held:
            continue
        weight = fit(train)
        for rows, chose, _one in held:
            place.append(where_it_landed(rows, chose, weight))
    return np.array(place, dtype=float)


def main():
    out = {}
    for name, builder in (("열두 축 (지금)", player_features),
                          ("극좌표 여섯 축", polar_features)):
        p = score(builder)
        out[name] = p
        print(f"  {name:<16} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
              f" · 자리 중앙값 {np.median(p):4.1f}   (n={len(p)})")

    a, b = out["열두 축 (지금)"], out["극좌표 여섯 축"]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    better, worse = int(np.sum(b < a)), int(np.sum(b > a))
    if better + worse:
        away = abs(better - (better + worse) / 2) / math.sqrt((better + worse) * 0.25)
        print(f"\n  극좌표가 올린 판 {better} · 내린 판 {worse} · 비긴 판 {n - better - worse}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 극좌표가 낫다" if away > 2 and better > worse
                 else ("  → 열두 축이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)\n")
    raise SystemExit(main())
