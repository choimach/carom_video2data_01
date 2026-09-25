"""세 공을 삼각형으로 보고 **코너와의 관계**를 더하면 순위가 좋아지나.

선수의 제안 (2026-09-25): *"세개의 공을 삼각형으로 나타내고 그중 삼각형의
중심과 제일 가까운 코너를 찾아 그 코너로부터 중심까지 거리를 정하면..."*

개수는 줄지 않는다 — 배치는 여섯 자유도이고 그것이 바닥이다. 그리고 오늘
**줄이는 것은 세 번 다 졌다** (NCA 3~4축 · 극좌표 6축 · 무게 학습).

그러나 제안의 새로움은 개수가 아니라 **코너**다. 지금 열두 축은 각 공이 단쿠션·
장쿠션에서 얼마나 떨어졌는지를 따로 셀 뿐, **코너와의 관계를 한 번도 보지
않는다.** 캐롬에서 코너는 실제로 중요하다 — 되돌아오기 21판이 전부 코너
샷이었다 (2026-09-25).

그러므로 **빼지 말고 더해서** 잰다.

    ~/.venvs/carom/bin/python tools/test_corner_features.py
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import learn_choices  # noqa: E402
import route_model  # noqa: E402
from learn_choices import (as_arrays, fit, load, where_it_landed,  # noqa: E402
                           with_neighbours, with_prior)
from route_model import player_features  # noqa: E402
from src.physics.table_calibration import (TABLE_LENGTH_MM as L,  # noqa: E402
                                           TABLE_WIDTH_MM as W)

CORNERS = np.array([[0.0, 0.0], [L, 0.0], [0.0, W], [L, W]])


def corner_features(row):
    """삼각형과 코너의 관계 — 넷.

    코너를 "가장 가까운 것"으로 고르므로 테이블의 네 겹 대칭이 저절로 풀린다.
    """
    cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
    others = [np.array(xy, dtype=float)
              for colour, xy in row["layout_mm"].items() if colour != row["cue"]]
    near, far = sorted(others, key=lambda xy: float(np.linalg.norm(xy - cue)))
    middle = (cue + near + far) / 3.0
    gaps = np.linalg.norm(CORNERS - middle, axis=1)
    corner = CORNERS[int(np.argmin(gaps))]
    to_corner = corner - middle
    # 삼각형의 넓이: 세 공이 뭉쳐 있나 퍼져 있나 (한 줄로 서면 0)
    a, b = near - cue, far - cue
    area = abs(float(a[0] * b[1] - a[1] * b[0])) / 2.0
    # 수구가 그 코너 쪽에 있나, 반대쪽에 있나
    span = float(np.linalg.norm(to_corner)) + 1e-9
    toward = float(np.dot(cue - middle, to_corner)) / span
    return np.array([
        float(np.min(gaps)),                                   # 코너에서 중심까지
        area,                                                  # 삼각형 넓이
        toward,                                                # 수구가 코너 쪽인가
        float(np.linalg.norm(cue - corner)),                   # 수구에서 그 코너까지
    ], dtype=float)


def both(row):
    return np.concatenate([player_features(row), corner_features(row)])


def score(builder, title):
    learn_choices.player_features = builder
    route_model.player_features = builder
    data = as_arrays(with_prior(with_neighbours(load())))
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
    print(f"  {title:<22} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
          f" · 자리 중앙값 {np.median(p):4.1f}   (n={len(p)})")
    return p


def main():
    print("프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)\n")
    a = score(player_features, "지금 (열두 축)")
    b = score(both, "코너 넷을 더해서")
    c = score(corner_features, "코너 넷만")
    n = min(len(a), len(b))
    a2, b2 = a[:n], b[:n]
    up, down = int(np.sum(b2 < a2)), int(np.sum(b2 > a2))
    if up + down:
        away = abs(up - (up + down) / 2) / math.sqrt((up + down) * 0.25)
        print(f"\n  코너를 더해 올라간 판 {up} · 내려간 판 {down} · 비긴 판 {n - up - down}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 더하는 것이 낫다" if away > 2 and up > down
                 else ("  → 더하지 않는 것이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
