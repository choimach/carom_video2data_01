"""차원은 그대로 두고 **무게만 배우면** 순위가 좋아지나.

2026-09-25에 두 가지가 드러났다:

* 열두 축에는 중복이 있고, 그 중복이 곧 **무게**다 (벌어진 각·쿠션거리가 두 번
  세어진다). 중복을 지워 여섯 축으로 만들면 순위가 **나빠진다** (5.8 표준편차).
  즉 그 무게가 옳았다.
* NCA로 3~4축까지 **줄이면** 이웃은 가까워지는데 답은 나아지지 않는다 (09-23).

그러면 남은 칸이 하나 있다 — **차원은 12로 두고 무게(와 축 섞음)만 배우기.**
그건 안 해 봤다. 지금 무게는 "중복을 몇 번 넣었나"로 우연히 정해져 있다.

학습과 평가는 **경기로 가른다**. 같은 경기가 양쪽에 걸치면 같은 테이블을 외운
다음 맞혔다고 하게 된다.

    ~/.venvs/carom/bin/python tools/test_learned_metric.py
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
from reduce_dims import nca  # noqa: E402
from route_model import player_features  # noqa: E402


def bucket_of(row):
    cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
    gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue))
            for c, xy in row["layout_mm"].items() if c != row["cue"]}
    first, side = row.get("first_object_ball"), row.get("struck_side")
    if not first or first not in gaps or side is None:
        return None
    return f"{gaps[first] == min(gaps.values())}|{'left' if side > 0 else 'right'}"


def main():
    import json
    rows = [r for r in json.load(open(os.path.join(ROOT, "data", "model.json"),
                                      encoding="utf-8"))
            if r.get("route") and r.get("layout_mm")]
    F = np.array([player_features(r) for r in rows], float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    match = np.array([r["match"] for r in rows])
    label = np.array([bucket_of(r) or "?" for r in rows])

    # 경기를 반씩 가른다. 무게는 앞쪽 절반에서만 배운다.
    names = sorted(set(match))
    teach = set(names[::2])
    mine = np.array([m in teach for m in match])
    keep = mine & (label != "?")
    pick = np.where(keep)[0]
    if len(pick) > 1100:
        pick = np.random.default_rng(0).choice(pick, 1100, replace=False)
    print(f"무게를 배우는 데 쓸 판 {len(pick)}개 (경기 {len(teach)}개)")
    A = nca(F[pick], label[pick], match[pick], dims=F.shape[1], rounds=120, step=0.02)
    print("배운 것: 12×12 행렬 (차원을 줄이지 않는다)\n")

    def run(builder_matrix, title):
        """이웃을 이 행렬 공간에서 찾고, 순위를 경기 단위 10겹으로 잰다."""
        def features_in_space(row):
            raw = (np.array(player_features(row), float) - RAW_MEAN) / RAW_STD
            return raw if builder_matrix is None else raw @ builder_matrix.T
        learn_choices.player_features = features_in_space
        route_model.player_features = features_in_space
        data = as_arrays(with_prior(with_neighbours(load())))
        # 무게를 배운 경기는 평가에서 뺀다
        data = [d for d in data if d[2].get("match") not in teach]
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
            for rowsX, chose, _one in te:
                place.append(where_it_landed(rowsX, chose, w))
        p = np.array(place, float)
        print(f"  {title:<22} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
              f" · 자리 중앙값 {np.median(p):4.1f}   (안 본 경기 {len(p)}판)")
        return p

    global RAW_MEAN, RAW_STD
    raw = np.array([player_features(r) for r in rows], float)
    RAW_MEAN, RAW_STD = raw.mean(0), raw.std(0) + 1e-9

    print("프로가 실제로 고른 길이 몇 번째에 오는가\n")
    a = run(None, "지금 (무게 그대로)")
    b = run(A, "무게를 배운 것")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    up, down = int(np.sum(b < a)), int(np.sum(b > a))
    if up + down:
        away = abs(up - (up + down) / 2) / math.sqrt((up + down) * 0.25)
        print(f"\n  배운 무게가 올린 판 {up} · 내린 판 {down} · 비긴 판 {n - up - down}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 배운 무게가 낫다" if away > 2 and up > down
                 else ("  → 지금이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
