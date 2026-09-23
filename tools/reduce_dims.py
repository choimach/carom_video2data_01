"""배치를 몇 개의 숫자로 줄인다 — 이웃 찾기가 막힌 진짜 이유를 푸는 일.

2026-09-23에 잰 것: 특징 12개를 표준화해 놓으면 **가장 가까운 이웃이 1.18
표준편차** 떨어져 있다. 축마다 0.34씩이다. 1,887판 중 가장 닮은 배치조차 1적구가
13 cm 다른 곳에 있다. **닮은 배치가 아니라 어렴풋이 비슷한 배치다.**

12차원에서 최근접 거리는 자료 개수의 **12제곱근**으로만 줄어든다. 절반으로
줄이려면 4,096배가 필요하고, 10,000판을 모아도 1.18이 1.03이 될 뿐이다. 그래서
자료도 특징 교체도 답이 아니었다.

★**남은 길은 차원을 줄이는 것이다.** 당구의 시스템들이 그렇게 한다 —
파이브앤하프는 숫자 두세 개로 길을 정한다. 3~4개로 줄이면 같은 자료가 그 공간을
빽빽하게 채우고, 그때 비로소 "이 배치에서 프로는 이렇게 친다"가 성립한다.

어느 3~4개인지는 손으로 고르지 않고 **자료가 고르게 한다**. 두 가지를 쓴다:

* **골라내기** — 12개 중 k개를 욕심껏 고른다. 어느 축이 쓸모 있는지 눈에 보인다.
* **NCA** — 12개를 k개로 **섞어** 내리는 행렬을 배운다. 목적함수가 곧
  "내 이웃이 나와 같은 선택을 했는가"라서, 우리가 재는 것을 그대로 최적화한다.

맞히려는 것은 **어느 공을 어느 면으로**다 — 선수가 정한 차례에서 ①②를 정하는 것.
경로 이름(5번)이 아니다.

    ~/.venvs/carom/bin/python tools/reduce_dims.py
    ~/.venvs/carom/bin/python tools/reduce_dims.py --dims 2 3 4 6
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "build", "app-data.json")
OUT = os.path.join(ROOT, "data", "layout_projection.json")
# route_model.player_features가 내놓는 차례. 무엇이 골라졌는지 사람이 읽으려고.
AXES = ["가까운공까지", "먼공까지", "두적구사이", "벌어진각",
        "가까운공방향", "먼공방향", "수구단쿠션", "수구장쿠션",
        "가까운공단쿠션", "가까운공장쿠션", "먼공단쿠션", "먼공장쿠션"]


def load():
    plays = [p for p in json.load(open(DATA, encoding="utf-8"))["plays"]
             if p.get("face") is not None]
    F = np.array([p["f"] for p in plays], dtype=float)
    label = np.array([f"{bool(p['near'])}|{p['face']}" for p in plays])
    match = np.array([p["match"] for p in plays])
    return F, label, match


def agreement(F, label, match, k=20):
    """이웃 k명 중 나와 같은 선택을 한 비율, 그리고 최다 선택의 몫(상한).

    같은 경기는 이웃에서 뺀다 — 같은 선수의 같은 버릇을 보고 맞혔다고 하게 된다.
    """
    same, top = [], []
    for i in range(len(F)):
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        near = np.argsort(gap)[:k]
        seen = label[near]
        if not len(seen):
            continue
        kinds, counts = np.unique(seen, return_counts=True)
        same.append(float(np.mean(seen == label[i])))
        top.append(float(counts.max()) / len(seen))
    return float(np.mean(same)), float(np.mean(top))


def near_distance(F, match):
    """가장 가까운 이웃까지의 거리 중앙값 — 공간이 얼마나 빽빽한가."""
    out = []
    for i in range(0, len(F), 3):
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        out.append(float(np.min(gap)))
    return float(np.median(out))


def nca(F, label, match, dims, rounds=150, step=0.02, seed=0):
    """이웃이 나와 같은 선택을 하도록 12개를 dims개로 내리는 행렬을 배운다.

    목적함수는 soft nearest neighbour: 가까울수록 무겁게 세어, 같은 선택을 한
    이웃의 몫을 키운다. 같은 경기는 빼고 센다.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.3, size=(dims, F.shape[1]))
    same = (label[:, None] == label[None, :])
    other = (match[:, None] != match[None, :])
    np.fill_diagonal(other, False)

    for _ in range(rounds):
        Y = F @ A.T
        d2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        d2[~other] = np.inf
        w = np.exp(-d2 - np.min(d2, axis=1, keepdims=True) * 0)
        w[~other] = 0.0
        total = w.sum(1, keepdims=True) + 1e-12
        p = w / total
        hit = (p * same).sum(1, keepdims=True)           # 맞은 몫

        # ∂(맞은 몫)/∂A — 같은 선택인 이웃은 당기고 아닌 이웃은 민다.
        pull = p * (same - hit)
        grad = np.zeros_like(A)
        for axis in range(dims):
            diff = Y[:, None, axis] - Y[None, :, axis]
            weighted = (pull * diff)
            grad[axis] = -2.0 * (weighted.sum(1)[:, None] * F
                                 - weighted @ F).sum(0) / len(F)
        A += step * grad
    return A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="*", default=[2, 3, 4, 6])
    parser.add_argument("--sample", type=int, default=900, help="NCA 학습에 쓸 판 수")
    args = parser.parse_args()

    F, label, match = load()
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    print(f"플레이 {len(F)}판 · 축 {F.shape[1]}개 · 선택 {len(set(label))}가지")
    kinds, counts = np.unique(label, return_counts=True)
    print(f"최빈 선택만 찍으면 {counts.max() / len(label):.1%}\n")

    base_same, base_top = agreement(F, label, match)
    print(f"  ── 지금 (12축) ──")
    print(f"     이웃과 일치 {base_same:.1%} · 상한 {base_top:.1%}"
          f" · 최근접 거리 {near_distance(F, match):.2f}\n")

    # 경기 단위로 나눈다. 같은 경기가 학습과 평가에 걸치면 외운 것을 맞혔다고 한다.
    matches = sorted(set(match))
    held = set(matches[::4])
    train = ~np.isin(match, list(held))
    keep = np.where(train)[0]
    if len(keep) > args.sample:
        keep = np.random.default_rng(1).choice(keep, args.sample, replace=False)

    best = None
    for dims in args.dims:
        A = nca(F[keep], label[keep], match[keep], dims)
        Z = F @ A.T
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        # 평가는 학습에 안 쓴 경기에서만.
        test = ~train
        same, top = agreement(Z[test], label[test], match[test])
        gap = near_distance(Z[test], match[test])
        print(f"  ── {dims}축으로 줄여 ──")
        print(f"     이웃과 일치 {same:.1%} · 상한 {top:.1%} · 최근접 거리 {gap:.2f}")
        if best is None or same > best[1]:
            best = (dims, same, top, A)

    # 견줄 자리: 같은 test 경기에서 12축 그대로면 얼마인가.
    test = ~train
    ref_same, ref_top = agreement(F[test], label[test], match[test])
    print(f"\n  ── 같은 test 경기에서 12축 그대로 ──")
    print(f"     이웃과 일치 {ref_same:.1%} · 상한 {ref_top:.1%}"
          f" · 최근접 거리 {near_distance(F[test], match[test]):.2f}")

    dims, same, top, A = best
    print(f"\n★ 가장 나은 것: {dims}축 · 일치 {same:.1%} (12축은 {ref_same:.1%})")
    json.dump({"note": "배치를 몇 개의 숫자로 내리는 행렬. tools/reduce_dims.py가 만든다.",
               "axes_in": AXES, "dims": dims,
               "matrix": [[round(float(v), 4) for v in row] for row in A]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
