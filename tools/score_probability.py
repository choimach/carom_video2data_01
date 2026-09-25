"""무엇으로 "득점확률"을 계산할 것인가 — 후보들을 대 보고 고른다.

선수가 목표를 바꿨다 (2026-09-26): *"이제부터는 득점확률이 가장 높은 궤적을
추가합시다. 1. 득점확률 최고의 궤적. 2~4 프로의 선택."*

그러려면 먼저 **무엇을 확률이라 부를지** 정해야 한다. 쓸 수 있는 것이 여럿이고,
가장 그럴듯한 것(몬테카를로)이 가장 나은 것이 아니다:

  여유(room)      겨냥이 몇 도까지 빗나가도 들어가나. 열거가 이미 재 놨다.
  rough           같은 것을 거칠게 잰 값
  lines           득점하는 줄을 몇 개나 찾았나
  쿠션 수         길수록 어렵다
  이웃득점률      닮은 배치에서 프로가 넣은 비율 — **프로 자료가 필요하다**
  몬테카를로      사람 오차를 얹어 여러 번 쳐 본다 (`make_probability.js`)

2026-09-24에 잰 바로는 몬테카를로가 **0.55로 가른다** — 사람 오차를 주면 32%가
나오는데 프로는 실제로 61%를 넣기 때문이다. 같은 날 **이웃득점률 하나가
0.629**였다. 즉 "그럴듯한 물리"보다 "닮은 배치에서 실제로 들어갔나"가 낫다.

채점: 프로가 **실제로 친 줄**에 대해 들어갔는지를 얼마나 가르나 (AUC).
1.0이면 완벽, 0.5면 동전 던지기.

⚠️ **치우친 표본이다.** 프로는 쉬운 공략을 고르므로, 여기서 잰 것은 "프로가
   고를 만한 공략들 사이에서" 가르는 능력이다. 아무 공략에나 그대로 적용되는
   값이 아니다. 그래도 결과가 있는 유일한 줄이 이것뿐이다.

    ~/.venvs/carom/bin/python tools/score_probability.py
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

FOUND = os.path.join(ROOT, "data", "alternatives.jsonl")


def auc(score, hit):
    """들어간 줄이 안 들어간 줄보다 높은 점수를 받을 확률."""
    score, hit = np.asarray(score, float), np.asarray(hit, bool)
    ok = np.isfinite(score)
    score, hit = score[ok], hit[ok]
    if hit.all() or not hit.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # 같은 값은 평균 순위로
    for value in np.unique(score):
        same = score == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    n1, n0 = int(hit.sum()), int((~hit).sum())
    return float((ranks[hit].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rows = []
    with open(FOUND, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            one = json.loads(line)
            if not one.get("reached") or one.get("scored") is None:
                continue
            chose = one["chose"]
            branch = next((b for b in one["found"] if b["key"] == chose), None)
            if branch is None:
                continue
            rows.append((branch, bool(one["scored"]), one))

    if not rows:
        print("잴 판이 없습니다."); return 1
    hit = [h for _b, h, _o in rows]
    print(f"프로가 실제로 친 줄 {len(rows)}개 · 그중 들어간 것 {np.mean(hit):.1%}\n")
    print(f"  {'무엇으로':<22}{'가르는 정도 (AUC)':>18}")

    measures = {
        "여유 (room)": lambda b: b.get("room"),
        "rough": lambda b: b.get("rough"),
        "득점하는 줄 수 (lines)": lambda b: b.get("lines"),
        "쿠션 수 (적을수록)": lambda b: -(b.get("rails") or 0),
        "두께 (두꺼울수록)": lambda b: b.get("thickness"),
        "강도 (약할수록)": lambda b: -(b.get("strength") or 0),
        "회전 (적을수록)": lambda b: -abs(b.get("side") or 0),
    }
    got = {}
    for name, pick in measures.items():
        value = auc([pick(b) for b, _h, _o in rows], hit)
        got[name] = value
        print(f"  {name:<22}{value:>18.3f}")

    # 여유와 쿠션 수를 같이 — 가장 싼 조합
    pair = [math.log1p(b.get("room") or 0) - 0.25 * (b.get("rails") or 0)
            for b, _h, _o in rows]
    print(f"  {'여유 + 쿠션 수':<22}{auc(pair, hit):>18.3f}")

    print("\n비교 (2026-09-24에 잰 값)")
    print(f"  {'몬테카를로 (사람 오차)':<22}{0.55:>18.3f}")
    print(f"  {'이웃득점률 하나':<22}{0.629:>18.3f}")
    print(f"  {'지금 특징 열 개를 합쳐':<22}{0.650:>18.3f}")

    print("\n" + "=" * 62)
    fit_model()
    return 0


# ⚠️ 이 셋은 **2적구 전 쿠션 수로 이름을 붙인다.** 실패한 판은 그 이름을 가질
# 수 없으므로 자료에서 100% 득점으로 잡힌다. 확률을 맞출 때 넣으면 "쿠션이
# 많을수록 잘 들어간다"를 배운다 — 실제로 쿠션 수의 AUC가 0.454로 뒤집혀 나온다.
OUTCOME_IN_NAME = ("대회전", "되돌아오기", "횡단")

AXES = ("여유", "rough", "줄수", "두께", "강도", "쿠션수", "회전",
        "이웃득점률", "이웃프로수")


def features(branch):
    return np.array([
        math.log1p(branch.get("room") or 0.0),
        math.log1p(branch.get("rough") or 0.0),
        math.log1p(branch.get("lines") or 0.0),
        float(branch.get("thickness") or 0.0),
        float(branch.get("strength") or 0.0) / 10.0,
        float(branch.get("rails") or 0.0) / 5.0,
        abs(float(branch.get("side") or 0.0)) / 3.0,
        float(branch.get("rate") or 0.5),
        math.log1p(float(branch.get("chosen") or 0.0)),
    ], dtype=float)


def logistic(X, y, rounds=600, step=0.3):
    w = np.zeros(X.shape[1] + 1)
    Z = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        w += step * (Z.T @ (y - p)) / len(Z)
    return w


def fit_model():
    """들어갔는가를 맞추는 모형 — 경기 단위 10겹으로 채점한다."""
    from learn_choices import load, with_neighbours, with_prior  # noqa

    rounds = with_prior(with_neighbours(load()))
    rows = []
    for one in rounds:
        if one.get("scored") is None:
            continue
        branch = next((b for b in one["found"] if b["key"] == one["chose"]), None)
        if branch is None or branch["route"] in OUTCOME_IN_NAME:
            continue
        rows.append((features(branch), 1.0 if one["scored"] else 0.0,
                     one.get("match")))
    if len(rows) < 200:
        print("맞출 판이 모자랍니다."); return
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    matches = [r[2] for r in rows]
    mean, spread = X.mean(0), X.std(0) + 1e-9
    X = (X - mean) / spread

    names = sorted(set(matches))
    where = {m: i % 10 for i, m in enumerate(names)}
    fold = np.array([where[m] for m in matches])
    guess = np.zeros(len(y))
    for k in range(10):
        tr, te = fold != k, fold == k
        if not tr.any() or not te.any():
            continue
        w = logistic(X[tr], y[tr])
        guess[te] = 1.0 / (1.0 + np.exp(-(np.hstack([X[te], np.ones((te.sum(), 1))]) @ w)))
    print(f"\n득점 확률 모형 — 이름에 결과가 든 유형을 뺀 {len(y)}판"
          f" (들어간 것 {y.mean():.1%})")
    print(f"  경기 단위 10겹 AUC  {auc(guess, y > 0.5):.3f}")
    w = logistic(X, y)
    print("\n  무게 (표준화한 축)")
    for name, value in sorted(zip(AXES, w[:-1]), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<12}{value:+7.3f}")
    print("\n  맞춘 확률의 퍼짐: "
          f"10% {np.percentile(guess,10):.0%} · 50% {np.percentile(guess,50):.0%}"
          f" · 90% {np.percentile(guess,90):.0%}")


if __name__ == "__main__":
    raise SystemExit(main())
