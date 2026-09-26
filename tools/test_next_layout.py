"""친 **뒤**의 배치가 순위를 좋게 하나 — 선수가 말한 포지션 플레이.

선수 (2026-09-26): *"충돌 후 수구, 1적구, 2적구가 어떻게 움직여서 최종위치가
어디에 있는가도 매우 중요해. 프로들은 이를 이용해서 키스를 떼고, 다음번 공략을
쉽게 만들고, 또한 실패했을 경우 상대 선수의 공략을 어렵게 만들 수 있어."*

★2026-09-26에 진 네 시도(NCA · 극좌표 · 코너 · 결과서명)는 **전부 지금 배치를
다시 적은 것**이었다. 친 뒤의 배치는 `player_features`의 열두 축에 아예 없다.
오늘 처음으로 **새 정보**를 넣는 실험이다.

자료가 이미 말하고 있다 (같은 날, `HANDOFF.md`):

    샷 1번(상대가 남긴 배치) 득점 55.1% → 5번+ 67.3%
    이웃으로 잰 '쉬움': 상대가 남긴 0.563 vs 자기가 남긴 0.591 (7.5 SD)

여기서 재는 것은 그 다음 단계다 — **후보마다 친 뒤의 배치가 얼마나 쉬운지를
붙이면, 프로가 실제로 고른 것을 더 잘 맞히나.**

## 쉬움을 어떻게 재나

물리를 쓰지 않는다. 친 뒤의 세 공 자리를 **`player_features`로 바꿔 프로 배치
2,466개에서 이웃 40명을 찾고, 그들이 얼마나 넣었나**를 본다. 같은 경기는 뺀다.

⚠️ **수구가 누구인지가 바뀐다.** 득점하면 같은 선수가 같은 수구로 이어 친다.
   그러므로 다음 배치의 수구는 **이 판과 같은 색**이다.

⚠️ **'쉬움' 자체가 약하다** — 실제 득점을 가르는 정도가 0.542뿐이다. 순위가
   안 올라도 이상하지 않다. 그래도 **새 정보**라는 점이 다르다.

★★ **쟀다. 가릴 수 없다 — 그런데 이것은 "안 된다"가 아니라 "못 잰다"이다.**
(2026-09-26, 프로 2,095판 · 후보 38,804개 · 경기 단위 10겹)

    프로가 고른 줄의 다음 배치 쉬움  0.6069
    프로가 버린 줄의 다음 배치 쉬움  0.6083   차이 −0.0014 · 0.7 표준편차

    지금 (열두 축)       1등 33.7% · 3등 안 60.6%
    다음 배치를 더해서    1등 33.7% · 3등 안 60.8%   0.5 표준편차

⚠️ **부정 결과로 적으면 안 된다.** 같은 날 잰 것과 정면으로 어긋난다 —
프로가 자기에게 남긴 배치(0.591)가 상대가 남긴 배치(0.563)보다 쉽다, 7.5 SD.
프로가 포지션을 본다는 증거는 **있다.** 우리가 **예측한** 다음 배치에서만
안 보인다.

★설명은 하나가 압도적이다: **시뮬이 공이 어디 멈추는지를 못 맞힌다.**

    1초 뒤 위치 오차 319 mm · 2초 뒤 594 mm
    1적구를 아예 못 맞히는 판 49%
    첫 쿠션 나간 각 흩어짐 ±13.3도

캐롬 한 샷은 4~8초다. 끝 자리 오차는 600 mm를 넘을 것이고 테이블 대각선의
20%다. 공 셋을 각각 그만큼 틀리게 놓은 배치에서 '쉬움'을 잰 것이라, 0.028짜리
신호가 살아남을 수 없다.

**그러므로 물리를 고친 뒤에 다시 돌릴 것.** 이 도구는 그대로 쓸 수 있다.

    ~/.venvs/carom/bin/python tools/test_next_layout.py
"""

import json
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
from model_dataset import canonical  # noqa: E402
from route_model import player_features  # noqa: E402

MODEL = os.path.join(ROOT, "data", "model.json")
ORDER = ("white", "yellow", "red")
NEAR = 40


def pool():
    rows = json.load(open(MODEL, encoding="utf-8"))
    rows = [r for r in rows if r.get("route") and r.get("layout_mm")]
    F = np.array([player_features(r) for r in rows], dtype=float)
    mean, spread = F.mean(0), F.std(0) + 1e-9
    return (rows, (F - mean) / spread, mean, spread,
            np.array([r["match"] for r in rows]),
            np.array([bool(r.get("scored")) for r in rows]))


ROWS, F, MEAN, SPREAD, MATCH, HIT = pool()


def ease(rest, cue, match):
    """친 뒤의 배치가 얼마나 쉬운가 — 닮은 프로 배치 40개가 얼마나 넣었나."""
    if not rest or any(v is None for pair in rest for v in pair):
        return None
    layout = {colour: list(map(float, xy)) for colour, xy in zip(ORDER, rest)}
    # ★캐노니컬로 다시 돌려야 한다. model.json의 배치는 **판마다 따로** 수구가
    # 한 사분면에 오도록 뒤집혀 있는데, `rest`는 **이 판의** 틀에 있다. 다음
    # 배치의 수구 자리가 다른 사분면이면 거울이 달라지므로, 그대로 견주면
    # 닮은 배치를 못 찾는다.
    layout, _turn, _flip = canonical(layout, cue)
    here = player_features({"cue": cue, "layout_mm": layout})
    here = (here - MEAN) / SPREAD
    gap = np.linalg.norm(F - here, axis=1)
    gap[MATCH == match] = np.inf
    near = np.argsort(gap)[:NEAR]
    if not np.isfinite(gap[near[0]]):
        return None
    return float(HIT[near].mean())


def attach(rounds):
    got = missing = 0
    for one in rounds:
        for branch in one["found"]:
            value = ease(branch.get("rest"), one["cue"], one.get("match"))
            branch["next_ease"] = 0.5 if value is None else value
            got += value is not None
            missing += value is None
    print(f"  다음 배치를 잰 후보 {got}개 · 못 잰 것 {missing}개")
    return rounds


def features_with_next(branch, layout=None, cue=None):
    base = learn_choices._features_plain(branch, layout, cue)
    return np.concatenate([base[:-1],
                           [branch.get("next_ease", 0.5)],
                           base[-1:]])


def score(builder, title, rounds):
    learn_choices.features = builder
    data = as_arrays(with_prior(rounds))
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
    print(f"  {title:<24} 1등 {np.mean(p == 1):5.1%} · 3등 안 {np.mean(p <= 3):5.1%}"
          f" · 자리 중앙값 {np.median(p):4.1f}   (n={len(p)})")
    return p


def main():
    rounds = attach(with_neighbours(load()))

    # 먼저 맨눈으로: 프로가 고른 줄의 다음 배치가 버린 줄보다 쉬운가?
    mine, theirs = [], []
    for one in rounds:
        for branch in one["found"]:
            (mine if branch["key"] == one["chose"] else theirs).append(branch["next_ease"])
    mine, theirs = np.array(mine), np.array(theirs)
    gap = mine.mean() - theirs.mean()
    spread = math.sqrt(mine.var() / len(mine) + theirs.var() / len(theirs))
    print(f"\n프로가 고른 줄의 다음 배치 쉬움 {mine.mean():.4f} ({len(mine)}개)")
    print(f"프로가 버린 줄의 다음 배치 쉬움 {theirs.mean():.4f} ({len(theirs)}개)")
    print(f"  차이 {gap:+.4f} · {abs(gap) / spread:.1f} 표준편차"
          + ("  → 고른 쪽이 쉽다" if gap > 0 else "  → 고른 쪽이 어렵다"))

    print("\n프로가 실제로 고른 길이 몇 번째에 오는가 (경기 단위 10겹)\n")
    learn_choices._features_plain = learn_choices.features
    learn_choices.NAMES = list(learn_choices.NAMES)
    a = score(learn_choices._features_plain, "지금 (열두 축)", rounds)
    b = score(features_with_next, "다음 배치를 더해서", rounds)
    n = min(len(a), len(b))
    up, down = int(np.sum(b[:n] < a[:n])), int(np.sum(b[:n] > a[:n]))
    if up + down:
        away = abs(up - (up + down) / 2) / math.sqrt((up + down) * 0.25)
        print(f"\n  올라간 판 {up} · 내려간 판 {down} · 비긴 판 {n - up - down}")
        print(f"  우연으로 보기 어려운 정도 {away:.1f} 표준편차"
              + ("  → 더하는 것이 낫다" if away > 2 and up > down
                 else ("  → 더하지 않는 것이 낫다" if away > 2 else "  → 가릴 수 없다")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
