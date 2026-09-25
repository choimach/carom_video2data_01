"""The first model: what route does a layout call for, and does it go in.

Two questions, kept apart because they are answered by different things and
fail in different ways.

* **Which route.** Given three balls and which one is the cue ball, which route
  did the professional choose? This is what the app has to answer, and the
  players' own choices are the only evidence there is.
* **Does it score.** Given that layout and that route, how often did it go in?
  This is what turns a list of candidate routes into a recommendation.

The model is nearest neighbours over layouts, which is the honest thing to fit
to twelve hundred plays: it has no parameters to overfit, it says which plays
an answer rests on, and where the data is thin it fails visibly rather than
confidently. Anything cleverer is for when there are ten thousand.

Distance treats the two object balls as interchangeable - the layout is the
same whether the red sits left or right of the other - so it is the better of
the two pairings. Layouts are already canonicalised into one quarter of the
table by `model_dataset.py`, which is what makes two mirrored shots find each
other at all.

⚠️ 대회전, 되돌아오기 and 횡단 are left out. All three are named by counting
cushions before the second object ball, so a play that missed cannot carry the
name, and their scoring rate is 100% by construction rather than by merit.

    ~/.venvs/carom/bin/python tools/route_model.py
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM  # noqa: E402

# Named by counting cushions before the second ball: the label carries the
# outcome, so they cannot be scored against.
OUTCOME_IN_NAME = ("대회전", "되돌아오기", "횡단")
NEIGHBOURS = 15


def features(row):
    """Cue ball, then the two object balls, in millimetres."""
    cue = row["layout_mm"][row["cue"]]
    others = [xy for colour, xy in row["layout_mm"].items() if colour != row["cue"]]
    return np.array(cue + others[0] + others[1], dtype=float)


def player_features(row):
    """The layout as a player would describe it, not as coordinates.

    Six numbers are what a table hands you; none of them is what anyone looks
    at. A player sees how far the first ball is, how open the angle between the
    two object balls is, and how much room each ball has to its rails - and
    those are the same numbers under a mirror, which raw coordinates are not
    until they are canonicalised.

    The two object balls are interchangeable, so they are ordered by distance
    from the cue ball: near one first.
    """
    cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
    others = [np.array(xy, dtype=float)
              for colour, xy in row["layout_mm"].items() if colour != row["cue"]]
    near, far = sorted(others, key=lambda xy: float(np.linalg.norm(xy - cue)))

    def room(point):
        """Distance to the nearest short rail and the nearest long rail."""
        return [min(point[0], TABLE_LENGTH_MM - point[0]),
                min(point[1], TABLE_WIDTH_MM - point[1])]

    to_near, to_far = near - cue, far - cue
    opening = np.degrees(np.arctan2(
        to_near[0] * to_far[1] - to_near[1] * to_far[0],
        float(np.dot(to_near, to_far))))
    return np.array([
        np.linalg.norm(to_near), np.linalg.norm(to_far),
        np.linalg.norm(far - near),
        opening,
        np.degrees(np.arctan2(to_near[1], to_near[0])),
        np.degrees(np.arctan2(to_far[1], to_far[0])),
        *room(cue), *room(near), *room(far),
    ], dtype=float)


def polar_features(row):
    """같은 배치를 겹침 없이 여섯 개로 — 극좌표.

    2026-09-25에 `player_features`의 열두 축을 재 보니 여섯 개가 **나머지에서
    그대로 계산된다** (차이 0.000):

      축 3 (적구 사이 거리)  = 축 1·2·5·6에서 코사인법칙
      축 4 (벌어진 각)       = 축 6 − 축 5
      축 9~12 (적구의 쿠션거리) = 수구 자리 + 각 공의 거리·방향

    배치는 공 셋 × 좌표 둘 = **여섯 자유도**뿐이고, 캐노니컬로 뒤집기를 없앴으니
    딱 여섯이다. 중복은 이웃 거리에서 **같은 사실을 두 번 세게** 만든다 —
    표준화하고 유클리드 거리를 재면 그 방향의 무게가 말없이 2배가 된다.

    ⚠️ PCA로는 안 보인다 (95%에 10개가 필요하다고 나온다). 중복이 코사인법칙
    이라 비선형이고 PCA는 선형 구조만 본다. 그래서 여태 드러나지 않았다.

    ⚠️⚠️ **그런데 이것을 쓰면 순위가 나빠진다. 쓰지 않는다.**
    (`tools/test_polar_features.py`, 2026-09-25, 프로 2,095판 · 경기 단위 10겹)

        열두 축 (지금)     1등 33.7% · 3등 안 60.6%
        극좌표 여섯 축      1등 31.8% · 3등 안 57.7%
        올린 판 471 · 내린 판 667 · 5.8 표준편차 → 열두 축이 낫다

    이웃은 **가까워진다** (아무 두 판 대비 최근접 거리 0.230 → 0.167). 그런데
    순위는 나빠진다. **"중복"이 곧 가중치였고, 그 가중치가 옳았다** — 적구가
    벌어진 정도와 공이 쿠션에 붙은 정도를 두 번 세는 것이, 수구의 절대 좌표와
    같은 무게로 한 번씩 세는 것보다 낫다. 선수가 실제로 보는 것이 그쪽이다.

    2026-09-23의 NCA와 같은 결론이다: **이웃을 가깝게 만드는 것과 좋은 답을
    내는 것은 다른 일이다.** 남겨 두는 것은 다시 이 생각을 할 사람을 위해서다.
    """
    cue = np.array(row["layout_mm"][row["cue"]], dtype=float)
    others = [np.array(xy, dtype=float)
              for colour, xy in row["layout_mm"].items() if colour != row["cue"]]
    near, far = sorted(others, key=lambda xy: float(np.linalg.norm(xy - cue)))
    to_near, to_far = near - cue, far - cue
    heading = np.degrees(np.arctan2(to_near[1], to_near[0]))
    opening = np.degrees(np.arctan2(
        to_near[0] * to_far[1] - to_near[1] * to_far[0],
        float(np.dot(to_near, to_far))))
    return np.array([
        # 수구가 어디 서 있나 (캐노니컬이라 가까운 쿠션까지의 거리가 곧 자리다)
        min(cue[0], TABLE_LENGTH_MM - cue[0]), min(cue[1], TABLE_WIDTH_MM - cue[1]),
        # 두 적구를 수구에서 본 극좌표
        float(np.linalg.norm(to_near)), heading,
        float(np.linalg.norm(to_far)), opening,
    ], dtype=float)


def scaled(rows, builder):
    """Features, each standardised on the training set's own spread."""
    table = np.array([builder(r) for r in rows])
    return table


def distance(one, many):
    """Layout distance, with the two object balls interchangeable."""
    cue = np.linalg.norm(many[:, 0:2] - one[0:2], axis=1)
    straight = (np.linalg.norm(many[:, 2:4] - one[2:4], axis=1)
                + np.linalg.norm(many[:, 4:6] - one[4:6], axis=1))
    swapped = (np.linalg.norm(many[:, 2:4] - one[4:6], axis=1)
               + np.linalg.norm(many[:, 4:6] - one[2:4], axis=1))
    return cue + np.minimum(straight, swapped)


def neighbours(one, many, k, kind="player"):
    gaps = (np.linalg.norm(many - one, axis=1) if kind == "player"
            else distance(one, many))
    order = np.argsort(gaps)[:k]
    return order, gaps[order]


def load(path):
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r["route"] not in OUTCOME_IN_NAME]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    return train, test


def report(train, test, k=NEIGHBOURS, kind="player"):
    builder = player_features if kind == "player" else features
    x_train = np.array([builder(r) for r in train])
    if kind == "player":
        middle, spread = x_train.mean(axis=0), x_train.std(axis=0)
        spread[spread == 0] = 1.0
        x_train = (x_train - middle) / spread
    routes = [r["route"] for r in train]
    scored = np.array([r["scored"] for r in train], dtype=float)

    common = max(set(routes), key=routes.count)
    base_rate = scored.mean()

    right = 0
    within = {2: 0, 3: 0}
    brier_model, brier_base = [], []
    for row in test:
        point = builder(row)
        if kind == "player":
            point = (point - middle) / spread
        order, gaps = neighbours(point, x_train, k, kind)
        votes = {}
        for index, gap in zip(order, gaps):
            weight = 1.0 / (1.0 + gap / (1.0 if kind == "player" else 100.0))
            votes[routes[index]] = votes.get(routes[index], 0.0) + weight
        ranked = [name for name, _ in sorted(votes.items(), key=lambda kv: -kv[1])]
        right += ranked[0] == row["route"]
        for place in (2, 3):
            within[place] += row["route"] in ranked[:place]

        same = [i for i in order if routes[i] == row["route"]]
        chance = scored[same].mean() if same else base_rate
        brier_model.append((chance - row["scored"]) ** 2)
        brier_base.append((base_rate - row["scored"]) ** 2)

    print(f"train {len(train)} · test {len(test)} · k={k} · 특징 {kind}")
    print()
    print("어떤 경로를 고를 것인가")
    print(f"  이웃 투표        {right / len(test) * 100:5.1f}%")
    majority = sum(1 for r in test if r["route"] == common) / len(test) * 100
    print(f"  가장 흔한 유형만 {majority:5.1f}%   ({common})")
    # What an assistant would actually show: a short list, not one answer. The
    # notes are full of how to choose between two candidates - 걸어치기 against
    # 빗겨치기, 안으로 돌리기 against 뒤돌리기 - so the list is the useful
    # output and the choosing is where a player's own knowledge goes.
    print(f"  상위 2개 안        {within[2] / len(test) * 100:5.1f}%")
    print(f"  상위 3개 안        {within[3] / len(test) * 100:5.1f}%")
    print()
    print("그 경로가 득점할 것인가 (낮을수록 좋음, Brier)")
    print(f"  이웃 평균        {np.mean(brier_model):.4f}")
    print(f"  전체 평균만      {np.mean(brier_base):.4f}   (득점률 {base_rate * 100:.0f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "model.json"))
    parser.add_argument("-k", type=int, default=NEIGHBOURS)
    parser.add_argument("--features", choices=("player", "raw"), default="player")
    args = parser.parse_args(argv)
    train, test = load(args.data)
    report(train, test, args.k, args.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
