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

# Named by counting cushions before the second ball: the label carries the
# outcome, so they cannot be scored against.
OUTCOME_IN_NAME = ("대회전", "되돌아오기", "횡단")
NEIGHBOURS = 15


def features(row):
    """Cue ball, then the two object balls, in millimetres."""
    cue = row["layout_mm"][row["cue"]]
    others = [xy for colour, xy in row["layout_mm"].items() if colour != row["cue"]]
    return np.array(cue + others[0] + others[1], dtype=float)


def distance(one, many):
    """Layout distance, with the two object balls interchangeable."""
    cue = np.linalg.norm(many[:, 0:2] - one[0:2], axis=1)
    straight = (np.linalg.norm(many[:, 2:4] - one[2:4], axis=1)
                + np.linalg.norm(many[:, 4:6] - one[4:6], axis=1))
    swapped = (np.linalg.norm(many[:, 2:4] - one[4:6], axis=1)
               + np.linalg.norm(many[:, 4:6] - one[2:4], axis=1))
    return cue + np.minimum(straight, swapped)


def neighbours(one, many, k):
    gaps = distance(one, many)
    order = np.argsort(gaps)[:k]
    return order, gaps[order]


def load(path):
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r["route"] not in OUTCOME_IN_NAME]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    return train, test


def report(train, test, k=NEIGHBOURS):
    x_train = np.array([features(r) for r in train])
    routes = [r["route"] for r in train]
    scored = np.array([r["scored"] for r in train], dtype=float)

    common = max(set(routes), key=routes.count)
    base_rate = scored.mean()

    right = 0
    brier_model, brier_base = [], []
    for row in test:
        order, gaps = neighbours(features(row), x_train, k)
        votes = {}
        for index, gap in zip(order, gaps):
            votes[routes[index]] = votes.get(routes[index], 0.0) + 1.0 / (1.0 + gap / 100.0)
        guess = max(votes, key=votes.get)
        right += guess == row["route"]

        same = [i for i in order if routes[i] == row["route"]]
        chance = scored[same].mean() if same else base_rate
        brier_model.append((chance - row["scored"]) ** 2)
        brier_base.append((base_rate - row["scored"]) ** 2)

    print(f"train {len(train)} · test {len(test)} · k={k}")
    print()
    print("어떤 경로를 고를 것인가")
    print(f"  이웃 투표        {right / len(test) * 100:5.1f}%")
    majority = sum(1 for r in test if r["route"] == common) / len(test) * 100
    print(f"  가장 흔한 유형만 {majority:5.1f}%   ({common})")
    print()
    print("그 경로가 득점할 것인가 (낮을수록 좋음, Brier)")
    print(f"  이웃 평균        {np.mean(brier_model):.4f}")
    print(f"  전체 평균만      {np.mean(brier_base):.4f}   (득점률 {base_rate * 100:.0f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "model.json"))
    parser.add_argument("-k", type=int, default=NEIGHBOURS)
    args = parser.parse_args(argv)
    train, test = load(args.data)
    report(train, test, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
