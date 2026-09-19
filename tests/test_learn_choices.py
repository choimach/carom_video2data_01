"""고른 것과 버린 것에서 배우는 쪽이 실제로 배우는지 확인한다.

여기서 지켜야 하는 것은 정확도가 아니라 **방향**이다. 순위를 배우는 코드는
조용히 아무것도 안 배워도 그럴듯한 숫자를 내놓는다 — 갈래가 몇 개 안 되는
판이 섞이면 아무 가중치나 30%쯤은 1등을 맞히기 때문이다. 그래서 답이 뻔한
자료를 만들어 두고, 거기서 못 배우면 걸리게 한다.
"""

import numpy as np

from tools.learn_choices import features, fit, where_it_landed


def a_branch(**over):
    branch = {"room": 1.0, "thickness": 0.3, "strength": 4.0, "rails": 3,
              "pushed": 3000, "lines": 20, "side": 0.0, "up": 0.0}
    branch.update(over)
    return branch


def test_every_feature_comes_out_a_number():
    row = features(a_branch(), {"white": [0, 0]}, "white")
    assert all(np.isfinite(row)), "특징에 숫자가 아닌 것이 있습니다"
    assert row[-1] == 1.0, "기준선이 사라졌습니다"


def test_it_learns_a_preference_that_is_actually_there():
    # 만든 규칙: 여유가 넓은 갈래를 고른다. 배우는 쪽이 이것을 못 찾으면
    # 진짜 자료에서도 못 찾는다.
    rounds = []
    rng = np.random.default_rng(7)
    for _ in range(60):
        rooms = rng.uniform(0.25, 8.0, size=5)
        rows = np.array([features(a_branch(room=r), None, None) for r in rooms])
        rounds.append((rows, int(np.argmax(rooms)), {"split": "train"}))
    weight = fit(rounds, rounds_of_weight=250)
    places = [where_it_landed(rows, chose, weight) for rows, chose, _ in rounds]
    assert np.mean(np.array(places) == 1) > 0.8, \
        f"있는 경향도 못 배웠습니다 (1등 비율 {np.mean(np.array(places) == 1):.0%})"
    assert weight[0] > 0, "여유가 넓을수록 고른다는 것을 반대로 배웠습니다"


def test_it_does_not_invent_a_preference_that_is_not_there():
    # 고른 것이 무작위면 배울 것이 없다. 그런데도 크게 기울면, 배우는 것이
    # 아니라 외우는 것이다.
    rng = np.random.default_rng(11)
    rounds = []
    for _ in range(60):
        rows = np.array([features(a_branch(room=float(r)), None, None)
                         for r in rng.uniform(0.25, 8.0, size=5)])
        rounds.append((rows, int(rng.integers(0, 5)), {"split": "train"}))
    weight = fit(rounds, rounds_of_weight=250)
    assert abs(weight[0]) < 1.5, f"없는 경향을 지어냈습니다 (여유 {weight[0]:+.2f})"
