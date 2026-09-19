"""브라우저의 이름 붙이기와 파이썬의 이름 붙이기가 같은 이름을 내는지.

한때 이 판정이 **세 벌**로 갈라져 있었다. 조언판에는 되돌아오기가 있고
대회전·횡단이 없었고, `tools/enumerate_alternatives.js`에는 대회전만 있었고,
`src/physics/route.py`에는 셋 다 있었다. 증상은 조용했다 — 프로가 되돌아오기로
친 12판을 열거기가 한 판도 재현하지 못했고, 그게 "탐색이 못 찾았다"로 보였다.
못 찾은 것이 아니라 **그 이름을 만들 줄 몰랐던** 것이다.

이제 JS는 `src/visualization/assistant/route.js` 한 벌이고, 이 시험이 그것을
파이썬에 붙들어 둔다. 쿠션 차례를 만들어 두 쪽에 똑같이 물어본다.

node가 없으면 건너뛴다.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.physics import route as py_route

ROUTE_JS = Path(__file__).resolve().parent.parent / "src" / "visualization" / "assistant" / "route.js"

SHORT, LONG = ("left", "right"), ("top", "bottom")

# (before, between, reached_second) — 판정을 가르는 자리마다 한 벌씩.
CASES = [
    (2, ["top", "left", "bottom"], True),            # 뱅크샷
    (1, ["top", "left", "bottom"], True),            # 걸어치기
    (0, ["top", "left", "top", "right", "bottom"], True),   # 대회전 (5쿠션)
    (0, ["top", "left", "top"], True),               # 되돌아오기 (장-단-장)
    (0, ["left", "top", "left"], True),              # 되돌아오기 (단-장-단)
    (0, ["top", "bottom", "top"], True),             # 횡단이 아니라 되돌아오기
    (0, ["top", "bottom", "top", "bottom"], True),   # 위와 같은 앞 세 개
    (0, ["bottom", "top", "bottom"], True),
    (0, ["top", "left", "bottom"], True),            # 장쿠션 먼저
    (0, ["left", "top", "bottom"], True),            # 단쿠션 먼저
    (0, ["left", "bottom", "top"], True),
    (0, ["top", "left"], True),                      # 3쿠션 미만
    (0, ["top", "left", "bottom"], False),           # 2적구에 못 닿음
]
FACES = ("left", "right")
CIRCUITS = (True, False)


class Event:
    """`classify`가 읽는 모양만 흉내 낸다 — kind, detail, frame."""

    def __init__(self, kind, detail, frame):
        self.kind, self.detail, self.frame = kind, detail, frame


def python_name(before, between, reached_second, face, circuit_right):
    events = []
    frame = 0
    for rail in LONG[:1] * before if before else []:
        pass
    for i in range(before):
        frame += 1
        events.append(Event("cushion", LONG[i % 2], frame))
    frame += 1
    events.append(Event("ball", "red", frame))
    for rail in between:
        frame += 1
        events.append(Event("cushion", rail, frame))
    if reached_second:
        frame += 1
        events.append(Event("ball", "yellow", frame))
    # struck_side: 양수가 왼쪽 면 (app_data.py와 같은 규약).
    got = py_route.classify(events, struck_side=(1.0 if face == "left" else -1.0),
                            circuit=(1.0 if circuit_right else -1.0))
    return got["route"]


def js_names(cases):
    script = f"""
    const ROUTE = require({json.dumps(str(ROUTE_JS))});
    const cases = {json.dumps(cases, ensure_ascii=False)};
    console.log(JSON.stringify(cases.map((c) => ROUTE.name({{
      before: c[0], between: c[1], reachedSecond: c[2],
      face: c[3], circuitRight: c[4],
    }}))));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_the_two_languages_name_the_same_route():
    cases, expected = [], []
    for before, between, reached in CASES:
        for face in FACES:
            for circuit in CIRCUITS:
                cases.append([before, between, reached, face, circuit])
                expected.append(python_name(before, between, reached, face, circuit))

    got = js_names(cases)
    wrong = []
    for case, mine, theirs in zip(cases, expected, got):
        # 파이썬은 못 정하면 "미분류"를, JS는 null을 쓴다. 같은 뜻으로 본다.
        if mine == py_route.UNKNOWN and theirs is None:
            continue
        if mine != theirs:
            wrong.append(f"  {case} → 파이썬 {mine!r} · JS {theirs!r}")
    assert not wrong, "두 쪽이 다른 이름을 붙입니다:\n" + "\n".join(wrong)


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_the_names_the_javascript_used_to_be_unable_to_produce():
    """되돌아오기와 횡단 — 이 둘이 없어서 17판이 재현되지 않았다."""
    got = js_names([
        [0, ["top", "left", "top"], True, "left", True],
        [0, ["left", "top", "left"], True, "right", False],
        [0, ["top", "bottom", "left"], True, "left", True],
    ])
    assert got[0] == "되돌아오기"
    assert got[1] == "되돌아오기"
    assert got[2] is not None
