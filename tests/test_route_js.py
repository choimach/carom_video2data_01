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
    (0, ["top", "bottom", "top"], True),             # 횡단 (장쿠션 3회)
    (0, ["top", "bottom", "top", "bottom"], True),   # 횡단 (4회)
    (0, ["top", "bottom", "top", "bottom", "top"], True),  # 횡단 — 대회전보다 먼저
    (0, ["bottom", "top", "bottom"], True),
    (0, ["left", "right", "left"], True),            # 횡단 (단쿠션끼리, 드물다)
    (0, ["top", "bottom", "left"], True),            # 더블 (2회 뒤 단쿠션)
    (0, ["left", "right", "top"], True),             # 더블 (단쿠션 2회 뒤 장쿠션)
    (0, ["top", "left", "bottom", "right", "top"], True),  # 대회전
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
def test_crossing_and_double_split_on_how_many_times_it_crossed():
    """그가 준 기준 (2026-09-20):

    *"횡단은 두 장쿠션 사이를 최소한 3회 이상 오간 이후에 득점하는 것. 더블쿠션은
    두 장쿠션 사이를 두 번 오간 이후 3쿠션 이후에는 단쿠션을 맞고 득점하는 것.
    아주 드물게는 두 단쿠션 사이를 오가면서도 가능은 하지만 아주 드물어."*

    그는 프로가 아니라 배우는 사람이고 본인도 확정하지 말라고 했다. 바깥 자료
    (japong.com)와는 맞는다 — "단쿠션에 나란하게 왕복". 라벨이 쌓이면
    rules_check.py가 점수를 매긴다.
    """
    got = js_names([
        [0, ["top", "bottom", "top"], True, "left", True],             # 3회
        [0, ["top", "bottom", "top", "bottom", "top"], True, "left", True],  # 5회
        [0, ["left", "right", "left"], True, "left", True],            # 단쿠션끼리
        [0, ["top", "bottom", "left"], True, "left", True],            # 더블
        [0, ["left", "right", "top"], True, "left", True],             # 더블
        [0, ["top", "left", "top"], True, "left", True],               # 되돌아오기
    ])
    assert got[:3] == ["횡단", "횡단", "횡단"], f"횡단을 못 알아봅니다: {got[:3]}"
    # 더블은 **최상위 이름이 아니다** — "더블은 빗겨치기의 하위구분이 맞아"
    # (2026-09-20). 이름은 계열 규칙이 붙이고, 더블은 꼬리표로 나란히 간다.
    assert "더블" not in got, "더블이 다시 이름 자리로 올라왔습니다"
    assert got[5] == "되돌아오기", "되돌아오기와 헷갈립니다"


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_double_is_a_tag_beside_the_name_not_the_name():
    """단-단-장이 빗겨치기로 나오는지 — taxonomy의 빗겨치기 행과 맞아야 한다.

    그 행은 "중단 빗겨치기(단-단-장 / 단-장-단-장)"이고 더블을 하위 구분으로
    올려 두었다. 이 배치가 계열 규칙만으로 빗겨치기가 되어야 둘이 맞는 것이고,
    실제로 그렇게 나온다 — 강제한 것이 아니다.
    """
    script = f"""
    const ROUTE = require({json.dumps(str(ROUTE_JS))});
    const ask = (between, face, circuitRight) => [
      ROUTE.name({{ before: 0, between, reachedSecond: true, face, circuitRight }}),
      ROUTE.subtypeOf(between),
    ];
    console.log(JSON.stringify([
      ask(["left", "right", "top"], "left", true),     // 단-단-장
      ask(["top", "bottom", "left"], "left", true),    // 장-장-단
      ask(["top", "left", "bottom"], "left", true),    // 오간 것이 아니다
    ]));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(done.stdout)
    assert got[0] == ["빗겨치기", "더블"], f"단-단-장이 {got[0]}로 나옵니다"
    assert got[1][1] == "더블", "장-장-단에도 더블 꼬리표가 붙어야 합니다"
    assert got[2][1] is None, "오가지 않은 것에 더블이 붙었습니다"
