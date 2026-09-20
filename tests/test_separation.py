"""분리각 — 선수가 준 정의와 세 가지 규칙을 지킨다.

정의: 수구가 제1적구를 맞은 직후 두 공이 갈라져 나가는 두 진행 방향 사이의 각.
적구는 중심을 잇는 선으로 떠나고, 수구는 그에 수직인 접선 쪽으로 튕긴다.

선수 (2026-09-20): *"분리각은 스트로크가 강할수록, 당점이 아래쪽으로 갈수록
커지고, vice versa야."* 그리고 거리에 대한 예측 — 같은 세기·같은 당점이라도
1적구가 멀면 작아진다 — 에는 *"대개 맞아"*.

이 셋은 서로 다른 세 가지가 아니라 **접촉 순간의 미끄럼** 하나다. 스턴(완전히
미끄러지는 중)이면 90도에 가깝고, 구르며 도착하면 앞으로 끌려가 작아진다.
하단 당점은 미끄럼을 늘리고, 강한 스트로크는 미끄럼이 죽기 전에 도착하게 하고,
먼 거리는 가는 동안 미끄럼을 죽인다.

처음 재봤을 때 시뮬레이터는 **셋 다 틀렸다** — 당점·세기·거리를 무엇으로 바꿔도
73.9도로 꿈쩍하지 않았다. `collide()`가 끝에서 `slip = velocity`로 덮어써서
수구의 구름을 지우고 있었고, 앞으로 나아가는 몫은 `CUE_CARRY`라는 상수가
흉내 내고 있었다. 두 공 사이의 충격은 중심을 잇는 선을 따라 두 중심을 지나므로
수평축 회전에 토크를 주지 않는다 — 구름은 충돌을 그대로 통과해야 한다.
"""

import numpy as np
import pytest

from src.physics import spin
from src.physics.strength import speed_for

STEP = 1 / 6000
# 분리각은 접촉 순간이 아니라 그 직후 몇 cm에서 자리잡는다. 남은 회전을 천이
# 끌어당기는 데 거리가 필요하기 때문이고, 선수가 눈으로 보는 것도 그 각이다.
AFTER_MM = 150.0


def separation(thickness=0.5, tips_vertical=0.0, strength=4.0, reach=400.0):
    """1적구까지 `reach` mm를 굴러가서 맞았을 때의 분리각(도)."""
    cue = spin.struck((0.0, 0.0), (1.0, 0.0), speed_for(strength),
                      tips_vertical=tips_vertical)
    gone = 0.0
    while gone < reach and cue.speed > 1.0:
        was = cue.position.copy()
        spin.roll_on(cue, STEP)
        gone += float(np.hypot(*(cue.position - was)))
    if cue.speed < 50:
        return None

    offset = spin.BALL_DIAMETER_MM * (1.0 - thickness)
    along = np.sqrt(max(0.0, spin.BALL_DIAMETER_MM ** 2 - offset ** 2))
    other = spin.Ball((cue.position[0] + along, cue.position[1] + offset))
    spin.collide(cue, other)

    from_cue, from_other = cue.position.copy(), other.position.copy()
    gone = 0.0
    while gone < AFTER_MM and cue.speed > 1.0:
        was = cue.position.copy()
        spin.roll_on(cue, STEP)
        spin.roll_on(other, STEP)
        gone += float(np.hypot(*(cue.position - was)))
    a, b = cue.position - from_cue, other.position - from_other
    if np.hypot(*a) < 1.0 or np.hypot(*b) < 1.0:
        return None
    cosine = float(np.dot(a, b) / (np.hypot(*a) * np.hypot(*b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def test_a_stun_hit_parts_near_a_right_angle():
    """적구는 중심선으로, 수구는 접선으로 — 스턴이면 직각에 가깝다.

    정확히 90도는 아니다: 반발계수가 1이 아니라 수구가 법선 방향 속도의
    (1-e)/2 = 2.8%를 남기고, 재는 150 mm 동안 이미 조금 굴러 앞으로 휜다.
    """
    for thickness in (0.6, 0.5, 0.4, 0.25):
        angle = separation(thickness=thickness, strength=7.0, reach=30.0)
        assert angle is not None
        assert 75.0 < angle < 92.0, f"두께 {thickness}에서 {angle:.0f}도"


def test_a_lower_tip_opens_the_separation():
    """당점이 아래로 갈수록 커진다."""
    low = separation(tips_vertical=-2.0)
    middle = separation(tips_vertical=0.0)
    high = separation(tips_vertical=3.0)
    assert low > middle > high, f"하단 {low:.1f} · 중단 {middle:.1f} · 상단 {high:.1f}"


def test_a_harder_stroke_opens_the_separation():
    """강하게 칠수록 커진다 — 미끄럼이 죽기 전에 도착하기 때문이다."""
    soft = separation(strength=1.5)
    firm = separation(strength=4.0)
    hard = separation(strength=7.0)
    assert soft < firm < hard, f"약 {soft:.1f} · 보통 {firm:.1f} · 강 {hard:.1f}"


def test_a_longer_run_to_the_ball_closes_it():
    """1적구가 멀수록 작아진다 — 가는 동안 미끄럼이 죽어 굴러서 도착한다."""
    near = separation(reach=30.0)
    far = separation(reach=1400.0)
    assert near > far + 3.0, f"가까이 {near:.1f} · 멀리 {far:.1f}"


def test_the_collision_does_not_wipe_the_cue_ball_roll():
    """구름이 충돌을 통과하는지 직접 본다 — 이것이 위 셋의 뿌리다."""
    cue = spin.struck((0.0, 0.0), (1.0, 0.0), speed_for(4.0), tips_vertical=3.0)
    for _ in range(600):
        spin.roll_on(cue, STEP)
    assert cue.rolling, "이 시험은 구르는 수구로 시작해야 합니다"
    rolling_before = cue.velocity - cue.slip

    other = spin.Ball((cue.position[0] + spin.BALL_DIAMETER_MM * 0.87,
                       cue.position[1] + spin.BALL_DIAMETER_MM * 0.5))
    spin.collide(cue, other)
    rolling_after = cue.velocity - cue.slip
    assert np.allclose(rolling_before, rolling_after, atol=1e-6), \
        "충돌이 수구의 구름을 바꿨습니다 — 중심선을 지나는 충격은 토크를 주지 않습니다"


@pytest.mark.parametrize("field", ["CUE_CARRY"])
def test_no_constant_stands_in_for_follow(field):
    """앞으로 나아가는 몫은 남은 구름이 낸다. 상수로 흉내 내면 스턴이 스턴이
    아니게 되고, 분리각이 당점에도 세기에도 꿈쩍하지 않는다."""
    assert getattr(spin, field) == 0.0, f"{field}가 다시 살아났습니다"


def test_the_cue_ball_deflects_toward_the_face_it_struck():
    """선수 (2026-09-21): *"반사각이 움직이는 쪽이 맞는 면 쪽이야."*

    ⚠️ 헷갈리기 쉬운 자리다. `struck_side`가 재는 `cross(진행방향, 적구까지)`는
    **적구가 어느 쪽에 있나**이고, **맞히는 면은 그 반대쪽**이다 — 적구가
    오른쪽에 있으면 수구는 그 왼쪽 면을 맞힌다. 그래서 "적구 쪽"과 "꺾이는 쪽"의
    부호가 반대로 나오는 것이 **정상**이고, 2026-09-21에 이것을 보고 "면 이름이
    뒤집혔다"고 잠깐 잘못 단정했다.

    여기서 지키는 것은 **맞힌 면과 꺾이는 쪽이 같다**는 것이다.
    """
    speed = speed_for(4.0)
    for where in (+1.0, -1.0):
        cue = spin.struck((0.0, 0.0), (1.0, 0.0), speed)
        offset = spin.BALL_DIAMETER_MM * 0.5 * where      # 적구 중심이 놓인 쪽
        along = np.sqrt(spin.BALL_DIAMETER_MM ** 2 - offset ** 2)
        spin.collide(cue, spin.Ball((along, offset)))

        # 수구가 지나가는 쪽 = 맞히는 면 = 적구가 있는 쪽의 반대.
        face = -where
        turned = np.sign(cue.velocity[1])
        assert turned == face, (
            f"적구가 {'+y' if where > 0 else '-y'}에 있으면 수구는 그 반대 면을 "
            f"맞고 그쪽으로 꺾여야 하는데 {turned:+.0f} 쪽으로 갔습니다")
