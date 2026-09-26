"""The route names, on sequences short enough to read."""

from src.physics.carom import Event
from src.physics.route import (
    BANK,
    GLANCING,
    RETURNING,
    BEHIND,
    CROSSING,
    FRONT,
    HOOK,
    LONG_AROUND,
    SIDE,
    UNKNOWN,
    classify,
)

LAYOUT_NEAR_TOP = {"white": (800.0, 300.0), "yellow": (1400.0, 700.0),
                   "red": (2000.0, 1100.0)}


def events(*items):
    return [Event(frame, kind, detail) for frame, kind, detail in items]


def test_a_cushion_before_the_object_ball_is_a_hook():
    shot = events((10, "cushion", "top"), (20, "ball", "red"),
                  (30, "cushion", "left"), (40, "cushion", "bottom"),
                  (50, "ball", "yellow"))
    assert classify(shot)["route"] == HOOK


def test_two_cushions_first_is_a_bank():
    shot = events((10, "cushion", "top"), (15, "cushion", "left"),
                  (20, "ball", "red"), (50, "ball", "yellow"))
    assert classify(shot)["route"] == BANK


def test_five_cushions_is_the_long_way_round():
    """쿠션 다섯이면 대회전 **꼬리표**가 붙는다 — 이름은 계열이 갖는다."""
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"),
                  (70, "ball", "yellow"))
    assert LONG_AROUND in (classify(shot).get("tags") or [])


def test_long_rail_to_long_rail_is_crossing():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "bottom"), (40, "cushion", "top"),
                  (50, "ball", "yellow"))
    assert classify(shot)["route"] == CROSSING


def test_off_a_short_rail_the_same_test_names_the_near_pair():
    shot = events((10, "ball", "red"), (20, "cushion", "right"),
                  (30, "cushion", "top"), (40, "ball", "yellow"))
    right_face = -40.0
    assert classify(shot, struck_side=right_face, circuit=1.0)["route"] == FRONT
    assert classify(shot, struck_side=right_face, circuit=-1.0)["route"] == GLANCING


def test_without_a_circuit_the_turn_stays_unnamed():
    shot = events((10, "ball", "red"), (20, "cushion", "right"),
                  (30, "cushion", "top"), (40, "ball", "yellow"))
    assert classify(shot, struck_side=-40.0)["route"] == UNKNOWN


def test_a_long_rail_first_splits_on_face_against_circuit():
    # The player's own test: right face and round to the right is 옆돌리기;
    # right face going round left is 뒤돌리기.
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "ball", "yellow"))
    right_face, left_face = -40.0, 40.0
    assert classify(shot, struck_side=right_face, circuit=1.0)["route"] == SIDE
    assert classify(shot, struck_side=right_face, circuit=-1.0)["route"] == BEHIND
    assert classify(shot, struck_side=left_face, circuit=-1.0)["route"] == SIDE
    assert classify(shot, struck_side=left_face, circuit=1.0)["route"] == BEHIND


def test_the_same_rail_twice_around_a_short_one_is_a_return():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "cushion", "top"),
                  (50, "ball", "yellow"))
    assert classify(shot, struck_side=-40.0, circuit=1.0)["route"] == RETURNING





def test_that_split_needs_a_circuit_and_says_so():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "ball", "yellow"))
    verdict = classify(shot)
    assert verdict["route"] == UNKNOWN
    assert verdict["basis"] == "geometry"


def test_counted_and_guessed_verdicts_are_marked_apart():
    hook = classify(events((10, "cushion", "top"), (20, "ball", "red"),
                           (30, "ball", "yellow")))
    turn = classify(events((10, "ball", "red"), (20, "cushion", "right"),
                           (30, "ball", "yellow")), struck_side=-40.0, circuit=1.0)
    assert hook["basis"] == "counted"
    assert turn["basis"] == "geometry"


def test_a_shot_that_never_reached_a_second_ball_is_not_a_grand_tour():
    # Five cushions, but the cue ball is still running: the count is of what it
    # did after missing, not of cushions taken on the way to anything.
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"))
    verdict = classify(shot)
    assert LONG_AROUND not in (verdict.get("tags") or [])
    assert verdict["reached_second"] is False


def test_a_grand_tour_keeps_the_family_it_came_from():
    """대회전은 이름이 아니라 꼬리표다 — 선수가 확정했다 (2026-09-26).

    *"그래서 대회전에는 유형이 같이붙어 — 옆돌리기 대회전, 뒤돌리기 대회전 등."*

    옛 계약은 `route == "대회전"`이었고, 그래서 바탕 계열이 사라졌다. 그러면
    "뒤돌리기라면 1쿠션은 장쿠션" 같은 조건도 같이 사라진다 — 선수가 화면에서
    짚은 것이 그것이다 (씨앗 596883888).
    """
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"),
                  (70, "ball", "yellow"))
    verdict = classify(shot, struck_side=40.0, circuit=1.0)
    assert verdict["route"] != LONG_AROUND, "대회전이 계열을 가로채면 안 된다"
    assert verdict["route"] not in ("미분류", None)
    assert LONG_AROUND in (verdict.get("tags") or []), verdict


def test_a_grand_tour_without_geometry_cannot_be_named():
    """계열을 모르면 이름도 없다 — 쿠션 수만으로 이름을 짓지 않는다."""
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"),
                  (70, "ball", "yellow"))
    verdict = classify(shot)
    assert verdict["route"] != LONG_AROUND
    assert LONG_AROUND in (verdict.get("tags") or []), verdict
