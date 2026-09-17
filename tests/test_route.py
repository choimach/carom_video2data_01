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
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"),
                  (70, "ball", "yellow"))
    assert classify(shot)["route"] == LONG_AROUND


def test_long_rail_to_long_rail_is_crossing():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "bottom"), (40, "cushion", "top"),
                  (50, "ball", "yellow"))
    assert classify(shot)["route"] == CROSSING


def test_off_a_short_rail_the_side_on_the_ball_decides():
    # The player's family rule: side on the same hand as the face struck is
    # 앞돌리기, side against it is 빗겨치기.
    shot = events((10, "ball", "red"), (20, "cushion", "right"),
                  (30, "cushion", "top"), (40, "ball", "yellow"))
    struck_right, right_english = -40.0, 12.0
    assert classify(shot, struck_side=struck_right, english=right_english)["route"] == FRONT
    assert classify(shot, struck_side=struck_right, english=-12.0)["route"] == GLANCING


def test_without_a_side_reading_that_pair_stays_unnamed():
    shot = events((10, "ball", "red"), (20, "cushion", "right"),
                  (30, "cushion", "top"), (40, "ball", "yellow"))
    assert classify(shot, struck_side=-40.0)["route"] == UNKNOWN


def test_a_long_rail_first_splits_on_where_the_second_cushion_went():
    # The player's own test: away from where he stands, or back toward him.
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "ball", "yellow"))
    assert classify(shot, away_mm=844.0)["route"] == BEHIND
    assert classify(shot, away_mm=-1063.0)["route"] == SIDE


def test_the_same_rail_twice_around_a_short_one_is_a_return():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "cushion", "top"),
                  (50, "ball", "yellow"))
    assert classify(shot, away_mm=400.0)["route"] == RETURNING





def test_that_split_needs_a_turn_angle_and_says_so():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "ball", "yellow"))
    verdict = classify(shot)
    assert verdict["route"] == UNKNOWN
    assert verdict["basis"] == "geometry"


def test_counted_and_guessed_verdicts_are_marked_apart():
    hook = classify(events((10, "cushion", "top"), (20, "ball", "red"),
                           (30, "ball", "yellow")))
    turn = classify(events((10, "ball", "red"), (20, "cushion", "right"),
                           (30, "ball", "yellow")), struck_side=-40.0, english=12.0)
    assert hook["basis"] == "counted"
    assert turn["basis"] == "geometry"


def test_a_shot_that_never_reached_a_second_ball_is_not_a_grand_tour():
    # Five cushions, but the cue ball is still running: the count is of what it
    # did after missing, not of cushions taken on the way to anything.
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"))
    verdict = classify(shot)
    assert verdict["route"] != LONG_AROUND
    assert verdict["reached_second"] is False


def test_the_count_still_names_a_grand_tour_when_the_second_ball_was_reached():
    shot = events((10, "ball", "red"), (20, "cushion", "left"),
                  (30, "cushion", "top"), (40, "cushion", "right"),
                  (50, "cushion", "bottom"), (60, "cushion", "left"),
                  (70, "ball", "yellow"))
    assert classify(shot)["route"] == LONG_AROUND
