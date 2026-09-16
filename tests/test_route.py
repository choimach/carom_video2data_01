"""The route names, on sequences short enough to read."""

from src.physics.carom import Event
from src.physics.route import (
    BANK,
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


def test_a_short_rail_first_is_behind():
    shot = events((10, "ball", "red"), (20, "cushion", "right"),
                  (30, "cushion", "top"), (40, "ball", "yellow"))
    assert classify(shot)["route"] == BEHIND


def test_the_cue_ball_own_side_is_front_and_across_is_side():
    # The cue ball starts at y = 300, so "top" (y = 0) is its own side.
    own = events((10, "ball", "red"), (20, "cushion", "top"),
                 (30, "cushion", "left"), (40, "ball", "yellow"))
    across = events((10, "ball", "red"), (20, "cushion", "bottom"),
                    (30, "cushion", "left"), (40, "ball", "yellow"))
    assert classify(own, LAYOUT_NEAR_TOP, "white")["route"] == FRONT
    assert classify(across, LAYOUT_NEAR_TOP, "white")["route"] == SIDE


def test_that_split_needs_a_layout_and_says_so():
    shot = events((10, "ball", "red"), (20, "cushion", "top"),
                  (30, "cushion", "left"), (40, "ball", "yellow"))
    verdict = classify(shot)
    assert verdict["route"] == UNKNOWN
    assert verdict["basis"] == "geometry"


def test_counted_and_guessed_verdicts_are_marked_apart():
    hook = classify(events((10, "cushion", "top"), (20, "ball", "red"),
                           (30, "ball", "yellow")))
    turn = classify(events((10, "ball", "red"), (20, "cushion", "right"),
                           (30, "ball", "yellow")))
    assert hook["basis"] == "counted"
    assert turn["basis"] == "geometry"
