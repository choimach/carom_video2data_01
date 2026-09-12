"""Scoring judged from the trajectory, tested on paths built to order.

Each case states one rule, so a failure names the rule that broke rather than
"a shot was misjudged".
"""

import numpy as np
import pytest

from src.physics.carom import (
    BALL_RADIUS_MM,
    Event,
    ball_events,
    cushion_events,
    judge_carom,
    shot_events,
)
from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM


def path(points, steps=40):
    """A cue-ball path through the given table positions."""
    out = []
    for a, b in zip(points[:-1], points[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(points[-1])
    return np.array(out, dtype=float)


def parked(xy, n):
    return np.repeat(np.array([xy], dtype=float), n, axis=0)


def test_a_rail_is_one_contact_however_long_the_ball_runs_along_it():
    along = path([(200.0, BALL_RADIUS_MM), (2600.0, BALL_RADIUS_MM)], steps=200)
    events = cushion_events(along)
    assert [e.detail for e in events] == ["top"]


def test_a_corner_counts_as_two_cushions():
    corner = path([(1400.0, 700.0), (BALL_RADIUS_MM, BALL_RADIUS_MM), (1400.0, 700.0)])
    rails = sorted(e.detail for e in cushion_events(corner))
    assert rails == ["left", "top"], "both rails of the corner count"


def test_leaving_a_rail_and_returning_counts_twice():
    there_and_back = path(
        [(500.0, BALL_RADIUS_MM), (500.0, 700.0), (900.0, BALL_RADIUS_MM)], steps=60
    )
    assert len([e for e in cushion_events(there_and_back) if e.detail == "top"]) == 2


def test_three_cushions_then_the_second_ball_scores():
    cue = path([
        (1400.0, 700.0),
        (2400.0, 300.0),          # first object ball sits here
        (TABLE_LENGTH_MM - BALL_RADIUS_MM, 500.0),
        (1800.0, BALL_RADIUS_MM),
        (BALL_RADIUS_MM, 400.0),
        (500.0, 150.0),           # second object ball
    ])
    others = {"red": parked((2400.0, 300.0), len(cue)), "yellow": parked((500.0, 150.0), len(cue))}
    scored, details = judge_carom(shot_events(cue, others))
    assert scored is True
    assert details["cushions"] >= 3
    assert details["first_ball"] == "red" and details["second_ball"] == "yellow"


def test_more_than_three_cushions_still_scores():
    """Four, five, seven - the rule is a minimum, and a grand tour is a point
    like any other."""
    cue = path([
        (1400.0, 700.0),
        (2400.0, 300.0),
        (TABLE_LENGTH_MM - BALL_RADIUS_MM, 500.0),
        (1800.0, BALL_RADIUS_MM),
        (BALL_RADIUS_MM, 400.0),
        (900.0, TABLE_WIDTH_MM - BALL_RADIUS_MM),
        (1500.0, 1000.0),
    ])
    others = {"red": parked((2400.0, 300.0), len(cue)), "yellow": parked((1500.0, 1000.0), len(cue))}
    scored, details = judge_carom(shot_events(cue, others))
    assert scored is True
    assert details["cushions"] >= 4


def test_two_cushions_does_not_score():
    """The feedback on p065 was exactly this: "2쿠션에 맞음"."""
    cue = path([
        (1400.0, 700.0),
        (2400.0, 300.0),
        (TABLE_LENGTH_MM - BALL_RADIUS_MM, 400.0),
        (1500.0, BALL_RADIUS_MM),
        (500.0, 150.0),
    ])
    others = {"red": parked((2400.0, 300.0), len(cue)), "yellow": parked((500.0, 150.0), len(cue))}
    scored, details = judge_carom(shot_events(cue, others))
    assert scored is False
    assert details["cushions"] == 2
    assert "required" in details["reason"]


def test_touching_the_first_ball_again_is_not_a_second_object_ball():
    cue = path([
        (1400.0, 700.0),
        (2400.0, 300.0),
        (TABLE_LENGTH_MM - BALL_RADIUS_MM, 400.0),
        (1500.0, BALL_RADIUS_MM),
        (BALL_RADIUS_MM, 300.0),
        (2400.0, 300.0),          # back onto the first ball
    ])
    others = {"red": parked((2400.0, 300.0), len(cue)), "yellow": parked((600.0, 1200.0), len(cue))}
    scored, details = judge_carom(shot_events(cue, others))
    assert scored is False
    assert "second object ball was missed" in details["reason"]


def test_hitting_nothing_does_not_score():
    cue = path([(1400.0, 700.0), (1400.0, 500.0)])
    others = {"red": parked((200.0, 200.0), len(cue)), "yellow": parked((2600.0, 1200.0), len(cue))}
    scored, details = judge_carom(shot_events(cue, others))
    assert scored is False
    assert details["cushions"] == 0


def test_cushions_taken_after_the_point_do_not_count_towards_it():
    """Only the path up to the second object ball decides the point."""
    events = [
        Event(10, "ball", "red"),
        Event(20, "cushion", "left"),
        Event(30, "cushion", "top"),
        Event(40, "ball", "yellow"),
        Event(50, "cushion", "right"),
        Event(60, "cushion", "bottom"),
    ]
    scored, details = judge_carom(events)
    assert scored is False
    assert details["cushions"] == 2, "the two rails after the carom are irrelevant"


def test_ball_contact_is_measured_centre_to_centre():
    cue = path([(1000.0, 700.0), (1200.0, 700.0)], steps=200)
    just_touching = parked((1200.0 + 61.5, 700.0), len(cue))
    assert [e.detail for e in ball_events(cue, {"red": just_touching})] == ["red"]
    clear = parked((1200.0 + 200.0, 700.0), len(cue))
    assert ball_events(cue, {"red": clear}) == []


def test_untracked_frames_do_not_invent_contacts():
    cue = path([(1000.0, 700.0), (1200.0, 700.0)], steps=100)
    cue[40:60] = np.nan
    others = {"red": parked((1200.0 + 61.5, 700.0), len(cue))}
    events = shot_events(cue, others)
    assert all(np.isfinite(cue[e.frame]).all() for e in events if e.kind == "ball")
