"""The simulator, checked against things that have an answer without measuring."""

import numpy as np
import pytest

from src.physics.simulator import (BALL_DIAMETER_MM, BALL_RADIUS_MM, CUE_CARRY,
                                   CUSHION_RESTITUTION, Table, collide,
                                   deceleration, simulate)
from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM


def test_a_ball_left_alone_rolls_straight_and_stops():
    layout = {"white": (400.0, 711.0), "yellow": (2700.0, 100.0), "red": (2700.0, 1300.0)}
    shot = simulate(layout, "white", (600.0, 0.0))
    path = shot.paths["white"]
    assert np.allclose(path[:, 1], 711.0, atol=1.0)     # nothing pushes it sideways
    assert path[-1, 0] > path[0, 0]
    assert shot.settled


def test_a_faster_ball_travels_further():
    """Distance covered, not distance from home: the faster ball reaches a
    cushion and comes back, so where it ends up says nothing."""
    layout = {"white": (200.0, 711.0), "yellow": (2700.0, 100.0), "red": (2700.0, 1300.0)}

    def covered(speed):
        path = simulate(layout, "white", (speed, 0.0)).paths["white"]
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    assert covered(1500.0) > covered(500.0)


def test_deceleration_rises_with_speed():
    """A ball slides before it rolls, and sheds speed far faster while it does."""
    assert deceleration(300.0) < deceleration(1200.0) < deceleration(2500.0)


def test_a_ball_stays_on_the_table():
    layout = {"white": (1422.0, 711.0), "yellow": (200.0, 200.0), "red": (2600.0, 1200.0)}
    shot = simulate(layout, "white", (3000.0, 1900.0))
    for path in shot.paths.values():
        assert path[:, 0].min() >= BALL_RADIUS_MM - 1
        assert path[:, 0].max() <= TABLE_LENGTH_MM - BALL_RADIUS_MM + 1
        assert path[:, 1].min() >= BALL_RADIUS_MM - 1
        assert path[:, 1].max() <= TABLE_WIDTH_MM - BALL_RADIUS_MM + 1


def test_a_ball_leaves_a_cushion_wider_than_it_arrived():
    """Measured on 1297 bounces: a ball arriving 18 degrees off the normal
    leaves at 34. It is the ball's own roll driving it along the rail."""
    table = Table()
    position = np.array([BALL_RADIUS_MM - 5.0, 700.0])
    incoming = np.array([-1000.0, 1000.0 * np.tan(np.radians(18.0))])
    _moved, velocity, rail = table.bounce(position, incoming)
    assert rail == "left"
    assert velocity[0] > 0                      # sent back onto the table
    assert velocity[1] > 0                      # still going the same way along it
    out = np.degrees(np.arctan2(abs(velocity[1]), abs(velocity[0])))
    assert out == pytest.approx(34.0, abs=2.0)
    assert np.hypot(*velocity) < np.hypot(*incoming)


def test_a_ball_arriving_square_comes_straight_back():
    """It has no direction to open into, so symmetry settles it."""
    table = Table()
    _moved, velocity, rail = table.bounce(np.array([BALL_RADIUS_MM - 5.0, 700.0]),
                                          np.array([-1000.0, 0.0]))
    assert rail == "left"
    assert velocity[1] == pytest.approx(0.0, abs=1e-9)
    assert velocity[0] > 0


def test_a_struck_ball_leaves_along_the_line_of_centres():
    """Which is what makes the thickness measurement work in the other direction."""
    a = np.array([0.0, 0.0])
    b = np.array([BALL_DIAMETER_MM * np.cos(np.pi / 6), BALL_DIAMETER_MM * np.sin(np.pi / 6)])
    _after_a, after_b = collide(a, np.array([1000.0, 0.0]), b, np.zeros(2))
    centres = (b - a) / np.linalg.norm(b - a)
    assert np.allclose(after_b / np.linalg.norm(after_b), centres, atol=1e-6)


def test_the_striking_ball_carries_forward_rather_than_leaving_square():
    """A frictionless cue ball leaves a cut square to the centres. A real one
    arrives rolling and the roll does not stop because the ball in front of it
    did - measured at 0.089 of its incoming speed over 297 collisions."""
    a = np.array([0.0, 0.0])
    b = np.array([BALL_DIAMETER_MM * np.cos(np.pi / 6), BALL_DIAMETER_MM * np.sin(np.pi / 6)])
    incoming = np.array([1000.0, 0.0])
    centres = (b - a) / np.linalg.norm(b - a)
    with_roll, _ = collide(a, incoming, b, np.zeros(2))
    without, _ = collide(a, incoming, b, np.zeros(2), carry=0.0)

    # Without the roll the striker keeps only what an imperfect contact left it.
    carried = float(np.dot(with_roll, centres)) - float(np.dot(without, centres))
    assert carried == pytest.approx(CUE_CARRY * abs(float(np.dot(incoming, centres))), rel=1e-6)
    assert float(np.dot(with_roll, centres)) > float(np.dot(without, centres)) > 0


def test_three_cushions_before_the_second_ball_is_a_point():
    class Fake:
        def __init__(self, events):
            self.events = events
        scored = simulate.__globals__["Shot"].scored
        cushions = simulate.__globals__["Shot"].cushions

    made = Fake([(1, "ball", "red"), (2, "cushion", "top"), (3, "cushion", "left"),
                 (4, "cushion", "bottom"), (5, "ball", "yellow")])
    assert made.scored("white") is True

    short = Fake([(1, "ball", "red"), (2, "cushion", "top"), (3, "ball", "yellow")])
    assert short.scored("white") is False

    # Four, five or six cushions score the same; the rule is three before the
    # second ball, not exactly three.
    long_way = Fake([(1, "ball", "red")] + [(i, "cushion", "top") for i in range(2, 8)]
                    + [(9, "ball", "yellow")])
    assert long_way.scored("white") is True


def test_aiming_finds_the_line_that_scores():
    """A layout arranged so one direction makes a carom and most do not."""
    from src.physics.aiming import probability, sweep

    layout = {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)}
    angles, scored = sweep(layout, "white", speeds=(2600.0,), angle_step_deg=2.0)
    outcomes = scored[2600.0]
    assert len(angles) == 180
    # Most directions do not score - which is the point of looking for the ones
    # that do. A map where everything scores is measuring nothing.
    assert 0 < outcomes.sum() < len(outcomes) * 0.2


def test_scoring_room_is_worth_more_than_a_scoring_line():
    """A line with scoring room either side beats an isolated one, because a
    stroke lands on a spread and not on a number."""
    from src.physics.aiming import probability

    isolated = np.zeros(360, dtype=bool)
    isolated[100] = True
    broad = np.zeros(360, dtype=bool)
    broad[200:208] = True

    chances = probability(np.logical_or(isolated, broad), angle_step_deg=1.0, spread_deg=1.5)
    assert chances[204] > chances[100]
    assert chances[100] > 0            # an isolated line is still worth something


def test_the_map_wraps_all_the_way_round():
    """359.9 degrees is next to 0.1, and a scoring line at the seam has room on
    both sides of it like any other."""
    from src.physics.aiming import probability

    outcomes = np.zeros(360, dtype=bool)
    outcomes[0] = outcomes[359] = True
    chances = probability(outcomes, angle_step_deg=1.0, spread_deg=1.0)
    assert chances[0] == pytest.approx(chances[359], rel=0.05)
    assert chances[358] > 0 and chances[1] > 0
