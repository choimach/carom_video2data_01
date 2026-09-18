"""A ball that carries spin, checked against what the table does."""

import numpy as np

from src.physics import spin


def roll_for(ball, seconds, step=1 / 240):
    for _ in range(int(seconds / step)):
        spin.roll_on(ball, step)


def test_a_struck_ball_slides_before_it_rolls():
    ball = spin.struck((700.0, 700.0), (1.0, 0.0), 2700.0)
    assert not ball.rolling
    roll_for(ball, 0.6)
    assert ball.rolling


def test_follow_leaves_it_already_rolling_and_draw_spinning_backwards():
    follow = spin.struck((700.0, 700.0), (1.0, 0.0), 2000.0, tips_vertical=2.5)
    draw = spin.struck((700.0, 700.0), (1.0, 0.0), 2000.0, tips_vertical=-2.5)
    assert float(np.hypot(*follow.slip)) < float(np.hypot(*draw.slip))


def test_side_is_read_back_in_the_tips_it_was_given():
    ball = spin.struck((700.0, 700.0), (1.0, 0.0), 2500.0, tips_side=2.0)
    assert spin.tips_of(ball) == 2.0


def rebound(tips, heading):
    ball = spin.struck((1400.0, 200.0), heading, 2500.0, tips_side=tips)
    ball.position = np.array([1400.0, spin.BALL_RADIUS_MM])
    spin.bounce(ball, "top")
    return np.degrees(np.arctan2(abs(ball.velocity[0]), abs(ball.velocity[1])))


def test_one_side_opens_a_rebound_and_the_other_closes_it():
    # Which of the two is "running" depends on the rail and on which way along
    # it the ball is going - so what is fixed is that they pull opposite ways.
    assert rebound(-3.0, (1.0, -1.0)) > rebound(0.0, (1.0, -1.0)) >= rebound(3.0, (1.0, -1.0))


def test_and_they_swap_when_the_ball_runs_the_other_way():
    # Same side on the ball, same rail, opposite direction along it: the side
    # that opened the angle now closes it. A model that gets this wrong sends
    # every reverse-side shot the wrong way round the table.
    assert rebound(3.0, (-1.0, -1.0)) > rebound(0.0, (-1.0, -1.0)) >= rebound(-3.0, (-1.0, -1.0))


def test_a_cushion_keeps_some_of_the_side_and_not_all():
    ball = spin.struck((1400.0, 200.0), (0.2, -1.0), 2500.0, tips_side=3.0)
    before = ball.side
    ball.position = np.array([1400.0, spin.BALL_RADIUS_MM])
    spin.bounce(ball, "top")
    assert 0 < ball.side < before


def test_a_cut_throws_the_object_ball_off_the_line_of_centres():
    striker = spin.struck((1000.0, 700.0), (1.0, 0.0), 2200.0, tips_side=3.0)
    struck_ball = spin.Ball((1000.0 + spin.BALL_DIAMETER_MM, 700.0))
    spin.collide(striker, struck_ball)
    # Straight through the centres would leave it travelling due x; the side on
    # the cue ball rubs it off that line.
    assert abs(struck_ball.velocity[1]) > 1.0
