"""Stroke geometry, tested against contacts whose answer is known by construction."""

import numpy as np
import pytest

from src.physics.stroke import (BALL_DIAMETER_MM, follow_draw, refine_contact,
                                sidespin, thickness)

FPS_STEP = 20.0  # mm per frame; a firm stroke


def contact(offset_ratio, frames=14, cue_speed=FPS_STEP):
    """Two tracks for a cue ball meeting a resting object ball.

    The cue ball runs along +x. The object ball sits `offset_ratio` of a ball
    width off that line, so the thickness is 1 - offset_ratio by construction,
    and leaves along the line joining the centres at the speed an elastic
    contact between equal masses gives it.
    """
    sine = offset_ratio
    cosine = float(np.sqrt(max(0.0, 1.0 - sine ** 2)))
    meet = frames // 2
    object_at = np.array([meet * cue_speed + BALL_DIAMETER_MM * cosine,
                          BALL_DIAMETER_MM * sine])

    cue = np.full((frames, 2), np.nan)
    obj = np.full((frames, 2), np.nan)
    centres = np.array([cosine, sine])          # object ball leaves along this
    tangent = np.array([-sine, cosine])         # cue ball leaves along this
    for f in range(frames):
        if f <= meet:
            cue[f] = [f * cue_speed, 0.0]
            obj[f] = object_at
        else:
            step = f - meet
            cue[f] = cue[meet] + tangent * cue_speed * sine * step
            obj[f] = object_at + centres * cue_speed * cosine * step
    return cue, obj, meet


@pytest.mark.parametrize("offset,expected", [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.8, 0.2)])
def test_thickness_matches_the_geometry_it_was_built_from(offset, expected):
    cue, obj, meet = contact(offset)
    assert thickness(cue, obj, meet, gap=1, span=4) == pytest.approx(expected, abs=0.03)


def test_a_stun_shot_reads_as_neither_follow_nor_draw():
    """The cue ball leaving square to the line of centres is what no spin looks
    like, whatever the thickness."""
    cue, obj, meet = contact(0.5)
    assert follow_draw(cue, obj, meet, gap=1, span=4) == pytest.approx(0.0, abs=0.05)


def test_follow_reads_positive_and_draw_negative():
    cue, obj, meet = contact(0.5)
    centres = np.array([np.sqrt(0.75), 0.5])
    for sign, name in ((1.0, "follow"), (-1.0, "draw")):
        bent = cue.copy()
        for f in range(meet + 1, len(cue)):
            bent[f] = cue[f] + centres * sign * FPS_STEP * 0.4 * (f - meet)
        value = follow_draw(bent, obj, meet, gap=1, span=4)
        assert value is not None, name
        assert (value > 0.2) if sign > 0 else (value < -0.2), f"{name} read {value}"


def test_a_mirror_rebound_reads_as_no_side():
    """Equal angles in and out is what a ball with nothing on it does."""
    frames = 14
    cue = np.zeros((frames, 2))
    for f in range(frames):
        if f <= 6:
            cue[f] = [100.0 + f * 20.0, 400.0 - f * 20.0]   # towards the top rail
        else:
            cue[f] = cue[6] + [(f - 6) * 20.0, (f - 6) * 20.0]
    assert sidespin(cue, 6, "top", gap=1, span=4) == pytest.approx(0.0, abs=2.0)


def test_a_wider_rebound_reads_as_running_side():
    frames = 14
    cue = np.zeros((frames, 2))
    for f in range(frames):
        if f <= 6:
            cue[f] = [100.0 + f * 20.0, 400.0 - f * 20.0]
        else:
            cue[f] = cue[6] + [(f - 6) * 34.0, (f - 6) * 12.0]  # flatter, so wider
    assert sidespin(cue, 6, "top", gap=1, span=4) > 10.0


def test_contact_is_refined_to_the_closest_approach():
    cue = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [150.0, 0.0], [200.0, 0.0]])
    obj = np.full((5, 2), 160.0)
    obj[:, 1] = 0.0
    assert refine_contact(cue, obj, 1, search=2) == 3
