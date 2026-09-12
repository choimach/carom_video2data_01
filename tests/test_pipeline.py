"""Where a play's verdict comes from."""

import pytest

from src.pipeline import label_from_inning_shape
from src.segmentation.shot_segmenter import Inning, Shot


def make_inning(count, cue="white", inferred=()):
    shots = [
        Shot(start_frame=i * 100, end_frame=i * 100 + 60, cue_ball=cue,
             start_positions={"white": (0, 0), "yellow": (1, 1), "red": (2, 2)},
             end_positions={"white": (3, 3), "yellow": (4, 4), "red": (5, 5)},
             complete=True, inferred=i in inferred)
        for i in range(count)
    ]
    return Inning(number=1, cue_ball=cue, shots=shots)


def test_a_complete_inning_decides_its_own_labels():
    """A turn of N points holds N scoring plays and then the miss that ends it.

    Once the play count matches the score there is nothing left to judge, and
    the rules are a better witness than watching the cue ball: against the clip
    labels collected by hand, the inning shape got 50 of 53 where the trajectory
    got 45.
    """
    inning = make_inning(4)
    for shot in inning.shots:  # what the trajectory judge thought, and got wrong
        shot.success = False

    label_from_inning_shape([inning], [(0, 400, "white", 3)])

    assert [s.success for s in inning.shots] == [True, True, True, False]
    assert {s.verdict_source for s in inning.shots} == {"inning"}


def test_an_inning_that_does_not_add_up_keeps_the_trajectory_verdict():
    """Where a play was missed or split in two the count will not match, and
    then the cue ball's path is all there is to go on."""
    inning = make_inning(3)  # three plays against a four-play turn
    for shot, verdict in zip(inning.shots, (True, False, True)):
        shot.success = verdict
        shot.verdict_source = "trajectory"

    label_from_inning_shape([inning], [(0, 400, "white", 3)])

    assert [s.success for s in inning.shots] == [True, False, True]
    assert {s.verdict_source for s in inning.shots} == {"trajectory"}


def test_a_play_never_shown_stays_unusable_even_though_the_shape_knows_it():
    """The shape says what an unseen play was; the video still never showed it,
    so it has no trajectory and cannot be used as data."""
    inning = make_inning(3, inferred=(1,))
    for shot in inning.shots:
        shot.success = None if shot.inferred else True

    label_from_inning_shape([inning], [(0, 300, "white", 2)])

    assert inning.shots[1].success is None
    assert inning.shots[1].inning_success is True
    assert [s.success for s in inning.shots] == [True, None, False]
