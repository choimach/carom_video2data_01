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


def test_a_play_with_another_behind_it_scored_even_if_the_inning_is_short():
    """A miss ends the turn, so a play that is followed by another one scored.

    This holds whether or not the inning adds up, and it is where the labelling
    gains most: the trajectory judge misses contacts and calls scoring plays
    misses, and no path can make a play a miss when the same player went on to
    play again.
    """
    inning = make_inning(3)  # three plays against a four-play turn
    for shot, verdict in zip(inning.shots, (True, False, True)):
        shot.success = verdict
        shot.verdict_source = "trajectory"

    label_from_inning_shape([inning], [(0, 400, "white", 3)])

    assert [s.success for s in inning.shots] == [True, True, True]
    assert [s.verdict_source for s in inning.shots] == ["inning", "inning", "trajectory"]


def test_the_last_play_of_a_short_inning_keeps_the_trajectory_verdict():
    """Where plays are missing the turn may have run on past what was seen, so
    the last play detected need not be the one that ended it."""
    inning = make_inning(2)  # two plays against a four-play turn
    for shot, verdict in zip(inning.shots, (False, True)):
        shot.success = verdict
        shot.verdict_source = "trajectory"

    label_from_inning_shape([inning], [(0, 400, "white", 3)])

    assert inning.shots[-1].success is True
    assert inning.shots[-1].verdict_source == "trajectory"
    assert inning.shots[-1].inning_success is None


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


def test_a_scoring_play_cut_short_after_the_score_is_still_usable():
    """What a scoring play is wanted for is the path up to the score.

    Rejecting every play the camera cut away from threw out 28 of the 56
    dropped in one match, each holding the cue ball's whole path through its
    cushions and onto the second object ball. The balls had not finished
    rolling, which is what `complete` records - their end positions are where
    the balls were when the camera left, not where they stopped.
    """
    from src.segmentation.shot_segmenter import play_rejections

    shot = make_inning(1).shots[0]
    shot.complete = False
    shot.success = True

    shot.verdict = {"reason": "only red was touched", "cushions": 4, "first_ball": "red"}
    assert "truncated" in play_rejections(shot, 60.0)

    shot.verdict = {"reason": "3 cushions before white", "cushions": 3,
                    "first_ball": "red", "second_ball": "white"}
    assert play_rejections(shot, 60.0) == []
