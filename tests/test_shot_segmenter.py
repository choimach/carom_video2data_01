"""Shot and inning segmentation tested on synthetic trajectories.

Synthetic tracks let each case state exactly one thing - a shot, a player
change, a camera cut, a tracking dropout - which is not something a real
broadcast clip can be asked to do.
"""

import numpy as np
import pytest

from src.segmentation.shot_segmenter import (
    BALL_COLOURS,
    Shot,
    _rest_and_motion,
    group_innings,
    reconcile_with_scoreboard,
    segment_shots,
    speeds,
)

FPS = 60.0
REST_SECONDS = 1.5


def build_track(shots, rest_seconds=REST_SECONDS, jitter_mm=1.2, seed=0):
    """Render a sequence of (cue_colour, speed_ms, roll_seconds) into tracks.

    Balls sit still between shots, with a millimetre of jitter standing in for
    the centroid noise of a real detector.
    """
    rng = np.random.default_rng(seed)
    home = {"white": np.array([700.0, 400.0]), "yellow": np.array([1900.0, 1000.0]),
            "red": np.array([2400.0, 500.0])}
    tracks = {c: [] for c in BALL_COLOURS}

    def hold(frames):
        for _ in range(frames):
            for c in BALL_COLOURS:
                tracks[c].append(home[c].copy())

    hold(int(rest_seconds * FPS))
    for cue, speed_ms, roll_seconds in shots:
        frames = int(roll_seconds * FPS)
        step = np.array([speed_ms * 1000.0 / FPS, 0.0])
        for _ in range(frames):
            home[cue] = home[cue] + step
            if home[cue][0] > 2700:  # bounce off the far cushion
                step = -step
            for c in BALL_COLOURS:
                tracks[c].append(home[c].copy())
        hold(int(rest_seconds * FPS))

    out = {}
    for c in BALL_COLOURS:
        arr = np.array(tracks[c], dtype=float)
        arr += rng.normal(0.0, jitter_mm, arr.shape)
        out[c] = arr
    live = np.ones(len(out["white"]), dtype=bool)
    return out, live


def test_single_shot_is_found_with_the_right_cue_ball():
    positions, live = build_track([("white", 1.5, 2.0)])
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 1
    assert shots[0].cue_ball == "white"
    assert shots[0].complete


def test_jitter_alone_does_not_create_a_shot():
    positions, live = build_track([])
    assert segment_shots(positions, live, FPS) == []


def test_consecutive_shots_by_one_player_are_one_inning():
    positions, live = build_track([("white", 1.5, 1.5), ("white", 1.2, 1.5), ("white", 1.8, 1.5)])
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 3
    innings = group_innings(shots)
    assert len(innings) == 1, "a run of scoring shots is a single inning"
    assert innings[0].cue_ball == "white"
    assert len(innings[0].shots) == 3


def test_cue_ball_changing_colour_starts_a_new_inning():
    positions, live = build_track([("white", 1.5, 1.5), ("yellow", 1.5, 1.5), ("white", 1.5, 1.5)])
    shots = segment_shots(positions, live, FPS)
    assert [s.cue_ball for s in shots] == ["white", "yellow", "white"]
    innings = group_innings(shots)
    assert [i.cue_ball for i in innings] == ["white", "yellow", "white"]
    assert [i.number for i in innings] == [1, 2, 3]


def test_red_is_never_treated_as_a_cue_ball():
    """Red moving first is a tracking error, not a shot: nobody plays the red."""
    positions, live = build_track([("red", 1.5, 1.5)])
    assert segment_shots(positions, live, FPS) == []


def test_shot_records_the_layout_it_was_played_from():
    positions, live = build_track([("white", 1.5, 1.5)])
    shot = segment_shots(positions, live, FPS)[0]
    assert set(shot.start_positions) == set(BALL_COLOURS)
    assert np.allclose(shot.start_positions["yellow"], (1900.0, 1000.0), atol=8.0)
    assert np.allclose(shot.start_positions["red"], (2400.0, 500.0), atol=8.0)


def test_camera_cut_before_the_balls_stop_marks_the_shot_truncated():
    positions, live = build_track([("white", 1.5, 2.0)])
    cut_from = int((REST_SECONDS + 0.5) * FPS)
    live[cut_from:] = False
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 1
    assert not shots[0].complete


def test_movement_seen_only_after_a_cut_does_not_start_a_shot():
    """Cutting to the table mid-roll shows moving balls with no onset to read."""
    positions, live = build_track([("white", 1.5, 2.0)])
    live[: int((REST_SECONDS + 0.4) * FPS)] = False
    assert segment_shots(positions, live, FPS) == []


def test_a_dropped_ball_does_not_break_segmentation():
    positions, live = build_track([("white", 1.5, 1.5), ("yellow", 1.5, 1.5)])
    # The cue hides the red ball for half a second before the second shot.
    blind = slice(int(4.0 * FPS), int(4.5 * FPS))
    positions["red"][blind] = np.nan
    shots = segment_shots(positions, live, FPS)
    assert [s.cue_ball for s in shots] == ["white", "yellow"]


def test_speeds_ignore_frames_with_no_detection():
    positions, live = build_track([("white", 2.0, 1.0)])
    positions["white"][100:110] = np.nan
    v = speeds(positions, FPS)
    assert np.isnan(v["white"][100:110]).all()
    assert np.isfinite(v["white"][200])


def test_scoreboard_agreement_is_reported_per_inning():
    positions, live = build_track([("white", 1.5, 1.5), ("yellow", 1.5, 1.5)])
    innings = group_innings(segment_shots(positions, live, FPS))
    assert len(innings) == 2
    board = {}
    for inning, number in zip(innings, (7, 8)):
        for frame in range(inning.start_frame, inning.end_frame + 1, 30):
            board[frame] = number
    checked = reconcile_with_scoreboard(innings, board)
    assert [agrees for _, _, agrees in checked] == [True, True]


def test_scoreboard_disagreement_is_surfaced():
    positions, live = build_track([("white", 1.5, 1.5), ("white", 1.5, 1.5)])
    innings = group_innings(segment_shots(positions, live, FPS))
    assert len(innings) == 1
    inning = innings[0]
    # The scoreboard advanced mid-inning, so a player change was missed.
    board = {inning.start_frame: 4, inning.end_frame: 5}
    (_, numbers, agrees) = reconcile_with_scoreboard(innings, board)[0]
    assert agrees is False
    assert numbers == [4, 5]


# --- play outcomes and the structure of an inning -------------------------

from src.segmentation.shot_segmenter import Inning, audit_innings, label_outcomes


def _inning(cue, successes):
    """An inning whose plays end in the given outcomes."""
    shots = []
    for i, ok in enumerate(successes):
        shot = Shot(i * 100, i * 100 + 50, cue, {}, {}, True)
        shot.success = ok
        shots.append(shot)
    return Inning(1, cue, shots)


def test_an_inning_holds_one_more_play_than_it_scores():
    """The player shoots on while scoring and the inning ends on the miss."""
    inning = _inning("white", [True, True, True, False])
    ok, reason = inning.check()
    assert ok is True
    assert inning.points == 3
    assert len(inning.shots) == inning.points + 1


def test_a_miss_only_inning_is_one_play():
    inning = _inning("yellow", [False])
    assert inning.check()[0] is True
    assert inning.points == 0


def test_an_inning_that_never_missed_is_flagged():
    """Every play scoring means the miss that ended the inning was missed."""
    ok, reason = _inning("white", [True, True]).check()
    assert ok is False
    assert "exactly one" in reason


def test_a_miss_in_the_middle_is_flagged():
    """A failure that is not last means a player change went unnoticed."""
    ok, reason = _inning("white", [True, False, True, False]).check()
    assert ok is False


def test_unlabelled_plays_are_not_judged():
    inning = _inning("white", [True, False])
    inning.shots[0].success = None
    assert inning.check()[0] is None
    assert inning.points is None


def test_outcomes_come_from_the_shooters_score_rising():
    positions, live = build_track([("white", 1.5, 1.5), ("white", 1.5, 1.5), ("yellow", 1.5, 1.5)])
    innings = group_innings(segment_shots(positions, live, FPS))
    assert [len(i.shots) for i in innings] == [2, 1]
    white_inning, yellow_inning = innings

    scores = {}
    def note(frame, white, yellow):
        scores[frame] = {"white": white, "yellow": yellow}
    # White scores on its first play, misses on the second; yellow then misses.
    note(white_inning.shots[0].start_frame, 4, 9)
    note(white_inning.shots[1].start_frame, 5, 9)
    note(yellow_inning.shots[0].start_frame, 5, 9)
    note(yellow_inning.shots[0].end_frame + 200, 5, 9)

    label_outcomes(innings, scores)
    assert [s.success for s in white_inning.shots] == [True, False]
    assert [s.success for s in yellow_inning.shots] == [False]
    assert white_inning.check()[0] is True
    assert yellow_inning.check()[0] is True


def test_audit_counts_consistent_and_broken_innings():
    innings = [_inning("white", [True, False]), _inning("yellow", [True, True])]
    report = audit_innings(innings)
    assert report["innings"] == 2
    assert report["consistent"] == 1
    assert len(report["broken"]) == 1


def test_a_teleporting_detection_does_not_hand_the_shot_to_a_still_ball():
    """With an impossible reading discarded, a motionless ball must not win by
    default - that recorded shots for balls that never moved."""
    positions, live = build_track([("white", 2.0, 1.5)])
    # The white detector jumps across the table for two frames, as it does when
    # it latches onto a graphic; yellow sits still throughout.
    onset = int((REST_SECONDS + 0.05) * FPS)
    positions["white"][onset:onset + 2] += np.array([2500.0, 0.0])
    shots = segment_shots(positions, live, FPS)
    assert all(s.cue_ball != "yellow" for s in shots)


from src.segmentation.shot_segmenter import fill_short_gaps


def test_short_dropouts_are_interpolated_but_real_occlusions_are_not():
    track = np.stack([np.arange(60.0) * 10.0, np.zeros(60)], axis=1)
    positions = {"white": track.copy()}
    positions["white"][10:12] = np.nan      # two-frame flicker
    positions["white"][30:45] = np.nan      # a real occlusion
    filled = fill_short_gaps(positions, max_gap_frames=6)["white"]
    assert np.allclose(filled[10:12, 0], [100.0, 110.0])
    assert np.isnan(filled[30:45, 0]).all()


def test_gap_filling_does_not_invent_positions_at_the_ends():
    positions = {"white": np.full((20, 2), np.nan)}
    positions["white"][5] = (100.0, 100.0)
    filled = fill_short_gaps(positions)["white"]
    assert np.isnan(filled[0, 0]) and np.isnan(filled[19, 0])


def test_cue_identification_survives_unequal_detection_rates():
    """The cue ball must not be decided by which colour the detector sees more.

    Regression: judging departure by the first frame a ball was *seen* moving
    handed the shot to whichever ball was detected more often. Improving white
    from 90% to 97% of frames flipped four plays onto white and broke the
    innings. Displacement from the resting place is immune to that.
    """
    positions, live = build_track([("yellow", 2.0, 1.5)])
    # Yellow, the ball actually struck, is only seen every third frame; white
    # sits still but is seen in every frame.
    drop = np.ones(len(positions["yellow"]), dtype=bool)
    drop[::3] = False
    positions["yellow"][drop] = np.nan
    shots = segment_shots(positions, live, FPS)
    assert [s.cue_ball for s in shots] == ["yellow"]


# --- turns taken from the scoreboard, and what is recoverable ---------------

from src.segmentation.shot_segmenter import (
    audit_turns,
    diagnose_turn,
    find_motion_bursts,
    innings_from_turns,
    recover_missing_plays,
)


def test_turns_from_the_board_survive_a_turn_with_no_plays_found():
    """Deriving innings from the cue ball changing colour merges the innings
    on either side of a turn whose plays were all missed; the board does not."""
    positions, live = build_track([("white", 1.5, 1.5), ("white", 1.5, 1.5)])
    shots = segment_shots(positions, live, FPS)
    n = len(positions["white"])
    turns = [(0, n // 3, "white", 0), (n // 3 + 1, 2 * n // 3, "yellow", 0),
             (2 * n // 3 + 1, n - 1, "white", 0)]
    innings = innings_from_turns(turns, shots)
    assert [i.cue_ball for i in innings] == ["white", "yellow", "white"]
    assert len(innings) == 3, "the empty yellow turn still exists"


def test_motion_bursts_find_a_play_whose_start_was_never_shown():
    positions, live = build_track([("white", 2.0, 2.0)])
    live[: int((REST_SECONDS + 0.4) * FPS)] = False  # cut away over the strike
    assert segment_shots(positions, live, FPS) == [], "strict pass cannot open it"
    assert find_motion_bursts(positions, live, FPS), "but the motion is still there"


def test_recovered_plays_take_the_cue_ball_from_the_turn():
    """A play whose strike was never shown carries no evidence of which ball
    was struck, so it is taken from the board rather than guessed."""
    positions, live = build_track([("yellow", 2.0, 2.0)])
    live[: int((REST_SECONDS + 0.4) * FPS)] = False
    inning = Inning(1, "yellow", [])
    added = recover_missing_plays(inning, positions, live, FPS, 1, (0, len(live) - 1))
    assert added == 1
    assert inning.shots[0].cue_ball == "yellow"
    assert inning.shots[0].inferred is True


def test_a_turn_the_camera_never_showed_is_not_called_a_segmentation_failure():
    positions, live = build_track([("white", 1.5, 1.5)])
    live[:] = False
    inning = Inning(1, "white", [])
    verdict, visible, missing = diagnose_turn(
        inning, positions, live, FPS, 2, (0, len(live) - 1))
    assert verdict == "not_shown"
    assert visible == 0.0
    assert missing == 2


def test_motion_left_over_after_the_detected_plays_is_a_segmentation_failure():
    """"Missed" has to mean there is something still there to find. A turn that
    was on camera but holds no unexplained motion has nothing to recover, so
    the share of the turn spent on camera cannot be the test."""
    positions, live = build_track([("white", 1.5, 1.5), ("white", 1.5, 1.5)])
    # Hide the second strike but not the roll that follows it: the strict pass
    # cannot open the play, yet its motion is plainly on screen.
    second_strike = int((2 * REST_SECONDS + 1.5) * FPS)
    live[second_strike - int(1.2 * FPS) : second_strike + int(0.3 * FPS)] = False
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 1
    inning = Inning(1, "white", shots)
    verdict, _, missing = diagnose_turn(
        inning, positions, live, FPS, 2, (0, len(live) - 1))
    assert verdict == "missed"
    assert missing == 1


def test_a_turn_with_no_unexplained_motion_is_not_called_a_failure():
    positions, live = build_track([("white", 1.5, 1.5)])
    inning = Inning(1, "white", segment_shots(positions, live, FPS))
    verdict, _, missing = diagnose_turn(
        inning, positions, live, FPS, 3, (0, len(live) - 1))
    assert verdict == "not_shown", "nothing left over means nothing to recover"
    assert missing == 2


def test_audit_separates_the_two_kinds_of_gap():
    positions, live = build_track([("white", 1.5, 1.5)])
    shots = segment_shots(positions, live, FPS)
    n = len(live)
    blind = np.array(live)
    blind[:] = True
    turns = [(0, n - 1, "white", 0)]
    report = audit_turns([Inning(1, "white", shots)], turns, positions, blind, FPS)
    assert len(report["complete"]) == 1
    report = audit_turns([Inning(1, "white", [])], [(0, n - 1, "white", 0)],
                         positions, np.zeros(n, bool), FPS)
    assert len(report["not_shown"]) == 1


# --- which plays are fit to keep -------------------------------------------

from src.segmentation.shot_segmenter import (confirm_labels, play_rejections,
                                              usable_plays, yield_report)


def _play(cue="white", success=True, complete=True, inferred=False, full_layout=True):
    layout = {c: (100.0, 100.0) for c in BALL_COLOURS} if full_layout else {"white": (1.0, 1.0)}
    shot = Shot(0, 60, cue, layout, dict(layout), complete, inferred=inferred)
    shot.success = success
    return shot


def test_a_complete_labelled_play_is_kept():
    assert play_rejections(_play(), FPS) == []


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"complete": False, "success": True}, "truncated"),
        ({"inferred": True}, "inferred"),
        ({"success": None}, "unlabelled"),
        ({"full_layout": False}, "layout_incomplete"),
    ],
)
def test_unusable_plays_are_named(kwargs, reason):
    assert reason in play_rejections(_play(**kwargs), FPS)


def test_an_inning_that_disagrees_with_the_board_is_dropped_whole():
    """Even sound-looking plays go: with a play missing from the inning, the
    outcomes of the rest may have been read against the wrong strokes."""
    good = Inning(1, "white", [_play(success=True), _play(success=False)])
    broken = Inning(2, "white", [_play(success=True), _play(success=True)])
    kept, dropped = usable_plays([good, broken], FPS)
    assert len(kept) == 2
    assert dropped["inning_inconsistent"] == 2


def test_yield_counts_against_what_the_match_contained():
    innings = [Inning(1, "white", [_play(success=True), _play(success=False)])]
    report = yield_report(innings, 10, FPS)
    assert report["usable_plays"] == 2
    assert report["yield"] == pytest.approx(0.2)
    assert report["successes"] == 1 and report["failures"] == 1


def test_a_truncated_miss_is_still_usable_but_a_truncated_score_is_not():
    """The broadcast cuts away the instant a point is lost, so demanding a
    settled finish discards failures specifically - the half of the data the
    project exists to collect."""
    missed = _play(success=False, complete=True)
    missed.complete = False
    assert play_rejections(missed, FPS) == []

    scored = _play(success=True, complete=False)
    assert "truncated" in play_rejections(scored, FPS)


def test_a_miss_cut_off_too_early_to_read_is_rejected():
    shot = Shot(0, int(0.5 * FPS), "white", {c: (1.0, 1.0) for c in BALL_COLOURS}, {}, False)
    shot.success = False
    assert "too_little_observed" in play_rejections(shot, FPS)


def test_a_confirmed_label_survives_a_broken_inning():
    """A missing play elsewhere in the inning does not make this play's own
    outcome unreadable."""
    scoring = _play(success=True)
    scoring.start_frame, scoring.end_frame = 600, 900
    other = _play(success=True)
    other.start_frame, other.end_frame = 1200, 1500
    inning = Inning(1, "white", [scoring, other])
    assert inning.check()[0] is False, "two scores and no miss: a play is missing"

    scores = {}
    for frame in range(0, 2400, 60):
        # White scores during the first play and holds steady around both ends.
        scores[frame] = {"white": 4 if frame < 900 else 5}
    confirm_labels([inning], scores, FPS)
    assert scoring.label_confirmed is True

    kept, dropped = usable_plays([inning], FPS)
    assert scoring in kept, "a corroborated play is not thrown away with its inning"


def test_a_label_that_moves_on_the_boundary_is_not_confirmed():
    """A point landing right at the edge of a play could belong to a play that
    was never detected."""
    shot = _play(success=True)
    shot.start_frame, shot.end_frame = 600, 900
    inning = Inning(1, "white", [shot])
    scores = {frame: {"white": 4 if frame < 590 else 5} for frame in range(0, 2400, 60)}
    confirm_labels([inning], scores, FPS)
    assert shot.label_confirmed is False


def test_outcomes_are_plain_booleans():
    """numpy's bool is not `False`, and identity checks on it silently reported
    a dataset with no failed plays in it at all."""
    positions, live = build_track([("white", 1.5, 1.5)])
    innings = group_innings(segment_shots(positions, live, FPS))
    shot = innings[0].shots[0]
    scores = {f: {"white": np.int64(3)} for f in range(0, 400, 30)}
    label_outcomes(innings, scores)
    assert shot.success is False
    assert isinstance(shot.success, bool)


# --- replays and the failure budget ----------------------------------------

from src.segmentation.shot_segmenter import (MAX_BLIND_SECONDS, check_failure_budget,
                                              cue_travel_mm, mark_replays)


def _play_at(start, layout, cue="white"):
    shot = Shot(start, start + 300, cue, dict(layout), dict(layout), True)
    return shot


def test_the_same_layout_twice_is_a_replay_not_a_new_play():
    """Ball positions are continuous and the table is never reset, so two plays
    opening from the same three positions are one play shown twice."""
    layout = {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)}
    live_play = _play_at(0, layout)
    replay = _play_at(int(12 * FPS), {c: (x + 5, y - 4) for c, (x, y) in layout.items()})
    assert mark_replays([live_play, replay], FPS) == 1
    assert replay.replay_of is live_play
    assert live_play.replay_of is None
    assert "replay" in play_rejections(replay, FPS)


def test_a_genuinely_different_layout_is_not_a_replay():
    a = _play_at(0, {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)})
    b = _play_at(int(12 * FPS),
                 {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (1200.0, 900.0)})
    assert mark_replays([a, b], FPS) == 0


def test_a_replay_much_later_in_the_match_is_not_linked():
    layout = {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)}
    a = _play_at(0, layout)
    b = _play_at(int(600 * FPS), layout)
    assert mark_replays([a, b], FPS) == 0


def test_failures_cannot_outnumber_a_players_turns():
    """Each turn ends in exactly one miss, so this is a hard ceiling - passing
    it proves some labels are wrong, which no per-play test can establish."""
    def inning(cue, outcomes):
        shots = []
        for ok in outcomes:
            s = _play_at(0, {c: (1.0, 1.0) for c in BALL_COLOURS}, cue)
            s.success = ok
            shots.append(s)
        return Inning(1, cue, shots)

    budget = check_failure_budget([inning("white", [False]), inning("white", [False, False])])
    assert budget["white"]["turns"] == 2
    assert budget["white"]["failures"] == 3
    assert budget["white"]["over_budget"] == 1

    sane = check_failure_budget([inning("white", [True, False]), inning("white", [False])])
    assert sane["white"]["over_budget"] == 0


def test_a_play_ends_when_the_camera_leaves_the_table():
    """A play must not run on through whatever the director cuts to. Left
    unbounded, detected plays swallowed player close-ups and beauty shots -
    sixteen and nineteen seconds long - and were labelled failures because the
    score had already moved by then."""
    positions, live = build_track([("white", 2.0, 3.0)])
    cut_at = int((REST_SECONDS + 0.6) * FPS)
    live[cut_at:] = False
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 1
    shot = shots[0]
    assert not shot.complete
    blind = (shot.end_frame - cut_at) / FPS
    assert blind <= MAX_BLIND_SECONDS + 0.2, f"play ran {blind:.1f}s past the cut"


def test_cue_travel_measures_the_observed_path():
    positions, live = build_track([("white", 2.0, 1.0)])
    shot = segment_shots(positions, live, FPS)[0]
    travel = cue_travel_mm(shot, positions)
    # 2 m/s for a second, then the ball sits still through the settle.
    assert 1800 < travel < 2600, travel


# --- carrying a play across a camera cut -----------------------------------

from src.segmentation.shot_segmenter import can_continue, stitch_across_cuts


def test_a_play_is_carried_on_when_the_table_comes_back_to_it():
    """Ending at the cut leaves the path stopping before the second object
    ball - the part that decides the point."""
    positions, live = build_track([("white", 2.0, 4.0)])
    cut = slice(int((REST_SECONDS + 1.0) * FPS), int((REST_SECONDS + 2.5) * FPS))
    live[cut] = False
    shots = segment_shots(positions, live, FPS)
    assert len(shots) == 1
    shot = shots[0]
    assert shot.end_frame < cut.stop, "the strict pass stops at the cut"
    assert not shot.complete

    v = speeds(positions, FPS)
    at_rest, _, _, fastest = _rest_and_motion(v, np.asarray(live, dtype=bool))
    stitched = stitch_across_cuts(shot, positions, live, FPS, at_rest, fastest)
    assert stitched > cut.stop, "the play should continue past the cut"


def _fastest_of(positions, live):
    return _rest_and_motion(speeds(positions, FPS), np.asarray(live, dtype=bool))[3]


def test_balls_still_rolling_when_the_table_returns_means_the_shot_goes_on():
    positions, live = build_track([("white", 2.0, 3.0)])
    live[120:150] = False
    fastest = _fastest_of(positions, live)
    assert can_continue(fastest, 119, 150, FPS) is True


def test_balls_at_rest_when_the_table_returns_means_the_shot_is_over():
    """Distance cannot decide this: the table's own diagonal is 3.2 m, so after
    a third of a second any layout is 'reachable' and the bound says nothing."""
    positions, live = build_track([("white", 2.0, 1.0)])
    settled = int((REST_SECONDS + 1.3) * FPS)
    live[settled - 30 : settled] = False
    fastest = _fastest_of(positions, live)
    assert can_continue(fastest, settled - 31, settled, FPS) is False


def test_a_long_absence_ends_the_play():
    positions, live = build_track([("white", 2.0, 2.0)])
    live[120:] = False
    shot = segment_shots(positions, live, FPS)[0]
    end = stitch_across_cuts(shot, positions, live, FPS)
    assert end == shot.end_frame
