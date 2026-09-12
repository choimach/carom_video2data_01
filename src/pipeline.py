"""
Video in, plays out.

The work splits in two, and the split is what makes the thing usable. Reading a
match is expensive - about fifty minutes for two and a half hours of video, all
of it decoding - and produces a small array of ball positions and scoreboard
readings. Everything after that is cheap and runs on the array in seconds.

So the scan is done once and cached, and the analysis can be re-run as often as
the judgement changes. Every threshold in this project was settled by re-running
analysis over a cached scan; doing it any other way would have meant an hour per
attempt.
"""

import json
import os
import time
import warnings

import cv2
import numpy as np

from src.physics.ball_detector import BallDetector
from src.physics.carom import judge_shot
from src.physics.kinematics import KinematicsEngine
from src.physics.stroke import stroke_of
from src.physics.table_calibration import calibrate, CalibrationError
from src.segmentation.scoreboard_ocr import ScoreboardReader
from src.segmentation.shot_segmenter import (
    audit_turns,
    check_failure_budget,
    confirm_labels,
    cue_travel_mm,
    innings_from_turns,
    label_outcomes,
    mark_replays,
    play_rejections,
    recover_missing_plays,
    segment_shots,
    drop_unreachable_points,
)

BALL_ORDER = ("white", "yellow", "red")
JUDGEMENT_LOOKAHEAD_SECONDS = 2.0


def find_calibration(video_path, search_from=600.0, search_to=None, step=120.0,
                     target_error_mm=2.5, verbose=True):
    """Calibrate on the best frame found, not the first.

    A broadcast opens on a standby card and returns to the table at intervals,
    so the first frame that calibrates at all is often one with a rail partly
    hidden. Sampling on and keeping the best costs a few seconds and buys a
    calibration with every diamond in view.
    """
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    if search_to is None:
        search_to = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    best = None
    t = search_from
    while t < search_to:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = capture.read()
        if ok:
            try:
                found = calibrate(frame)
            except CalibrationError:
                found = None
            if found is not None:
                score = (found.diamond_count, -found.reprojection_error)
                if best is None or score > (best[1].diamond_count, -best[1].reprojection_error):
                    best = (t, found)
                # Stopping early used to wait for all 28 markers. A calibration
                # now reports the markers its homography actually kept, so a
                # table whose short rails carry only a few clean ones - the
                # Ankara table does - never reaches 28 and the search ran to the
                # end of the match for nothing. What the exit needs to know is
                # that the fit is good and rests on more than the bare minimum.
                if found.diamond_count >= 18 and found.reprojection_error <= target_error_mm:
                    break
        t += step
    capture.release()
    if best is None:
        raise CalibrationError(f"no frame of {video_path} shows a full table")
    if verbose:
        print(f"calibrated on t={best[0]:.0f}s: {best[1]}", flush=True)
    return best[1], fps


def find_play_start(video_path, calibration, search_from=0.0, search_to=None,
                    coarse_step=120.0, fine_step=10.0, verbose=True):
    """Where the match actually begins.

    A broadcast opens with a standby card or a reel of promotional video - half
    an hour on one of these VODs, over an hour on another - and scanning it is
    pure decoding for nothing. This walks forward coarsely until the table
    appears, then steps back and finds the first appearance to within a few
    seconds.

    Returns a second, or None if the table never shows up.
    """
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    if search_to is None:
        search_to = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    def table_at(second):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
        ok, frame = capture.read()
        return bool(ok and calibration.is_table_visible(frame))

    coarse = None
    t = search_from
    while t < search_to:
        if table_at(t):
            coarse = t
            break
        t += coarse_step
    if coarse is None:
        capture.release()
        return None

    # Step back over the coarse stride to find where it first appears.
    start = max(search_from, coarse - coarse_step)
    while start < coarse:
        if table_at(start):
            coarse = start
            break
        start += fine_step
    capture.release()
    if verbose:
        print(f"match starts around t={coarse:.0f}s "
              f"({coarse / 60:.0f} min of it is not the match)", flush=True)
    return coarse


def scan(video_path, track_path, start=0.0, end=None, calibration=None, verbose=True):
    """The expensive pass: ball positions and scoreboard readings, frame by frame.

    Saves to `track_path` and returns what it saved. Ball detection runs only
    where the table is actually on screen - about a quarter of a broadcast - and
    the scoreboard is read once a second, which is far more often than it
    changes.
    """
    warnings.filterwarnings("ignore")
    if calibration is None:
        calibration, fps = find_calibration(video_path, verbose=verbose)
    else:
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
        capture.release()

    capture = cv2.VideoCapture(video_path)
    if end is None:
        end = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

    detector = BallDetector(calibration)
    reader = ScoreboardReader()
    total = int((end - start) * fps)
    track = np.full((total, 7), np.nan, np.float32)
    boards = []
    began = time.time()

    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            track = track[:index]
            break
        if calibration.is_table_visible(frame):
            track[index, 0] = 1.0
            found = detector.detect(frame)
            for slot, colour in enumerate(BALL_ORDER):
                if colour in found:
                    track[index, 1 + 2 * slot], track[index, 2 + 2 * slot] = found[colour]["mm"]
        if index % int(fps) == 0:
            board = reader.read(frame)
            if board is not None:
                blank = lambda v: -1 if v is None else int(v)
                boards.append((
                    index, blank(board.inning),
                    blank(board.scores.get("white")), blank(board.runs.get("white")),
                    blank(board.scores.get("yellow")), blank(board.runs.get("yellow")),
                ))
        if verbose and index and index % 60000 == 0:
            spent = time.time() - began
            print(f"  {index}/{total} ({index / total * 100:.0f}%) "
                  f"{spent / 60:.1f} min, ETA {(total - index) / (index / spent) / 60:.0f} min",
                  flush=True)
    capture.release()

    os.makedirs(os.path.dirname(track_path) or ".", exist_ok=True)
    np.savez_compressed(
        track_path, track=track, boards=np.array(boards, dtype=np.int64),
        start=start, fps=fps, mm_per_px=calibration.mm_per_px,
        corners=calibration.corners, matrix=calibration.matrix,
        reprojection_error=calibration.reprojection_error,
    )
    if verbose:
        live = track[:, 0] == 1.0
        print(f"scanned {len(track)} frames in {(time.time() - began) / 60:.1f} min; "
              f"table visible {live.mean() * 100:.0f}%", flush=True)
    return track_path


def load_scan(track_path):
    data = np.load(track_path)
    track = data["track"]
    positions = {c: track[:, 1 + 2 * i:3 + 2 * i].astype(float) for i, c in enumerate(BALL_ORDER)}
    return {
        "positions": positions,
        "live": track[:, 0] == 1.0,
        "boards": data["boards"],
        "fps": float(data["fps"]),
        "start": float(data["start"]),
        "mm_per_px": float(data["mm_per_px"]),
        # Scans written before this field existed simply have no error to report.
        "reprojection_error": float(data["reprojection_error"])
        if "reprojection_error" in data.files else None,
    }


MATCH_TARGET = 50  # a carom match ends the moment a player reaches it
MAX_RUN_STEP = 3   # points a run may gain between two readings a second apart


def turns_from_board_rows(rows, target=MATCH_TARGET):
    """Player turns from the saved scoreboard readings.

    Only the player at the table has a run box, and that box counts the points
    made in this turn - so a turn's boundaries and its point total both come
    straight off the board, including turns whose plays were never shown.

    The match distance bounds what the board can be saying, and that is what
    keeps one bad reading from poisoning a whole match. A player on 37 cannot
    make a run of 72, because the match would have ended at 50; and once anyone
    reaches 50 there are no more turns to have, so the two scores can never sum
    past 99. Three of the first five matches broke one of those - white on 54,
    a match standing at 50-50, and a single turn read as 72 points that carried
    its match to 159 - and each was one misread box, not a broken match.
    """
    turns = []
    for frame, _inning, white, white_run, yellow, yellow_run in rows:
        active = [(c, r) for c, r in (("white", white_run), ("yellow", yellow_run)) if r >= 0]
        if len(active) != 1:
            continue
        colour, run = active[0][0], int(active[0][1])
        if run > target:
            continue  # no run can pass the match distance
        if turns and turns[-1][2] == colour and run >= turns[-1][3]:
            # A run climbs a point at a time, and the board is read once a
            # second while a play takes several, so a jump means a misread digit
            # and not a burst of scoring. One 72 read where the run stood at 5
            # took its match to 159 points, and clipping it to what was left to
            # score only moved the damage: the player hit 50 eighteen turns
            # early and the rest of the match was thrown away as a ceremony.
            if run - turns[-1][3] > MAX_RUN_STEP:
                continue
            turns[-1] = (turns[-1][0], int(frame), colour, run)
        else:
            turns.append((int(frame), int(frame), colour, run))
    return bound_to_match(turns, target)


def bound_to_match(turns, target=MATCH_TARGET):
    """Hold a run of turns to what the rules allow, and stop it at the win.

    A turn's points are clipped to what the player still had left to score, and
    everything after the winning point is dropped - it belongs to a ceremony, a
    replay or a second match on the same VOD, not to this one.
    """
    scored, bounded = {}, []
    for start, end, colour, points in turns:
        room = target - scored.get(colour, 0)
        if room <= 0:
            break
        points = min(int(points), room)
        scored[colour] = scored.get(colour, 0) + points
        bounded.append((start, end, colour, points))
        if scored[colour] >= target:
            break  # the match is over; whatever follows is not part of it
    return bounded


def label_from_inning_shape(result_innings, turns):
    """Let the shape of a turn say which of its plays scored.

    A turn ends when the player misses, so the only miss in it is its last
    play. Two things follow, and they are worth stating separately because one
    is much stronger than the other:

    * Any play with another play after it in the same inning scored. There is
      nothing to judge - a miss would have ended the turn. This holds whether
      or not the inning adds up, and it is where nearly all the gain is.
    * The last play detected in an inning is the turn's real last play only
      when the inning adds up: N points and N+1 plays. Where plays are missing,
      the turn may have gone on past what was seen, and nothing follows about
      the last one. There the cue ball's path is all there is.

    Measured against the clip labels you wrote by hand on the LIWC match, over
    the 93 plays this can speak for: 88 right, 95%. Watching the cue ball alone
    gets 84%. The gain is concentrated exactly where the reasoning above says
    it should be - innings that do not add up go from 81% to 94%, because the
    trajectory judge misses contacts and calls scoring plays misses, and a play
    with another play behind it cannot be a miss whatever its path looked like.
    """
    for inning, (_first, _last, _colour, points) in zip(result_innings, turns):
        complete = len(inning.shots) == points + 1
        for position, shot in enumerate(inning.shots):
            followed = position < len(inning.shots) - 1
            if followed:
                scored = True
            elif complete:
                scored = False
            else:
                continue  # the turn may have run on unseen; leave the trajectory
            shot.verdict_source = "inning"
            shot.inning_success = scored
            if not shot.inferred:
                # The shape knows what an unseen play was, but the video never
                # showed it, so it still has no trajectory to be used as data.
                shot.success = scored


def analyse(scan_data, recover=True):
    """The cheap pass: plays, innings, verdicts, and an account of what was lost."""
    live = scan_data["live"]
    fps, start = scan_data["fps"], scan_data["start"]
    # Clean before anything reads the tracks: a detection that jumped is not
    # noise to be averaged out, it is a position that was never occupied.
    positions = drop_unreachable_points(scan_data["positions"], fps)
    scan_data = dict(scan_data, positions=positions)

    turns = turns_from_board_rows(scan_data["boards"])
    shots = segment_shots(positions, live, fps)
    replays = mark_replays(shots, fps)
    distinct = [s for s in shots if s.replay_of is None]

    innings = innings_from_turns(turns, distinct)
    recovered = 0
    if recover:
        for inning, (first, last, _colour, points) in zip(innings, turns):
            recovered += recover_missing_plays(inning, positions, live, fps, points + 1, (first, last))

    ordered = [s for inning in innings for s in inning.shots]
    ordered.sort(key=lambda s: s.start_frame)
    # A recovered play stands for a stroke that was never on screen, so it has
    # no trajectory to judge and must not bound anyone else's. Letting these act
    # as the next play cut the judgement window of the real play before them and
    # turned two thirds of this match's points into misses.
    watched = [s for s in ordered if not s.inferred]
    lookahead = int(JUDGEMENT_LOOKAHEAD_SECONDS * fps)
    kinematics = KinematicsEngine(fps=fps)
    for position, shot in enumerate(watched):
        limit = watched[position + 1].start_frame if position + 1 < len(watched) else None
        scored, details = judge_shot(shot, positions, lookahead=lookahead, limit=limit)
        shot.success = bool(scored)
        # Kept apart from success, which the inning's shape may overwrite, so the
        # two witnesses can still be compared afterwards - and so a sample of
        # the plays they disagree on can be pulled out for review.
        shot.trajectory_success = bool(scored)
        shot.verdict = details
        shot.verdict_source = "trajectory"
        shot.cue_travel = cue_travel_mm(shot, positions)
        for field, value in stroke_of(shot, positions, details.get("events") or []).items():
            setattr(shot, field, value)
        path = positions[shot.cue_ball][shot.start_frame:shot.end_frame + 1]
        shot.cue_speed = kinematics.calculate_speed([p for p in path if np.isfinite(p).all()])
    for shot in ordered:
        if shot.inferred:
            shot.success = None       # unknowable: the stroke was never shown
            shot.verdict = {"reason": "the play was never on screen"}
            shot.verdict_source = None
            shot.cue_travel = None
            shot.cue_speed = None
            for field in ("thickness", "spin_y", "spin_x", "spin_rail"):
                setattr(shot, field, None)

    # The scoreboard is kept as a second opinion, not as the verdict.
    scores = {
        int(row[0]): {"white": max(row[2], 0) + max(row[3], 0),
                      "yellow": max(row[4], 0) + max(row[5], 0)}
        for row in scan_data["boards"]
    }
    # label_outcomes writes to shot.success, and copying the inning's list still
    # shares the shots themselves - so asking the board for a second opinion
    # overwrote the trajectory's verdict and turned 48 scored plays into 27.
    # Hold the verdict, let the board write, then put it back.
    verdicts = {id(s): s.success for s in ordered}
    label_outcomes(innings, scores)
    confirm_labels(innings, scores, fps)
    for shot in ordered:
        shot.scoreboard_success = shot.success
        shot.success = verdicts[id(shot)]

    label_from_inning_shape(result_innings=innings, turns=turns)

    for shot in ordered:
        shot.rejections = play_rejections(shot, fps)

    n = len(scan_data["live"])
    return {
        "turns": turns,
        "innings": innings,
        "plays": ordered,
        "replays": replays,
        "recovered": recovered,
        "expected_plays": sum(points + 1 for *_rest, points in turns),
        "audit": audit_turns(innings, turns, positions, live, fps, (0, n - 1)),
        "failure_budget": check_failure_budget(innings),
        "start": start,
        "fps": fps,
    }


def report(result, stream=print):
    """A short account of what a match gave up, and what it held back."""
    turns, plays = result["turns"], result["plays"]
    expected, fps, start = result["expected_plays"], result["fps"], result["start"]
    by_colour = {}
    for *_rest, colour, points in ((t[0], t[1], t[2], t[3]) for t in turns):
        entry = by_colour.setdefault(colour, [0, 0])
        entry[0] += 1
        entry[1] += points

    stream(f"turns {len(turns)}   expected plays {expected}   detected {len(plays)} "
           f"({len(plays) / expected * 100:.0f}%)")
    for colour, (count, points) in sorted(by_colour.items()):
        stream(f"   {colour:7s} {count:3d} turns, {points:3d} points")
    scored = sum(1 for s in plays if s.success)
    stream(f"verdicts: {scored} scored / {len(plays) - scored} missed   replays excluded {result['replays']}")
    usable = [s for s in plays if not s.rejections]
    stream(f"usable plays: {len(usable)}  ({len(usable) / expected * 100:.0f}% of the match)")
    reasons = {}
    for shot in plays:
        if shot.rejections:
            reasons[shot.rejections[0]] = reasons.get(shot.rejections[0], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        stream(f"   dropped {reason:22s} {count}")
    for colour, budget in result["failure_budget"].items():
        flag = "ok" if budget["over_budget"] == 0 else f"OVER BY {budget['over_budget']}"
        stream(f"   {colour:7s} turns {budget['turns']:3d}, misses {budget['failures']:3d}  {flag}")
    audit = result["audit"]
    for verdict in ("complete", "missed", "not_shown", "edge", "extra"):
        rows = audit.get(verdict) or []
        if rows:
            stream(f"   {verdict:11s} {len(rows):3d} turns, {sum(r['missing'] for r in rows):+4d} plays")


def _plain(value):
    """numpy scalars out of the way before JSON sees them.

    np.bool_ reports its class name as "bool", so the failure reads
    "Object of type bool is not JSON serializable" and looks impossible.
    """
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def export_json(result, path):
    """Everything a play is, in one file: layout, stroke, verdict, provenance."""
    start, fps = result["start"], result["fps"]
    plays = []
    for inning in result["innings"]:
        for number, shot in enumerate(inning.shots, start=1):
            verdict = getattr(shot, "verdict", {}) or {}
            plays.append({
                "inning": inning.number,
                "shot_number": number,
                "cue_ball": shot.cue_ball,
                "success": shot.success,
                "scoreboard_success": getattr(shot, "scoreboard_success", None),
                "trajectory_success": getattr(shot, "trajectory_success", None),
                "inning_success": getattr(shot, "inning_success", None),
                "verdict_source": getattr(shot, "verdict_source", None),
                "cushions_before_second": verdict.get("cushions"),
                "first_object_ball": verdict.get("first_ball"),
                "second_object_ball": verdict.get("second_ball"),
                "layout_mm": shot.start_positions,
                "final_mm": shot.end_positions,
                "cue_speed_ms": getattr(shot, "cue_speed", None),
                "thickness": getattr(shot, "thickness", None),
                "spin_y": getattr(shot, "spin_y", None),
                "spin_x": getattr(shot, "spin_x", None),
                "spin_rail": getattr(shot, "spin_rail", None),
                "cue_travel_mm": getattr(shot, "cue_travel", None),
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "start_second": start + shot.start_frame / fps,
                "end_second": start + shot.end_frame / fps,
                "complete": shot.complete,
                "inferred": shot.inferred,
                "label_confirmed": getattr(shot, "label_confirmed", False),
                "rejected_for": getattr(shot, "rejections", []),
            })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_plain({"expected_plays": result["expected_plays"], "plays": plays}),
                  handle, indent=1)
    return len(plays)


def play_trajectory(scan_data, shot, colours=BALL_ORDER, trim_trailing_rest=True):
    """The three balls' paths through one play, in table millimetres.

    Returns {colour: (n, 2) array}, NaN where a ball was not detected. This is
    the record the whole pipeline exists to produce - the layout says what the
    player faced, the verdict says how it ended, and this says what happened in
    between.

    Frames after everything has come to rest are dropped by default: a play's
    tail is the segmenter waiting out its rest window, and it is the same
    position repeated.
    """
    window = slice(shot.start_frame, shot.end_frame + 1)
    paths = {c: scan_data["positions"][c][window].copy() for c in colours}
    if trim_trailing_rest and len(paths[colours[0]]) > 1:
        moving = np.zeros(len(paths[colours[0]]), dtype=bool)
        for xy in paths.values():
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            moving[1:] |= np.nan_to_num(step, nan=0.0) > 1.0
        last = np.flatnonzero(moving)
        if last.size:
            end = min(len(moving), int(last[-1]) + 2)
            paths = {c: xy[:end] for c, xy in paths.items()}
    return paths


def export_trajectories(scan_data, result, path, usable_only=True):
    """Write every play's three paths to one .npz, keyed by inning and shot.

    One file per match rather than one per play: a hundred plays is a hundred
    thousand points, which is nothing to load at once and a nuisance to open a
    hundred times.
    """
    arrays, index = {}, []
    for inning in result["innings"]:
        for number, shot in enumerate(inning.shots, start=1):
            if usable_only and getattr(shot, "rejections", None):
                continue
            if shot.inferred:
                continue
            key = f"i{inning.number:03d}s{number:02d}"
            paths = play_trajectory(scan_data, shot)
            for colour, xy in paths.items():
                arrays[f"{key}_{colour}"] = xy.astype(np.float32)
            index.append({
                "key": key,
                "inning": inning.number,
                "shot_number": number,
                "cue_ball": shot.cue_ball,
                "success": bool(shot.success) if shot.success is not None else None,
                "frames": int(len(paths[BALL_ORDER[0]])),
                "start_second": result["start"] + shot.start_frame / result["fps"],
            })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, index=json.dumps(index), fps=result["fps"], **arrays)
    return len(index)


def cut_clips(video_path, result, directory, usable_only=True, lead_in=4.0, lead_out=7.0,
              width=960, crf=28, verbose=True):
    """One video file per play, for looking at what the numbers describe.

    Runs to the balls stopping rather than to the recorded end, and opens well
    before the strike: a clip that starts on the stroke does not show the layout
    it was played from.
    """
    import subprocess

    os.makedirs(directory, exist_ok=True)
    made = []
    for inning in result["innings"]:
        for number, shot in enumerate(inning.shots, start=1):
            if usable_only and getattr(shot, "rejections", None):
                continue
            if shot.inferred:
                continue
            begin = result["start"] + shot.start_frame / result["fps"]
            finish = result["start"] + shot.end_frame / result["fps"]
            name = (f"i{inning.number:03d}s{number:02d}_{shot.cue_ball}_"
                    f"{'score' if shot.success else 'miss'}_t{int(begin)}.mp4")
            out = os.path.join(directory, name)
            command = [
                "ffmpeg", "-v", "error", "-ss", f"{max(0.0, begin - lead_in):.2f}",
                "-i", video_path, "-t", f"{(finish - begin) + lead_in + lead_out:.2f}",
                "-an", "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf),
                "-preset", "veryfast", out, "-y",
            ]
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode == 0:
                shot.clip_path = out
                made.append(out)
            elif verbose:
                print(f"  clip failed for {name}: {done.stderr.strip()[:100]}", flush=True)
    return made
