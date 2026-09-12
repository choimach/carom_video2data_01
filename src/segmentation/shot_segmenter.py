"""
Shot and inning segmentation from ball trajectories.

Two boundaries matter and they are not the same one:

* a **shot** starts when the balls are at rest and a cue ball is struck, and
  ends when everything comes to rest again;
* an **inning** ends only when the player misses, so a run of successful shots
  by the same player is one inning containing many shots.

The player who is shooting is not written anywhere on the table - it is the
colour of the cue ball that moves first. In three-cushion each player owns one
cue ball for the whole match, so the cue ball changing colour between two shots
*is* the change of player, and therefore the inning boundary. Red is never a cue
ball, which makes the test a strong one: if red moves first, it is a tracking
error, not a shot.

This runs on vision alone. It is meant to be cross-checked against the inning
number read from the scoreboard: where the two disagree, the segmentation of
that stretch is not trustworthy and should be flagged rather than stored.
"""

import numpy as np

BALL_COLOURS = ("white", "yellow", "red")
CUE_COLOURS = ("white", "yellow")

# A ball centre is only accurate to about a pixel, which at broadcast scale is
# ~2.5 mm; at 60 fps that alone reads as 0.15 m/s. Speeds are therefore measured
# over several frames and thresholds sit well clear of that noise floor.
SPEED_BASELINE_FRAMES = 3
REST_SPEED_MS = 0.25
ONSET_SPEED_MS = 0.8
MIN_REST_SECONDS = 0.35
MIN_SHOT_SECONDS = 0.4
MAX_SHOT_SECONDS = 20.0
# The fastest a person can send a ball with a cue is about 14-16 m/s (50-56
# km/h); 18 leaves room above that rather than sitting on it. At 60 fps this is
# 300 mm between frames, so a quarter-metre step is a hard shot and not an
# error - an earlier limit of 9 m/s was discarding the opening frames of the
# hardest strokes. The point of the limit is to reject the detector jumping to
# another object, and the jumps that matter are metres, not centimetres.
MAX_CUE_SPEED_MS = 18.0
ONSET_JUMP_MARGIN_MM = 120.0
# How far a ball must leave its resting place to count as departed: comfortably
# past centroid jitter, and well inside a ball's own width so the reading comes
# within a frame or two of the strike.
DEPARTURE_MM = 25.0
# Once the camera leaves the table, what the balls did next is unverifiable, so
# the play ends there. Without this a play runs on through whatever the
# director cuts to - a player's face, a beauty shot of the three balls, a
# trajectory replay - until something in that other scene happens to look like
# balls at rest, and the whole stretch is recorded as one play.
MAX_BLIND_SECONDS = 1.0
# A three-cushion cue ball crosses the table several times. One that moved less
# than this never played a shot; it is a fragment of one, cut short by a false
# reading of rest.
MIN_CUE_TRAVEL_MM = 500.0


class Shot:
    """One play: the balls are at rest, a cue ball is struck, everything settles.

    This is the unit the whole pipeline exists to produce - a layout, a stroke,
    and whether it scored.
    """

    def __init__(self, start_frame, end_frame, cue_ball, start_positions, end_positions, complete,
                 inferred=False):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cue_ball = cue_ball
        self.start_positions = start_positions  # {colour: (x_mm, y_mm)} - the layout played from
        self.end_positions = end_positions
        self.complete = complete  # False when the camera cut away before the balls stopped
        self.success = None  # True/False once the scoreboard has been read; None while unknown
        # True when the strike itself was never on screen and the play was
        # reconstructed from the motion that followed it.
        self.inferred = inferred
        # Set by confirm_labels(): the outcome above is attributable to this
        # play on its own, not merely to the inning it sits in.
        self.label_confirmed = False
        # Set by mark_replays(): the earlier play this one is a second showing
        # of, or None when it is the live play.
        self.replay_of = None

    def duration(self, fps):
        return (self.end_frame - self.start_frame) / fps

    def __repr__(self):
        flags = "" if self.complete else " truncated"
        flags += " inferred" if self.inferred else ""
        return f"<Shot {self.start_frame}-{self.end_frame} cue={self.cue_ball}{flags}>"


class Inning:
    def __init__(self, number, cue_ball, shots):
        self.number = number
        self.cue_ball = cue_ball
        self.shots = shots

    @property
    def start_frame(self):
        return self.shots[0].start_frame

    @property
    def end_frame(self):
        return self.shots[-1].end_frame

    @property
    def points(self):
        """Points scored in this inning, or None while the plays are unlabelled."""
        if any(shot.success is None for shot in self.shots):
            return None
        return sum(1 for shot in self.shots if shot.success)

    def check(self):
        """Test the inning against the structure of the game.

        A player keeps shooting while scoring and the inning ends on the miss,
        so an inning holds exactly (points scored + 1) plays, of which only the
        last one failed. Any other shape means a play was missed or invented,
        and this is the one check that can say so per inning rather than in
        aggregate.

        Returns (ok, reason). ok is None while the plays are unlabelled.
        """
        if any(shot.success is None for shot in self.shots):
            return None, "plays not labelled yet"
        failures = [i for i, shot in enumerate(self.shots) if not shot.success]
        if len(failures) != 1:
            return False, f"{len(failures)} failed plays; an inning ends on exactly one"
        if failures[0] != len(self.shots) - 1:
            return False, f"the failed play is #{failures[0] + 1} of {len(self.shots)}, not the last"
        return True, f"{self.points} points in {len(self.shots)} plays"

    def __repr__(self):
        return f"<Inning {self.number} cue={self.cue_ball} shots={len(self.shots)}>"


def speeds(positions, fps, baseline=SPEED_BASELINE_FRAMES):
    """Speed in m/s per colour, measured over a multi-frame baseline.

    Returns {colour: array of shape (N,)}; NaN wherever either end of the
    baseline is missing.
    """
    out = {}
    for colour, xy in positions.items():
        n = len(xy)
        v = np.full(n, np.nan)
        if n > baseline:
            delta = xy[baseline:] - xy[:-baseline]
            v[baseline:] = np.linalg.norm(delta, axis=1) / (baseline / fps) / 1000.0
        out[colour] = v
    return out


def fill_short_gaps(positions, max_gap_frames=6):
    """Bridge brief detection dropouts by interpolating between the ends.

    A ball does not vanish for two frames and reappear, but a colour detector
    does lose one to a highlight, a shadow, or the cue crossing it - here white
    flickers out in roughly a tenth of its frames, almost always for one or two
    at a time. Anything longer than a few frames is a real occlusion and is left
    as a gap.

    Do NOT run this before identifying the cue ball. A bridged gap that spans
    the strike makes the ball look like it left at the start of the gap, which
    is earlier than the ball that actually moved first - applying it there
    reassigned four plays to the wrong player and broke the alternation the
    rules guarantee. It is for exporting trajectories, not for judging onsets.
    """
    filled = {}
    for colour, xy in positions.items():
        out = np.array(xy, dtype=float, copy=True)
        known = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
        if known.sum() < 2:
            filled[colour] = out
            continue
        indices = np.where(known)[0]
        for a, b in zip(indices[:-1], indices[1:]):
            span = b - a
            if 1 < span <= max_gap_frames + 1:
                for axis in (0, 1):
                    out[a + 1:b, axis] = np.interp(
                        np.arange(a + 1, b), [a, b], [out[a, axis], out[b, axis]]
                    )
        filled[colour] = out
    return filled


def _rest_and_motion(v, live):
    """Per-frame verdicts from the balls that are actually visible."""
    def fastest_of(colours):
        stacked = np.vstack([v[c] for c in colours])
        finite = np.isfinite(stacked)
        # -inf rather than NaN so an all-missing frame needs no nanmax, which
        # would warn on every such column.
        return finite.any(axis=0), np.max(np.where(finite, stacked, -np.inf), axis=0)

    seen, fastest = fastest_of(BALL_COLOURS)
    usable = live & seen
    at_rest = usable & (fastest < REST_SPEED_MS)
    # Only white and yellow can be struck, so only they can open a shot.
    _, cue_fastest = fastest_of(CUE_COLOURS)
    onset = usable & (cue_fastest > ONSET_SPEED_MS)
    return at_rest, onset, usable, np.where(np.isfinite(fastest), fastest, 0.0)


def _rest_position(xy, last_rest_frame, window):
    """Where a ball was sitting before the strike, from the frames it was seen."""
    lo = max(0, last_rest_frame - window)
    seen = xy[lo:last_rest_frame + 1]
    seen = seen[np.isfinite(seen[:, 0]) & np.isfinite(seen[:, 1])]
    return np.median(seen, axis=0) if len(seen) else None


def _first_sustained_departure(displacement, confirmations=3):
    """First reading past the departure threshold that the next ones confirm.

    A struck ball does not return to where it was sitting, so a lone reading
    out past the threshold is a misdetection rather than a departure. Without
    this, one stray white blob - the detector places about one in forty
    somewhere wrong - claims the shot from the ball that was really struck,
    because a single frame is all the test looks at.
    """
    seen = np.where(np.isfinite(displacement))[0]
    for position, index in enumerate(seen):
        if displacement[index] <= DEPARTURE_MM:
            continue
        following = seen[position + 1:position + 1 + confirmations]
        if len(following) < confirmations:
            return None
        if np.all(displacement[following] > DEPARTURE_MM):
            return int(index)
    return None


def _cue_ball_at(positions, v, frame, fps, last_rest_frame, lookback=8):
    """Which cue ball was struck: the one that left its resting place first.

    Not the fastest one. In three-cushion the opponent's cue ball is an object
    ball, so a shot that drives the cue ball into it has both white and yellow
    travelling fast within a fifth of a second, often with the struck-into ball
    the faster of the two. Only the order of departure separates them, since
    nothing moves before it is hit.

    Departure is measured as distance from the resting position, not as the
    first frame the ball is seen moving. Those are not the same test when the
    two balls are detected at different rates: the better-detected ball is seen
    leaving sooner simply because it is seen more often, and the shot gets
    handed to it. Fixing white's detection from 90% to 97% of frames did
    exactly that - four plays flipped to white, four innings vanished, and the
    scoreboard went from three points short to four points over. Displacement
    grows monotonically once a ball is struck, so a missed frame delays the
    reading by that frame rather than changing which ball crossed first.
    """
    window = int(0.6 * fps)
    start = max(0, frame - lookback)
    rest_window = max(1, int(MIN_REST_SECONDS * fps))

    candidates = []
    for colour in CUE_COLOURS:
        peak = v[colour][frame:frame + window]
        peak = peak[np.isfinite(peak) & (peak <= MAX_CUE_SPEED_MS)]
        # The ball has to be moving like a struck ball: above the onset speed
        # and below anything a cue can produce. Without the lower bound, a
        # motionless ball wins by default once the impossible reading is gone.
        if not peak.size or peak.max() <= ONSET_SPEED_MS:
            continue
        rest = _rest_position(positions[colour], last_rest_frame, rest_window)
        if rest is None:
            continue
        segment = positions[colour][start:frame + window]
        displacement = np.linalg.norm(segment - rest, axis=1)
        departed = _first_sustained_departure(displacement)
        if departed is None:
            continue
        candidates.append((departed, -float(peak.max()), colour))
    if not candidates:
        return None
    return min(candidates)[2]


def _last_seen_before(xy, frame, window):
    """Where a ball was last detected in the `window` frames before `frame`."""
    for index in range(frame, max(-1, frame - window), -1):
        if 0 <= index < len(xy) and np.isfinite(xy[index]).all():
            return index, xy[index]
    return None, None


def _is_continuous_onset(xy, start, last_rest_frame, fps):
    """Did this cue ball actually travel here, or did the detector jump to it?

    A ball that was never seen during the rest before the shot, or that turns up
    further away than it could possibly have rolled, is a different object
    wearing the same colour - a sponsor graphic, a shirt, a reflection.
    """
    window = max(1, start - last_rest_frame) + int(0.5 * fps)
    seen_frame, rest_position = _last_seen_before(xy, start, window)
    if rest_position is None:
        return False
    elapsed = max(1, start - seen_frame) / fps
    allowed = MAX_CUE_SPEED_MS * 1000.0 * elapsed + ONSET_JUMP_MARGIN_MM
    onset_position = xy[start] if np.isfinite(xy[start]).all() else rest_position
    return float(np.linalg.norm(onset_position - rest_position)) <= allowed


def _positions_at(positions, frame, live, search=30):
    """Ball positions at a frame, falling back to the nearest frame that has them."""
    out = {}
    for colour, xy in positions.items():
        for offset in range(search):
            for index in (frame - offset, frame + offset):
                if 0 <= index < len(xy) and live[index] and np.isfinite(xy[index]).all():
                    out[colour] = (float(xy[index][0]), float(xy[index][1]))
                    break
            if colour in out:
                break
    return out


def segment_shots(positions, live, fps):
    """Split a tracked stretch into shots.

    positions: {colour: (N, 2) array of table millimetres, NaN where unseen}
    live:      (N,) boolean, True where the main camera is on the table
    """
    live = np.asarray(live, dtype=bool)
    v = speeds(positions, fps)
    at_rest, onset, usable, _ = _rest_and_motion(v, live)

    min_rest = max(1, int(MIN_REST_SECONDS * fps))
    min_shot = max(1, int(MIN_SHOT_SECONDS * fps))
    max_shot = int(MAX_SHOT_SECONDS * fps)

    # Measuring speed over several frames smears the strike across that many
    # frames, so between "no longer at rest" and "clearly struck" there are a
    # few frames that are neither. They must not erase the rest that preceded
    # them, or every shot is rejected for having started from nowhere.
    grace = SPEED_BASELINE_FRAMES + 3

    shots = []
    rest_run = 0
    since_rest = 0
    last_rest_frame = 0
    frame = 0
    n = len(live)
    while frame < n:
        if at_rest[frame]:
            rest_run += 1
            since_rest = 0
            last_rest_frame = frame
            frame += 1
            continue
        if usable[frame]:
            since_rest += 1
            if since_rest > grace:
                # Movement this long after the last rest is a camera cut landing
                # mid-roll, or noise; neither starts a shot.
                rest_run = 0
        if not (onset[frame] and rest_run >= min_rest):
            frame += 1
            continue

        start = frame
        cue = _cue_ball_at(positions, v, start, fps, last_rest_frame)
        if cue is None or not _is_continuous_onset(positions[cue], start, last_rest_frame, fps):
            rest_run = 0
            since_rest = 0
            frame += 1
            continue

        # Walk forward to the point where everything has settled again.
        end, settled, last_usable = start, 0, start
        max_blind = int(MAX_BLIND_SECONDS * fps)
        cursor = start + min_shot
        while cursor < n and cursor - start < max_shot:
            if usable[cursor]:
                last_usable = cursor
            elif cursor - last_usable > max_blind:
                break  # the camera has left the table; the rest is unverifiable
            if at_rest[cursor]:
                settled += 1
                if settled >= min_rest:
                    end = cursor
                    break
            elif usable[cursor]:
                settled = 0
            cursor += 1
        complete = end > start
        if not complete:
            # The balls were never seen stopping, so the play ends where the
            # camera left it, not at the time limit. Running it out to the cap
            # and resuming after would skip the next twenty seconds of table -
            # which is how a truncated play came to swallow the play that
            # followed it, in two of the turns the scoreboard flagged as short.
            end = max(last_usable, min(n - 1, start + min_shot))

        shots.append(
            Shot(
                start_frame=start,
                end_frame=end,
                cue_ball=cue,
                start_positions=_positions_at(positions, start, live),
                end_positions=_positions_at(positions, end, live),
                complete=complete,
            )
        )
        rest_run = 0
        since_rest = 0
        frame = end + 1

    return shots


def group_innings(shots, first_number=1):
    """Group shots into innings: the cue ball changing colour means a new player."""
    innings = []
    for shot in shots:
        if innings and innings[-1].cue_ball == shot.cue_ball:
            innings[-1].shots.append(shot)
        else:
            innings.append(Inning(first_number + len(innings), shot.cue_ball, [shot]))
    return innings


def _score_near(scores, colour, frame, prefer_before):
    """The shooter's running score as read closest to a frame."""
    candidates = [f for f in scores if colour in scores[f]]
    if not candidates:
        return None
    side = [f for f in candidates if (f <= frame if prefer_before else f >= frame)]
    chosen = (max(side) if prefer_before else min(side)) if side else min(
        candidates, key=lambda f: abs(f - frame)
    )
    return scores[chosen][colour]


def label_outcomes(innings, scores, settle_frames=30):
    """Mark each play as scoring or not from the scoreboard.

    scores: {frame: {colour: running score}} as read by OCR, where the colour is
    the player's cue ball.

    A play scored if the shooter's score is higher after the balls settle than
    it was when the play began. Labelling each play on its own, rather than
    dividing an inning's points among its plays, means a missed play shows up as
    a broken inning in Inning.check() instead of quietly shifting every label.
    """
    for inning in innings:
        colour = inning.cue_ball
        for shot in inning.shots:
            before = _score_near(scores, colour, shot.start_frame, prefer_before=True)
            after = _score_near(scores, colour, shot.end_frame + settle_frames, prefer_before=False)
            # bool(), not the numpy comparison result: a numpy bool fails
            # `is False`, which silently counted every failed play as neither a
            # success nor a failure and reported zero failures in the dataset.
            shot.success = None if before is None or after is None else bool(after > before)
    return innings


def audit_innings(innings):
    """Summarise how many innings have the shape the rules require."""
    checked = [(inning,) + inning.check() for inning in innings]
    known = [c for c in checked if c[1] is not None]
    return {
        "innings": len(innings),
        "checked": len(known),
        "consistent": sum(1 for c in known if c[1]),
        "broken": [(c[0], c[2]) for c in known if not c[1]],
    }


def reconcile_with_scoreboard(innings, scoreboard_innings):
    """Compare vision-derived innings against the scoreboard's inning number.

    scoreboard_innings: {frame: inning_number} as read by OCR.

    Returns a list of (inning, scoreboard_number, agrees). Disagreement means a
    shot was missed or invented somewhere in that stretch - the shots there
    should be flagged, not silently stored.
    """
    if not scoreboard_innings:
        return [(inning, None, None) for inning in innings]

    frames = np.array(sorted(scoreboard_innings))
    result = []
    for inning in innings:
        window = frames[(frames >= inning.start_frame) & (frames <= inning.end_frame)]
        if not window.size:
            result.append((inning, None, None))
            continue
        values = [scoreboard_innings[int(f)] for f in window]
        # A correctly segmented inning spans exactly one scoreboard inning number.
        unique = sorted(set(values))
        result.append((inning, unique[0] if len(unique) == 1 else unique, len(unique) == 1))
    return result


def find_motion_bursts(positions, live, fps, span=None, min_seconds=0.5, merge_gap_seconds=1.0):
    """Stretches where a ball is moving, whether or not the strike was seen.

    segment_shots() only opens a play it watched begin, from rest. That is the
    right default - it is what makes the cue ball identifiable - but it cannot
    see a play the director cut away from and returned to mid-roll, and those
    plays are invisible to it rather than merely inaccurate.

    Bursts are merged across short gaps, because a camera cut inside one play
    would otherwise split it into two.
    """
    live = np.asarray(live, dtype=bool)
    v = speeds(positions, fps)
    _, _, usable, fastest = _rest_and_motion(v, live)
    moving = usable & (fastest > REST_SPEED_MS)

    lo, hi = (0, len(moving)) if span is None else span
    merge_gap = int(merge_gap_seconds * fps)
    bursts, start, gap = [], None, 0
    for index in range(max(0, lo), min(len(moving), hi)):
        if moving[index]:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > merge_gap:
                bursts.append((start, index - gap))
                start, gap = None, 0
    if start is not None:
        bursts.append((start, min(len(moving), hi) - 1))
    return [(a, b) for a, b in bursts if (b - a) / fps >= min_seconds]


def _burst_is_covered(burst, shots, tolerance=6):
    """Has a play actually been opened on this stretch of motion?

    Overlap is not the test. A play the camera cut away from never settles, so
    it runs on until the table comes back - straight over the next play's
    motion. Asked only whether a shot overlaps, that stretch looks accounted
    for and the swallowed play stays invisible. Asking whether a shot *starts*
    here separates the two.
    """
    start, end = burst
    return any(start - tolerance <= shot.start_frame <= end for shot in shots)


def innings_from_turns(turns, shots, first_number=1):
    """Innings taken from the scoreboard, with the detected plays slotted in.

    Deriving innings from the cue ball changing colour fails in one particular
    way: if every play of a turn is missed, that turn leaves no trace and the
    two innings around it silently merge into one. The board does not have that
    problem - it says whose turn it is even when nothing was tracked - so the
    turns come from there and vision only fills them.
    """
    innings = []
    for number, (start, end, colour, _) in enumerate(turns, start=first_number):
        inside = [s for s in shots if start <= s.start_frame <= end]
        innings.append(Inning(number, colour, inside))
    return innings


def recover_missing_plays(inning, positions, live, fps, expected_plays, span):
    """Add plays for motion inside a turn that the strict pass could not open.

    Only used where the board says plays are missing, and the cue ball is taken
    from the turn rather than guessed at, because a play whose strike was never
    shown carries no evidence of which ball was struck. Recovered plays are
    marked inferred so that anything built on them can tell them apart from
    plays that were watched from rest.
    """
    if len(inning.shots) >= expected_plays:
        return 0
    recovered = []
    added = 0
    for a, b in find_motion_bursts(positions, live, fps, span=span):
        if _burst_is_covered((a, b), inning.shots) or any(
            a <= end and start <= b for start, end in recovered
        ):
            continue
        inning.shots.append(
            Shot(
                start_frame=a,
                end_frame=b,
                cue_ball=inning.cue_ball,
                start_positions=_positions_at(positions, a, np.asarray(live, dtype=bool)),
                end_positions=_positions_at(positions, b, np.asarray(live, dtype=bool)),
                complete=False,
                inferred=True,
            )
        )
        recovered.append((a, b))
        added += 1
        if len(inning.shots) >= expected_plays:
            break
    inning.shots.sort(key=lambda s: s.start_frame)
    return added


# Below this share of a turn spent on the main camera, the plays in it were
# never shown and no amount of re-searching will find them.
MIN_VISIBLE_FRACTION = 0.10


def diagnose_turn(inning, positions, live, fps, expected_plays, span, window=None):
    """Why a turn does not hold the plays the scoreboard says it should.

    The distinction that matters for a dataset: a play the segmenter missed can
    be recovered, a play the broadcast never showed cannot. Asking how much of
    the turn was on the main camera is too blunt to separate them - a turn can
    be on camera a quarter of its length and still have spent all of it on the
    play that was found. The test that works is whether any motion inside the
    turn is left over once the detected plays are accounted for: if there is
    none, the missing play left no trace to recover.
    """
    start, end = span
    live = np.asarray(live, dtype=bool)
    segment = live[start:end + 1]
    visible = float(segment.mean()) if segment.size else 0.0
    missing = expected_plays - len(inning.shots)

    if missing < 0:
        return "extra", visible, missing
    if missing == 0:
        return "complete", visible, 0
    # A turn running past either edge of the analysed stretch is missing plays
    # that lie outside it, which is not a failure of anything here.
    if window is not None and (start <= window[0] or end >= window[1]):
        return "edge", visible, missing
    if visible < MIN_VISIBLE_FRACTION:
        return "not_shown", visible, missing

    leftover = [
        burst
        for burst in find_motion_bursts(positions, live, fps, span=span)
        if not _burst_is_covered(burst, inning.shots)
    ]
    return ("missed" if leftover else "not_shown"), visible, missing


def audit_turns(innings, turns, positions, live, fps, window=None):
    """Per-turn accounting of what was found, missed, or never on screen."""
    report = {"complete": [], "missed": [], "not_shown": [], "extra": [], "edge": []}
    for inning, (start, end, _, points) in zip(innings, turns):
        verdict, visible, missing = diagnose_turn(
            inning, positions, live, fps, points + 1, (start, end), window
        )
        report[verdict].append(
            {"inning": inning.number, "cue": inning.cue_ball, "expected": points + 1,
             "found": len(inning.shots), "missing": missing, "visible": visible}
        )
    return report


# A failed play ends the inning, and the broadcast cuts to the player's face
# the moment it is clear the point is gone - so the balls are almost never seen
# stopping. Requiring a settled finish therefore throws away failures
# specifically, which is how a first pass over this match produced 7 usable
# plays, all of them successes. What a failure has to carry is the layout it
# was played from and enough of the stroke to characterise it; the tail of the
# roll adds little once no point was scored.
MIN_OBSERVED_SECONDS = 1.0


def play_rejections(shot, fps):
    """Every reason this play is unfit for the dataset; empty means it is usable.

    The bar differs by outcome on purpose. For a scoring play the path itself
    is the subject, so a play cut short is no use. For a miss, the question is
    what was on the table and how it was struck.
    """
    reasons = []
    if shot.replay_of is not None:
        reasons.append("replay")
    if shot.inferred:
        reasons.append("inferred")
    if shot.success is None:
        reasons.append("unlabelled")
    if len(shot.start_positions) != len(BALL_COLOURS):
        reasons.append("layout_incomplete")
    if shot.success:
        if not shot.complete:
            reasons.append("truncated")
        elif len(shot.end_positions) != len(BALL_COLOURS):
            reasons.append("no_final_layout")
    elif shot.duration(fps) < MIN_OBSERVED_SECONDS:
        reasons.append("too_little_observed")
    return reasons


def usable_plays(innings, fps, require_consistent_inning=True):
    """The plays worth keeping, and why the rest were dropped.

    A play in an inning that does not match the scoreboard is kept only if its
    own label is corroborated - see confirm_labels(). Dropping such innings
    whole is the safe default, but on this match it discarded 84 of 100
    detected plays over a missing play elsewhere in the same inning.
    """
    kept, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for inning in innings:
        consistent = inning.check()[0] is True
        for shot in inning.shots:
            if require_consistent_inning and not consistent and not getattr(
                shot, "label_confirmed", False
            ):
                drop("inning_inconsistent")
                continue
            reasons = play_rejections(shot, fps)
            if reasons:
                drop(reasons[0])
            else:
                kept.append(shot)
    return kept, dropped


def yield_report(innings, expected_plays, fps):
    """How many usable plays a match gave up, against how many it contained."""
    kept, dropped = usable_plays(innings, fps)
    detected = sum(len(i.shots) for i in innings)
    return {
        "expected_plays": expected_plays,
        "detected_plays": detected,
        "usable_plays": len(kept),
        "yield": len(kept) / expected_plays if expected_plays else 0.0,
        "successes": sum(1 for s in kept if s.success),
        "failures": sum(1 for s in kept if s.success is False),
        "dropped": dropped,
    }


def confirm_labels(innings, scores, fps, quiet_seconds=1.0, settle_seconds=0.5):
    """Mark plays whose outcome can be pinned on that play alone.

    An inning failing its check does not make every play in it wrong - it means
    a play is missing somewhere, and the plays either side of the hole may still
    be readable. A label is safe when the shooter's score is steady just before
    the stroke and steady again just after it settles: whatever the board did in
    between belongs to this play and nothing else. Where the score moves right
    on a boundary, the point could belong to a play that was never detected, and
    the label is left unconfirmed.
    """
    quiet = int(quiet_seconds * fps)
    settle = int(settle_seconds * fps)
    for inning in innings:
        colour = inning.cue_ball
        for shot in inning.shots:
            before = _score_near(scores, colour, shot.start_frame, prefer_before=True)
            earlier = _score_near(scores, colour, shot.start_frame - quiet, prefer_before=True)
            after = _score_near(scores, colour, shot.end_frame + settle, prefer_before=False)
            later = _score_near(scores, colour, shot.end_frame + settle + quiet, prefer_before=False)
            shot.label_confirmed = (
                None not in (before, earlier, after, later)
                and earlier == before
                and after == later
                and shot.success is not None
            )
    return innings


# Two plays cannot start from the same three-ball layout: positions are
# continuous and a table is never reset mid-match. A repeat is the broadcast
# showing the play again.
REPLAY_LAYOUT_TOLERANCE_MM = 60.0
REPLAY_MAX_GAP_SECONDS = 120.0


def _layout_vector(shot):
    if len(shot.start_positions) != len(BALL_COLOURS):
        return None
    return np.array([shot.start_positions[c] for c in BALL_COLOURS], dtype=float)


def mark_replays(shots, fps, tolerance_mm=REPLAY_LAYOUT_TOLERANCE_MM,
                 max_gap_seconds=REPLAY_MAX_GAP_SECONDS):
    """Flag plays that are a second showing of one already seen.

    A replay is tracked exactly like live play and there is nothing in the
    trajectory to say otherwise - but the scoreboard does not move during it,
    so the replay of a scoring play gets labelled a miss. That is how this
    match came to hold 53 failures when the rules allow at most 37: every
    replay both invented a failure and hid the success it was replaying.

    The test is the opening layout. Ball positions are continuous and the table
    is never reset, so two plays starting from the same three positions are the
    same play shown twice; the eleven pairs found here agreed to 1-11 mm.
    """
    max_gap = max_gap_seconds * fps
    originals = []
    replays = 0
    for shot in shots:
        shot.replay_of = None
        layout = _layout_vector(shot)
        if layout is None:
            originals.append((shot, None))
            continue
        for earlier, earlier_layout in originals:
            if earlier_layout is None:
                continue
            if shot.start_frame - earlier.start_frame > max_gap:
                continue
            if np.linalg.norm(layout - earlier_layout, axis=1).max() < tolerance_mm:
                shot.replay_of = earlier
                replays += 1
                break
        if shot.replay_of is None:
            originals.append((shot, layout))
    return replays


def check_failure_budget(innings):
    """A player cannot miss more often than they came to the table.

    Each turn ends in exactly one miss, so failures per cue ball can never
    exceed that player's number of turns. Exceeding it does not mean some
    labels are marginal - it means some are certainly wrong, which no
    per-play test can tell you on its own.
    """
    turns, failures = {}, {}
    for inning in innings:
        turns[inning.cue_ball] = turns.get(inning.cue_ball, 0) + 1
        failures[inning.cue_ball] = failures.get(inning.cue_ball, 0) + sum(
            1 for shot in inning.shots if shot.success is False
        )
    return {
        colour: {
            "turns": turns[colour],
            "failures": failures.get(colour, 0),
            "over_budget": max(0, failures.get(colour, 0) - turns[colour]),
        }
        for colour in turns
    }


def cue_travel_mm(shot, positions):
    """How far the cue ball was actually seen to move during the play."""
    xy = positions[shot.cue_ball][shot.start_frame:shot.end_frame + 1]
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


# Straight-line distance between the furthest corners of the playing surface.
# No ball can be further from where it was than this, whatever the gap, so a
# reachability bound on its own admits any layout once the gap passes a third
# of a second - which is why a shot is carried on by motion, not by distance.
TABLE_DIAGONAL_MM = float(np.hypot(2844.0, 1422.0))


def can_continue(fastest, last_seen, resumed, fps, look_seconds=0.3):
    """Is the table coming back to the same shot, still in progress?

    A three-cushion shot rolls for several seconds and the director cuts away
    mid-roll, so ending a play at the cut leaves its path stopping before the
    second object ball - the part that decides the point. On this match that
    truncation accounted for 46 of the 49 plays the geometric judge could not
    call.

    The evidence that the shot is still running is that something is still
    moving when the table reappears. If everything is at rest, the shot ended
    while off screen and whatever comes next is a new one.
    """
    if resumed <= last_seen or resumed >= len(fastest):
        return False
    window = fastest[resumed:resumed + max(1, int(look_seconds * fps))]
    return bool(window.size) and float(np.nanmax(window)) > REST_SPEED_MS


def stitch_across_cuts(shot, positions, live, fps, at_rest=None, fastest=None,
                       max_gap_seconds=3.0, settle_seconds=MIN_REST_SECONDS):
    """Extend a play that was cut off, across the gap, while it is still running.

    Returns the new end frame, or the old one when the play cannot be carried
    on. The gap allowed is short on purpose: over a longer one the table could
    come back to the *next* shot already rolling, and stitching would join two
    plays into one.
    """
    live = np.asarray(live, dtype=bool)
    if fastest is None:
        _, _, _, fastest = _rest_and_motion(speeds(positions, fps), live)
    n = len(live)
    end = shot.end_frame
    max_gap = int(max_gap_seconds * fps)
    settle = max(1, int(settle_seconds * fps))

    while end < n - 1:
        resume = end + 1
        while resume < n and not live[resume]:
            resume += 1
        if resume >= n or resume - end > max_gap:
            break
        if not can_continue(fastest, end, resume, fps):
            break
        cursor, settled = resume, 0
        while cursor < n and live[cursor]:
            if at_rest is not None and at_rest[cursor]:
                settled += 1
                if settled >= settle:
                    return cursor
            else:
                settled = 0
            cursor += 1
        end = cursor - 1
    return end


def drop_unreachable_points(positions, fps, max_speed_ms=MAX_CUE_SPEED_MS, margin=1.4,
                            agreement=3, agreement_mm=80.0):
    """Discard detections no ball could have reached, leaving a gap instead.

    At 60 fps a ball at the fastest a cue can send it moves about 150 mm
    between frames. Steps far beyond that are the detector jumping to something
    else - a reflection, the other cue ball, a hand - and they are not harmless
    noise: one play's cue ball accumulated 141 m of travel and appeared to
    strike the right cushion forty-one times.

    A rejected point is dropped, not interpolated; the ball's position in that
    frame is simply unknown. When several consecutive readings agree with each
    other somewhere unreachable, it is the anchor that was wrong, so the track
    re-anchors there rather than rejecting the rest of the play.
    """
    limit = max_speed_ms * 1000.0 / fps * margin
    cleaned = {}
    for colour, xy in positions.items():
        out = np.array(xy, dtype=float, copy=True)
        finite = np.flatnonzero(np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]))
        if len(finite) < 2:
            cleaned[colour] = out
            continue
        anchor = finite[0]
        position = 1
        while position < len(finite):
            index = finite[position]
            gap = max(1, index - anchor)
            if float(np.linalg.norm(out[index] - out[anchor])) <= limit * gap:
                anchor = index
                position += 1
                continue
            following = finite[position:position + agreement]
            consistent = (
                len(following) >= agreement
                and np.all(np.linalg.norm(out[following] - out[index], axis=1) <= agreement_mm)
            )
            if consistent:
                anchor = index  # the anchor was the outlier, not this
                position += 1
                continue
            out[index] = np.nan
            position += 1
        cleaned[colour] = out
    return cleaned
