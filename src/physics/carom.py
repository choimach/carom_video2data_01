"""
Deciding from the trajectory whether a shot scored.

A three-cushion point is a geometric fact, not a timing one: the cue ball must
reach the second object ball having touched at least three cushions on the way.
"At least" matters - four, five or seven cushions score exactly the same, and a
grand tour is not a different kind of result.

Reading it off the trajectory rather than off the scoreboard removes the one
thing that made labelling unreliable. The board moves anywhere from fourteen
seconds before a play ends to forty-four seconds after it, and it does not move
at all while a replay is on screen - which turned replayed points into misses.
The path itself carries no such ambiguity, and it says the same thing whether
the play is live or being shown again.
"""

import numpy as np

from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM

BALL_DIAMETER_MM = 61.5
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0
REQUIRED_CUSHIONS = 3

# Centre-to-centre distance counts as a contact up to this much beyond a ball's
# width, covering the couple of millimetres of centroid error and the frame or
# two either side of the touch that 60 fps leaves unsampled.
CONTACT_TOLERANCE_MM = 12.0
# A cushion is touched when the centre comes within a radius of the nose line.
CUSHION_TOLERANCE_MM = 12.0


class Event:
    """One thing that happened to the cue ball."""

    def __init__(self, frame, kind, detail):
        self.frame = frame
        self.kind = kind  # "cushion" or "ball"
        self.detail = detail  # rail name, or the colour of the ball touched

    def __repr__(self):
        return f"<{self.kind}:{self.detail}@{self.frame}>"


def _runs(flags):
    """Start and end index of each contiguous True run."""
    out, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def cushion_events(cue_xy, length_mm=TABLE_LENGTH_MM, width_mm=TABLE_WIDTH_MM,
                   tolerance_mm=CUSHION_TOLERANCE_MM):
    """Cushion contacts, one per touch.

    A ball running along a rail touches it once, not once per frame, so
    contiguous frames against the same rail collapse into a single event. The
    two rails of a corner stay separate events, which is what the rules count.
    """
    events = []
    limits = ((0, length_mm, "left", "right"), (1, width_mm, "top", "bottom"))
    for axis, limit, low_name, high_name in limits:
        coordinate = cue_xy[:, axis]
        seen = np.isfinite(coordinate)
        for name, near in (
            (low_name, seen & (coordinate <= BALL_RADIUS_MM + tolerance_mm)),
            (high_name, seen & (coordinate >= limit - BALL_RADIUS_MM - tolerance_mm)),
        ):
            for start, end in _runs(near):
                events.append(Event((start + end) // 2, "cushion", name))
    return events


def ball_events(cue_xy, others, tolerance_mm=CONTACT_TOLERANCE_MM):
    """Contacts between the cue ball and each object ball, one per touch."""
    events = []
    for colour, xy in others.items():
        distance = np.linalg.norm(cue_xy - xy, axis=1)
        touching = np.isfinite(distance) & (distance <= BALL_DIAMETER_MM + tolerance_mm)
        for start, end in _runs(touching):
            events.append(Event((start + end) // 2, "ball", colour))
    return events


def shot_events(cue_xy, others):
    """Everything the cue ball touched, in order."""
    return sorted(cushion_events(cue_xy) + ball_events(cue_xy, others), key=lambda e: e.frame)


def judge_carom(events, required_cushions=REQUIRED_CUSHIONS):
    """Did this shot score?

    Returns (scored, details). The count is of cushions touched before the cue
    ball reaches the *second* object ball - the other colour. Touching the first
    object ball again does not make a second object ball, and cushions taken
    after the second contact are irrelevant.
    """
    balls = [e for e in events if e.kind == "ball"]
    if not balls:
        return False, {"reason": "the cue ball touched no object ball", "cushions": 0}

    first = balls[0]
    second = next((e for e in balls if e.detail != first.detail), None)
    if second is None:
        return False, {
            "reason": f"only {first.detail} was touched; the second object ball was missed",
            "cushions": sum(1 for e in events if e.kind == "cushion"),
            "first_ball": first.detail,
        }

    cushions = [e for e in events if e.kind == "cushion" and e.frame < second.frame]
    scored = len(cushions) >= required_cushions
    return scored, {
        "reason": (
            f"{len(cushions)} cushions before {second.detail}"
            if scored
            else f"only {len(cushions)} cushions before {second.detail}; {required_cushions} required"
        ),
        "cushions": len(cushions),
        "first_ball": first.detail,
        "second_ball": second.detail,
        "rails": [e.detail for e in cushions],
    }


def judge_shot(shot, positions, lookahead=0, limit=None):
    """Judge one segmented play straight from its tracked trajectory.

    `lookahead` frames past the play's recorded end are included, capped at
    `limit` (normally the next play's start). The segmenter ends a play once
    the balls read as at rest, but "at rest" is a speed threshold, and a cue
    ball can still be creeping the last few centimetres to the second object
    ball when it trips - four confirmed points on the labelled match landed in
    the two seconds after the play had officially ended.
    """
    end = shot.end_frame + 1 + lookahead
    if limit is not None:
        end = min(end, limit)
    window = slice(shot.start_frame, end)
    cue_xy = positions[shot.cue_ball][window]
    others = {c: xy[window] for c, xy in positions.items() if c != shot.cue_ball}
    events = shot_events_by_motion(cue_xy, others)
    scored, details = judge_carom(events)
    details["events"] = events
    return scored, details


# An object ball departs its resting place only when something hits it, and
# that departure is visible long after the frame of contact - unlike the
# contact itself, which happens when the cue ball is at its fastest and its
# blurred centroid is least trustworthy. Measured on confirmed successes, the
# first object ball's closest approach reads 63 mm at the median but 81 mm on a
# shot struck hard, against a true 61.5: too far for any usable threshold.
# 5 mm, not 25: a cue ball arriving after five or six cushions barely nudges the
# object ball, and a threshold set for a firm hit misses exactly the shots the
# players are proudest of. Measured against the hand-labelled match, lowering it
# recovered five successes and cost no false ones.
DEPARTURE_MM = 5.0
ATTRIBUTION_MM = 250.0


def _departures(xy, rest_window=12, threshold_mm=None, confirmations=3, settle_frames=5):
    """Where a ball left its resting place, and where that place was.

    Returns (frame, resting_position) pairs. The resting position is what the
    contact must be attributed to: by the time a firmly struck ball is first
    seen to have moved, it may already be far from whatever hit it.

    After a departure the ball is simply given a while to get clear before a new
    resting place is taken. Requiring it to be seen settled first looked more
    principled and cost a quarter of the accuracy on real shots - a ball that is
    struck again while still rolling never settles, and its second contact
    disappeared.
    """
    threshold_mm = DEPARTURE_MM if threshold_mm is None else threshold_mm
    out = []
    resting = None
    index, n = 0, len(xy)
    while index < n:
        point = xy[index]
        if not np.isfinite(point).all():
            index += 1
            continue
        if resting is None:
            window = xy[max(0, index - rest_window):index + 1]
            window = window[np.isfinite(window[:, 0])]
            resting = np.median(window, axis=0) if len(window) else point
            index += 1
            continue
        if float(np.linalg.norm(point - resting)) > threshold_mm:
            ahead = xy[index:index + confirmations + 1]
            ahead = ahead[np.isfinite(ahead[:, 0])]
            if len(ahead) >= confirmations and np.all(
                np.linalg.norm(ahead - resting, axis=1) > threshold_mm
            ):
                out.append((index, resting))
                resting = None
                index += settle_frames  # let it get clear before looking again
                continue
        index += 1
    return out


def _nearest_approach(xy, point, frame, before=6, after=3):
    """Closest a ball came to `point` in the frames around `frame`, ignoring gaps."""
    segment = xy[max(0, frame - before):frame + after]
    segment = segment[np.isfinite(segment[:, 0]) & np.isfinite(segment[:, 1])]
    if not len(segment):
        return np.inf
    return float(np.linalg.norm(segment - point, axis=1).min())


def contact_events(cue_xy, others, attribution_mm=None, departure_mm=None,
                   touch_mm=BALL_DIAMETER_MM, merge_frames=15):
    """Cue-ball contacts, from the object ball moving or from the balls touching.

    Neither signal is enough alone, and they fail in opposite places. Motion
    catches a firm hit but not a thin one: a cue ball arriving after five or six
    cushions has little left, and the object ball shifts 5-30 mm - over in a few
    frames. Distance catches the thin hit, where both balls are nearly still and
    the centres read true, but not the firm one: the first object ball is struck
    when the cue ball is fastest and its blurred centroid lags, so the closest
    approach can read 80 mm against a true 61.5.

    Taking either as evidence covers both. A ball that starts moving was hit by
    whatever was closest at that moment; checking which ball that was keeps a
    kiss - the first object ball driving the second - from being read as a
    carom the cue ball never made.
    """
    attribution_mm = ATTRIBUTION_MM if attribution_mm is None else attribution_mm
    events = []
    for colour, xy in others.items():
        found = list(_departures(xy, threshold_mm=departure_mm))
        # Centres closer than a ball's width can only mean the two touched.
        distance = np.linalg.norm(cue_xy - xy, axis=1)
        for start, end in _runs(np.isfinite(distance) & (distance <= touch_mm)):
            middle = (start + end) // 2
            if all(abs(middle - f) > 4 for f, _ in found):
                found.append((middle, xy[middle]))
        for frame, resting in sorted(found):
            # Measure against where the ball was sitting when it was hit, not
            # where it has got to by the time the move is confirmed - and over
            # a few frames either side, not the one frame of departure. The
            # first object ball is struck when the cue ball is at its fastest,
            # which is exactly when the detector is likeliest to have lost it
            # or to place its blurred centroid a ball's width behind. Judged on
            # that single frame, the first contact of four confirmed points on
            # the labelled match went to nobody, and their second contact was
            # then read as the first.
            cue_distance = _nearest_approach(cue_xy, resting, frame)
            rivals = [
                _nearest_approach(other, resting, frame)
                for name, other in others.items()
                if name != colour
            ]
            rivals = [r for r in rivals if np.isfinite(r)]
            if cue_distance > attribution_mm:
                continue
            if rivals and min(rivals) < cue_distance:
                continue  # another ball was nearer: this is a kiss, not a carom
            # One touch, one event: motion and distance often both witness the
            # same contact, and a nudged ball can read as leaving twice.
            if events and events[-1].detail == colour and frame - events[-1].frame <= merge_frames:
                continue
            events.append(Event(frame, "ball", colour))
    return sorted(events, key=lambda e: e.frame)


def shot_events_by_motion(cue_xy, others):
    """Everything the cue ball touched, ordered, with contacts read from motion."""
    return sorted(cushion_events(cue_xy) + contact_events(cue_xy, others), key=lambda e: e.frame)
