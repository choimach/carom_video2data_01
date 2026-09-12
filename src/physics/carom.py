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


def judge_shot(shot, positions):
    """Judge one segmented play straight from its tracked trajectory."""
    window = slice(shot.start_frame, shot.end_frame + 1)
    cue_xy = positions[shot.cue_ball][window]
    others = {c: xy[window] for c, xy in positions.items() if c != shot.cue_ball}
    events = shot_events(cue_xy, others)
    scored, details = judge_carom(events)
    details["events"] = events
    return scored, details
