"""What the player did to the cue ball, read back from where the balls went.

Three numbers describe a stroke beyond its speed and direction: how much of the
object ball was covered, how much top or bottom spin was on the cue ball, and
how much side. None of them is visible - the balls carry no markings the camera
can resolve, and at four metres a second they are a blur. All three have to be
inferred from what the balls did afterwards.

The standard billiard approximation does the work: two equal balls in a
frictionless contact exchange momentum along the line joining their centres, so
the object ball leaves along that line and the cue ball leaves perpendicular to
it. Everything the cue ball does other than leave perpendicular is spin.
"""

import numpy as np

BALL_DIAMETER_MM = 61.5

# Rail normals in table millimetres, pointing into the playing surface.
RAIL_NORMALS = {
    "top": (0.0, 1.0),
    "bottom": (0.0, -1.0),
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
}

# Frames either side of a contact are the ones to distrust: the ball is at its
# fastest, its blurred centroid is least reliable, and the two balls overlap.
CONTACT_GAP = 2
TRAVEL_SPAN = 6
MIN_SPEED_MM_PER_FRAME = 1.5
# A stroke's cue ball is travelling far faster than this when it arrives. Below
# it, the direction of a seven-frame step is mostly centroid noise - and a cue
# ball measured at 2 mm a frame "launching" an object ball at 51 is not a thin
# hit, it is a lost track.
MIN_STROKE_SPEED_MM_PER_FRAME = 5.0
# Equal masses cannot send the struck ball away faster than the striking one
# arrived. A little over 1 is measurement error; well over it is a mistake.
MAX_SPEED_RATIO = 1.1
# The cue ball keeps the part of its speed the object ball did not take, which
# is sin of the same angle the thickness came from - so the two are a check on
# each other. Measured across 258 contacts they agree to 0.03 wherever the
# tracks are sound. A quarter is far outside that, and what it catches is the
# near-full hit, where the balls sit almost in line and a detector that swaps
# them leaves the cue ball holding the object ball's speed.
MAX_RESIDUAL_DISAGREEMENT = 0.25


def _cross(a, b):
    """The scalar cross product of two plane vectors: |a||b| sin of the angle."""
    return float(a[0] * b[1] - a[1] * b[0])


def _travel(xy, first, last):
    """Unit direction of travel and speed in mm per frame over a span of frames.

    Returns (None, 0.0) when the ball was not seen enough, or did not move far
    enough for its direction to mean anything.
    """
    first, last = max(0, int(first)), min(len(xy) - 1, int(last))
    if last <= first:
        return None, 0.0
    window = xy[first:last + 1]
    seen = np.flatnonzero(np.isfinite(window).all(axis=1))
    if len(seen) < 2:
        return None, 0.0
    step = window[seen[-1]] - window[seen[0]]
    distance = float(np.hypot(*step))
    frames = int(seen[-1] - seen[0])
    if distance < 1e-6 or frames == 0:
        return None, 0.0
    speed = distance / frames
    if speed < MIN_SPEED_MM_PER_FRAME:
        return None, speed
    return step / distance, speed


def refine_contact(cue_xy, object_xy, contact_frame, search=2):
    """The frame at which the two centres came closest, near the detected one.

    Contact is detected from the object ball leaving, which is a frame or two
    after the balls actually touched. One frame matters here: a cue ball at four
    metres a second covers more than a ball width between frames.
    """
    best, at = None, int(contact_frame)
    for frame in range(max(0, at - search), min(len(cue_xy), at + search + 1)):
        a, b = cue_xy[frame], object_xy[frame]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        separation = float(np.hypot(*(a - b)))
        if best is None or separation < best[0]:
            best = (separation, frame)
    return at if best is None else best[1]


def thickness(cue_xy, object_xy, contact_frame, gap=CONTACT_GAP, span=TRAVEL_SPAN):
    """How much of the object ball the cue ball covered, from 0 to 1.

    1 is a full ball - centres in line - and 0 is a miss. The object ball leaves
    along the line of centres, so the angle between where the cue ball was going
    and where the object ball went is the angle between the cue ball's path and
    that line. The centres are one ball apart at contact, which makes the
    perpendicular offset D*sin(angle), and the fraction of the ball covered
    1 - sin(angle).

    Reading the angle off the object ball's departure rather than off the two
    centres at the moment of contact is deliberate: contact is detected to
    within a frame or so, and a ball moving 250 mm in a frame is four ball
    widths out of place by then. Where the object ball went is not in doubt.
    """
    incoming, cue_speed = _travel(cue_xy, contact_frame - gap - span, contact_frame - gap)
    departure, object_speed = _travel(object_xy, contact_frame + gap, contact_frame + gap + span)
    if incoming is None or departure is None:
        return None
    if cue_speed < MIN_STROKE_SPEED_MM_PER_FRAME:
        return None
    if object_speed > cue_speed * MAX_SPEED_RATIO:
        return None
    sine = abs(_cross(incoming, departure))

    # Cross-check against the cue ball's own residual speed, which the same
    # angle predicts independently of anything the object ball did.
    _, cue_after = _travel(cue_xy, contact_frame + gap, contact_frame + gap + span)
    if cue_after > 0 and abs(cue_after / cue_speed - sine) > MAX_RESIDUAL_DISAGREEMENT:
        return None
    return float(np.clip(1.0 - sine, 0.0, 1.0))


def follow_draw(cue_xy, object_xy, contact_frame, gap=CONTACT_GAP, span=TRAVEL_SPAN):
    """Top or bottom spin on the cue ball at contact, from -1 to 1.

    A cue ball with no spin of its own leaves a cut at a right angle to the line
    of centres - the tangent line. Top spin carries it forward of the tangent,
    bottom spin pulls it back. What is returned is how much of the cue ball's
    departure lies along the line of centres rather than across it: positive for
    follow, negative for draw, around zero for a stun.

    This says nothing on a full ball, where the cue ball stops and has no
    direction to read, and little on a very thin one, where the tangent and the
    original path are nearly the same line. Both come back as None.

    UNVALIDATED, unlike the other two here. The independent test - follow should
    carry the cue ball further after the contact than draw does - came back flat
    (correlation -0.01 over 188 contacts, draw 16.8 frame-lengths against follow
    18.1). That is not evidence the number is wrong: in three-cushion the cue
    ball meets a cushion within half a second of almost every contact, and where
    it goes after that says more about the cushion than about the spin. It is
    evidence that nothing here has confirmed the number either. There is also a
    known bias - thickness() rejects contacts where the cue ball kept more speed
    than its angle explains, which is exactly what strong spin does - so treat
    the spread of these as a floor, not a measurement.
    """
    centres, _ = _travel(object_xy, contact_frame + gap, contact_frame + gap + span)
    cue_out, cue_speed = _travel(cue_xy, contact_frame + gap, contact_frame + gap + span)
    incoming, cue_in = _travel(cue_xy, contact_frame - gap - span, contact_frame - gap)
    if centres is None or cue_out is None or incoming is None:
        return None
    if cue_in < MIN_STROKE_SPEED_MM_PER_FRAME:
        return None
    # On a thin hit the object ball barely deflects and the tangent sits almost
    # on the cue ball's own path, so any noise in either reads as huge spin.
    if abs(float(np.dot(incoming, centres))) < 0.25:
        return None
    return float(np.clip(np.dot(cue_out, centres), -1.0, 1.0))


def sidespin(cue_xy, cushion_frame, rail, gap=CONTACT_GAP, span=TRAVEL_SPAN):
    """Side spin, as the degrees by which a rail rebound missed the mirror angle.

    A ball with no side on it comes off a cushion at the angle it went in, near
    enough, losing a little of its speed along the rail to friction. Running
    side sends it off wider than it arrived and reverse side narrower, so the
    difference between the two angles is a measure of the side that was on it.

    Positive is wider than the mirror angle - running side. The rail is named
    rather than derived, because which rail was struck is already known.

    Checked against the speed the ball keeps along the rail, which this
    measurement does not look at: over 338 cushions, rebounds this calls reverse
    keep 0.35 of it, ones it calls neutral 0.92, and ones it calls running 1.17.
    Grip and kill are what side does to a cushion, and the three separate
    cleanly, so this is reading side and not noise.
    """
    normal = RAIL_NORMALS.get(rail)
    if normal is None:
        return None
    normal = np.array(normal, dtype=float)
    before, _ = _travel(cue_xy, cushion_frame - gap - span, cushion_frame - gap)
    after, _ = _travel(cue_xy, cushion_frame + gap, cushion_frame + gap + span)
    if before is None or after is None:
        return None
    # Angle from the rail's normal, on each side of the bounce.
    into = np.degrees(np.arctan2(abs(_cross(before, normal)), abs(float(np.dot(before, normal)))))
    out = np.degrees(np.arctan2(abs(_cross(after, normal)), abs(float(np.dot(after, normal)))))
    return float(out - into)


def stroke_of(shot, positions, events):
    """Thickness, follow/draw and side for one play, or None where unreadable.

    `events` is what judge_shot recorded: the cue ball's cushions and contacts,
    in order. The first ball contact is the stroke's own - later ones have the
    object ball's own spin in them - and the first cushion is the one that still
    carries what the player put on, before it is worn off and passed around.

    judge_shot works on a window cut from the match, so the frames on those
    events count from the play's own start, not from the match's. Reading them
    against the full tracks samples some arbitrary earlier moment instead, which
    is a mistake that does not announce itself: it returns numbers.
    """
    cue_xy = positions[shot.cue_ball]
    base = shot.start_frame
    out = {"thickness": None, "spin_y": None, "spin_x": None, "spin_rail": None}

    first_ball = next((e for e in events if e.kind == "ball"), None)
    if first_ball is not None and first_ball.detail in positions:
        object_xy = positions[first_ball.detail]
        at = refine_contact(cue_xy, object_xy, base + first_ball.frame)
        out["thickness"] = thickness(cue_xy, object_xy, at)
        out["spin_y"] = follow_draw(cue_xy, object_xy, at)

    first_cushion = next((e for e in events if e.kind == "cushion"), None)
    if first_cushion is not None:
        out["spin_x"] = sidespin(cue_xy, base + first_cushion.frame, first_cushion.detail)
        out["spin_rail"] = first_cushion.detail
    return out
