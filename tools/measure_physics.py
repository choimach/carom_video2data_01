"""Measure the constants a simulator needs, from the plays already captured.

Four numbers stand between a tracked trajectory and one that can be predicted:
how fast a rolling ball loses speed to the cloth, how much of its speed a
cushion gives back along the rail's normal and along the rail itself, and how
much one ball hands to another. All four are in the data; none of them has to
be looked up.

Speeds are worked in millimetres per frame and converted once at the end, and
every stretch used is checked for the thing that would spoil it: a ball that is
not being seen, a path that is bending, an event inside the window.

    ~/.venvs/carom/bin/python tools/measure_physics.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.stroke import RAIL_NORMALS, _cross, _travel, refine_contact  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402

BALLS = ("white", "yellow", "red")
STRAIGHT_WINDOW = 20        # frames; a third of a second at 60 fps
MIN_SPEED = 3.0             # mm per frame - below this the direction is noise
MAX_BEND_DEG = 1.5          # a free-rolling ball travels in a straight line


def plays(limit=None):
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:limit]:
        data = load_scan(path)
        result = analyse(data)
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None):
                    continue
                yield data, result, shot


def free_runs(shot, positions, colour, min_frames=30):
    """Stretches where a ball rolled with nothing happening to it.

    For the cue ball the event list says when something happened, so between
    two events it is free. The other balls have no event list and are taken
    whole; the straightness test then does the filtering.
    """
    xy = positions[colour]
    marks = [0]
    if colour == shot.cue_ball:
        marks += [int(e.frame) for e in ((shot.verdict or {}).get("events") or [])]
    marks.append(shot.end_frame - shot.start_frame)
    marks = sorted(set(marks))
    for begin, end in zip(marks, marks[1:]):
        # Keep clear of both events: the frames around one are the least
        # trustworthy in the play.
        first, last = shot.start_frame + begin + 4, shot.start_frame + end - 4
        if last - first < min_frames:
            continue
        piece = xy[first:last + 1]
        if len(piece) and np.isfinite(piece).all():
            yield piece


def run_deceleration(piece, max_bend_deg=MAX_BEND_DEG):
    """Deceleration over one free run, in mm per frame squared.

    Fitted to the distance travelled, not to differences between consecutive
    frames. A ball moving 20 mm a frame, against a centroid good to a
    millimetre or two, gives per-frame speeds that scatter by a tenth - enough
    for a rolling ball to appear to speed up - while the same noise barely
    moves a parabola fitted through sixty positions.
    """
    step = piece[-1] - piece[0]
    span = float(np.hypot(*step))
    if span < 1.0:
        return None, 0.0
    direction = step / span
    offsets = piece - piece[0]
    along = offsets @ direction
    across = np.abs(offsets @ np.array([-direction[1], direction[0]]))
    if across.max() > span * np.tan(np.radians(max_bend_deg)):
        return None, 0.0            # the path bends: not a free roll
    t = np.arange(len(piece), dtype=float)
    quad, linear, _const = np.polyfit(t, along, 2)
    if linear < MIN_SPEED or quad > 0:
        return None, float(linear)
    return -2.0 * float(quad), float(linear)


def cloth_deceleration(sample=None):
    """Deceleration of a freely rolling ball, with the speed it was doing."""
    rows, fps_seen = [], []
    for data, result, shot in plays(sample):
        fps_seen.append(result["fps"])
        for colour in BALLS:
            for piece in free_runs(shot, data["positions"], colour):
                decel, speed = run_deceleration(piece)
                if decel is not None:
                    rows.append((decel, speed))
    fps = float(np.median(fps_seen)) if fps_seen else 60.0
    return (np.array(rows) if rows else np.empty((0, 2))), fps


def cushion_bounces(sample=None):
    """Normal and tangential speed ratios either side of a rail contact."""
    rows = []
    for data, result, shot in plays(sample):
        cue = data["positions"][shot.cue_ball]
        for event in (shot.verdict or {}).get("events") or []:
            if event.kind != "cushion" or event.detail not in RAIL_NORMALS:
                continue
            at = shot.start_frame + event.frame
            normal = np.array(RAIL_NORMALS[event.detail], dtype=float)
            before, v0 = _travel(cue, at - 8, at - 2)
            after, v1 = _travel(cue, at + 2, at + 8)
            if before is None or after is None or v0 < MIN_SPEED:
                continue
            n0, n1 = abs(float(np.dot(before, normal))) * v0, abs(float(np.dot(after, normal))) * v1
            t0, t1 = abs(_cross(before, normal)) * v0, abs(_cross(after, normal)) * v1
            if n0 < 0.5 or t0 < 0.5:
                continue
            angle = np.degrees(np.arctan2(t0, n0))
            rows.append((n1 / n0, t1 / t0, angle, v0))
    return np.array(rows) if rows else np.empty((0, 4))


# Windows after a contact, as (first, last) frames, for watching a struck ball
# settle. Their midpoints are what the extrapolation is fitted against.
AFTER_WINDOWS = ((1, 4), (2, 6), (4, 9), (8, 14))


def ball_on_ball(sample=None):
    """How much speed one ball hands another, measured back to the contact.

    A struck ball leaves with no spin of its own, so it slides before it rolls,
    and a ball that starts sliding settles at five sevenths of the speed it
    began with. Measuring a few frames after the contact therefore catches it
    part way down that slope, not at the top: the same collisions read 0.90 at
    three frames, 0.83 at six and 0.66 at eighteen, passing straight through
    5/7 on the way. Quoting any one of those as the restitution would be
    quoting the cloth. Fitting the early windows and running the line back to
    the contact gives the number the collision actually produced.
    """
    rows = []
    for data, result, shot in plays(sample):
        events = (shot.verdict or {}).get("events") or []
        first = next((e for e in events if e.kind == "ball"), None)
        if first is None or first.detail not in data["positions"]:
            continue
        cue = data["positions"][shot.cue_ball]
        obj = data["positions"][first.detail]
        at = refine_contact(cue, obj, shot.start_frame + first.frame)
        incoming, v_in = _travel(cue, at - 8, at - 2)
        departure, _ = _travel(obj, at + 2, at + 8)
        if incoming is None or departure is None or v_in < 5.0:
            continue
        along_in = abs(float(np.dot(incoming, departure))) * v_in
        if along_in < 1.0:
            continue
        ratios = []
        for begin, end in AFTER_WINDOWS:
            _, speed = _travel(obj, at + begin, at + end)
            ratios.append(speed / along_in if speed > 0 else np.nan)
        rows.append([float(np.dot(incoming, departure))] + ratios)
    return np.array(rows) if rows else np.empty((0, 1 + len(AFTER_WINDOWS)))


def restitution_at_contact(rows):
    """Run the measured windows back to the moment of contact."""
    if len(rows) == 0:
        return None, []
    centres = np.array([(a + b) / 2.0 for a, b in AFTER_WINDOWS], dtype=float)
    medians = np.array([float(np.nanmedian(rows[:, 1 + i])) for i in range(len(AFTER_WINDOWS))])
    slope, intercept = np.polyfit(centres, medians, 1)
    return float(intercept), list(zip(centres, medians))


def quote(name, values, unit="", digits=3):
    if len(values) == 0:
        print(f"  {name:<34} no measurements")
        return
    median = float(np.median(values))
    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    print(f"  {name:<34} {median:.{digits}f}{unit}   (n={len(values)}, "
          f"middle half spans {spread:.{digits}f})")


def main(argv=None):
    sample = int(argv[0]) if argv else None

    print("cloth")
    runs, fps = cloth_deceleration(sample)
    if len(runs):
        quote("deceleration, mm/frame^2", runs[:, 0], "", 4)
        median = float(np.median(runs[:, 0]))
        print(f"  {'in metres per second squared':<34} {median * fps * fps / 1000:.3f} m/s^2"
              f"   ({median * fps * fps / 1000 / 9.81:.4f} g)")
        for lo, hi in zip([3, 8, 15, 25, 40], [8, 15, 25, 40, 200]):
            sel = (runs[:, 1] >= lo) & (runs[:, 1] < hi)
            if sel.sum() < 5:
                continue
            band = float(np.median(runs[sel, 0])) * fps * fps / 1000
            print(f"    at {lo * fps / 1000:.1f}-{hi * fps / 1000:.1f} m/s"
                  f"   n={int(sel.sum()):4d}   {band:.3f} m/s^2")

    print("\ncushion")
    bounces = cushion_bounces(sample)
    if len(bounces):
        quote("restitution across the rail", bounces[:, 0])
        quote("speed kept along the rail", bounces[:, 1])
        steep = bounces[bounces[:, 2] < 35]
        shallow = bounces[bounces[:, 2] > 55]
        quote("  restitution, square-on (<35 deg)", steep[:, 0] if len(steep) else np.array([]))
        quote("  restitution, glancing (>55 deg)", shallow[:, 0] if len(shallow) else np.array([]))

    print("\nball on ball")
    hits = ball_on_ball(sample)
    at_contact, curve = restitution_at_contact(hits)
    if at_contact is not None:
        for frames, value in curve:
            print(f"    {frames:4.1f} frames after   {value:.3f}")
        print(f"  {'run back to the contact':<34} {at_contact:.3f}   (n={len(hits)})")
        print(f"  {'a struck ball settles at 5/7':<34} 0.714   - passed through, not the collision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
