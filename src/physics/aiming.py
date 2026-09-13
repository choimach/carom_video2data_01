"""Which way to send the cue ball, and how likely each way is to score.

A three-cushion shot either scores or it does not, but a player does not choose
an exact angle - they choose an intention and deliver it with some error. So
the useful quantity is not whether one line scores, it is how much scoring room
there is around it. A line that scores but misses a degree either side is worth
less than one that scores across four degrees, and only the second is worth
aiming at.

That is computed by sweeping the angle finely enough that neighbouring lines
serve as each other's trials, and then asking how many of them score within the
spread a real stroke lands in. One sweep answers it for any amount of aiming
error, which is what makes this affordable: the alternative, repeating a noisy
shot at every angle, costs the same work many times over.

WHAT THIS CANNOT DO YET. Swept against fourteen shots professionals actually
scored, it puts the line they played at a probability of 0.00, with the nearest
line it thinks scores a median of 20 degrees away. It found scoring lines on
every one of those layouts - it simply put them somewhere else. So this is not
an oracle for whether a given shot goes in, and nothing below should be read as
one.

The error budget says why, and it is not mysterious. A cue ball reaching a
second object ball after three cushions has travelled some eight metres to a
corridor 123 mm wide, which is a tenth of a degree of tolerance at the far end
and a quarter of a degree at the near one once the cushions have multiplied it.
Measured on random layouts the scoring windows are 0.25 degrees across - at or
below the resolution they were swept at. The simulator lands 100 mm off at the
FIRST rail, about 4 degrees, before any of that amplification. It is an order
of magnitude short, and so is the 4 degrees this pipeline measures a real cue
ball's opening direction to, which means better physics alone would not fix it.

One suspicion checked and dismissed: the rebound curve is anchored at the origin
on no data, giving its first segment a slope of 2.5 that would multiply an error
by that much at every shallow bounce. Removing the segment changes neither the
window widths nor their number. Whatever is wrong is elsewhere.
"""

import numpy as np

from src.physics.simulator import simulate

# A quarter of a degree across a table length is 12 mm, comfortably finer than
# anything the aiming spread below cares about.
ANGLE_STEP_DEG = 0.25
# What a stroke actually lands within. Professionals are far better than the
# 4 degrees this pipeline measures a cue ball's opening direction to, and that
# figure is the camera's error and not theirs - so this is an assumption, and
# it is a parameter for that reason.
AIM_SPREAD_DEG = 0.8
DEFAULT_SPEEDS = (1600.0, 2200.0, 2800.0, 3600.0)


def sweep(layout, cue, speeds=DEFAULT_SPEEDS, angle_step_deg=ANGLE_STEP_DEG, **kwargs):
    """Fire the cue ball every way round the circle, at each speed.

    Returns (angles in degrees, {speed: boolean array of whether it scored}).
    """
    angles = np.arange(0.0, 360.0, angle_step_deg)
    radians = np.radians(angles)
    scored = {}
    for speed in speeds:
        outcomes = np.zeros(len(angles), dtype=bool)
        for index, angle in enumerate(radians):
            velocity = (speed * np.cos(angle), speed * np.sin(angle))
            outcomes[index] = simulate(layout, cue, velocity, **kwargs).scored(cue)
        scored[speed] = outcomes
    return angles, scored


def probability(outcomes, angle_step_deg=ANGLE_STEP_DEG, spread_deg=AIM_SPREAD_DEG):
    """How often a stroke aimed at each angle would score, given aiming error.

    The stroke is taken to land on a normal spread about where it was aimed, so
    this is that spread convolved with the outcomes. Wrapping is right and not a
    convenience: the angles run the whole way round, and 359.9 degrees is next
    to 0.1.
    """
    sigma = max(spread_deg, 1e-6) / angle_step_deg
    half = int(np.ceil(3 * sigma))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([outcomes[-half:], outcomes, outcomes[:half]]).astype(float)
    return np.convolve(padded, kernel, mode="valid")


def best_aim(layout, cue, speeds=DEFAULT_SPEEDS, spread_deg=AIM_SPREAD_DEG,
             angle_step_deg=ANGLE_STEP_DEG, **kwargs):
    """The angle and speed with the most scoring room around it.

    Returns a dict with the chosen line, how likely it is to score, and the
    whole map it was chosen from, so the second-best line is there to look at
    too - on many layouts there are several, and which one a player takes
    depends on where it leaves the balls.
    """
    angles, scored = sweep(layout, cue, speeds, angle_step_deg, **kwargs)
    maps = {speed: probability(outcomes, angle_step_deg, spread_deg)
            for speed, outcomes in scored.items()}

    best = None
    for speed, chances in maps.items():
        index = int(np.argmax(chances))
        if best is None or chances[index] > best["probability"]:
            best = {"angle_deg": float(angles[index]), "speed_mm_s": float(speed),
                    "probability": float(chances[index])}
    return {**best, "angles": angles, "outcomes": scored, "maps": maps,
            "scoring_lines": {s: int(o.sum()) for s, o in scored.items()}}


def windows(angles, chances, floor=0.5):
    """The stretches of angle where a shot is more likely than not to score.

    Reported as (start, end, best angle, best chance), in degrees. A layout with
    three of these has three answers, and a player picking between them is
    choosing on something this does not model - where the balls finish.
    """
    above = chances >= floor
    if not above.any():
        return []
    step = angles[1] - angles[0]
    found, start = [], None
    wrapped = np.concatenate([above, above[:1]])
    for index in range(len(above)):
        if wrapped[index] and start is None:
            start = index
        elif not wrapped[index] and start is not None:
            piece = chances[start:index]
            peak = start + int(np.argmax(piece))
            found.append((float(angles[start]), float(angles[index - 1] + step),
                          float(angles[peak]), float(chances[peak])))
            start = None
    if start is not None:
        piece = chances[start:]
        peak = start + int(np.argmax(piece))
        found.append((float(angles[start]), float(angles[-1] + step),
                      float(angles[peak]), float(chances[peak])))
    return found
