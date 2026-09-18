"""Find a shot that scores from a layout, and say how much room it has.

This is what the assistant needs and the dataset alone cannot give: a path
through *these* three balls, not a path a professional took through three balls
that stood somewhere similar.

The method is a sweep, not a solver. For every opening direction at a few
speeds and a little side, play the shot in the simulator and keep the ones that
carom. A line that scores on its own is worth less than one with scoring lines
either side of it - a player aims with error, and the wide window is the shot
worth recommending - so candidates are ranked by how many degrees of
neighbouring directions also score.

⚠️ The simulator this rests on is 80 mm out over a path and carries no spin as
a state, and scoring windows measure a quarter of a degree. So a line from here
is a drawing of the route, not a prescription: it shows which way the shot goes
round, and it will not be exact.

    ~/.venvs/carom/bin/python tools/solve_shot.py
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate  # noqa: E402

SWEEP_STEP_DEGREES = 0.5
SPEEDS_MM_S = (1600.0, 2200.0, 2800.0, 3400.0)
SIDES = (-6.0, 0.0, 6.0)


def attempts(layout, cue, step=SWEEP_STEP_DEGREES, speeds=SPEEDS_MM_S, sides=SIDES):
    """Every sweep result, scoring or not, as (degrees, speed, side, shot)."""
    out = []
    for degrees in np.arange(0.0, 360.0, step):
        heading = np.radians(degrees)
        direction = np.array([np.cos(heading), np.sin(heading)])
        for speed in speeds:
            for side in sides:
                shot = simulate(layout, cue, direction * speed, side_degrees=side)
                out.append((float(degrees), speed, side, shot))
    return out


def widest(layout, cue, **kwargs):
    """The scoring shots, best first, by how much aiming room each has.

    Room is counted in degrees of neighbouring directions that also score at
    the same speed and side - the thing a player actually has to hit.
    """
    tries = attempts(layout, cue, **kwargs)
    step = kwargs.get("step", SWEEP_STEP_DEGREES)
    scoring = {}
    for degrees, speed, side, shot in tries:
        if shot.scored(cue):
            scoring.setdefault((speed, side), set()).add(round(degrees, 3))

    best = []
    for degrees, speed, side, shot in tries:
        if not shot.scored(cue):
            continue
        window = scoring[(speed, side)]
        room = step
        for direction in (step, -step):
            at = round(degrees + direction, 3)
            while at % 360 in window or at in window:
                room += step
                at = round(at + direction, 3)
        best.append({"degrees": degrees, "speed": speed, "side": side,
                     "room_degrees": room, "shot": shot})
    best.sort(key=lambda row: (-row["room_degrees"], row["speed"]))
    return best


def path_of(shot, cue, fps=60.0, points=40):
    path = np.array(shot.paths[cue])
    if len(path) <= points:
        return [[int(round(x)), int(round(y))] for x, y in path]
    keep = np.linspace(0, len(path) - 1, points).astype(int)
    return [[int(round(path[i][0])), int(round(path[i][1]))] for i in keep]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", help="JSON: {\"white\": [x, y], ...}")
    parser.add_argument("--cue", default="white")
    parser.add_argument("--step", type=float, default=SWEEP_STEP_DEGREES)
    args = parser.parse_args(argv)

    layout = (json.loads(args.layout) if args.layout else
              {"white": [700.0, 950.0], "yellow": [1150.0, 520.0], "red": [2250.0, 430.0]})
    found = widest(layout, args.cue, step=args.step)
    print(f"{len(found)} scoring lines found")
    for row in found[:5]:
        shot = row["shot"]
        print(f"  {row['degrees']:6.1f}° at {row['speed']:.0f} mm/s, side {row['side']:+.0f}: "
              f"room {row['room_degrees']:.1f}°, {len(shot.cushions)} cushions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
