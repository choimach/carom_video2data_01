"""Is the simulator wrong about the table, or about where the ball was sent?

Both would look the same from outside: a simulated path that leaves the real
one. They are not the same problem. If the physics is right and only the
opening direction is out - and this pipeline measures that to about 4 degrees -
then searching a few degrees either side should find a line whose whole
simulated path lies on the one the camera saw, and the simulator is worth
improving inputs for. If no line in that range fits, the physics is wrong and
better inputs would not rescue it.

So: take plays that scored, search the opening around what was measured, and
keep the best fit to the observed trajectory.

    ~/.venvs/carom/bin/python tools/fit_openings.py --plays 20
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src.physics.simulator import simulate  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402
from validate_simulator import opening  # noqa: E402


def mismatch(observed, simulated, frames):
    """Mean distance between two paths over the frames both cover."""
    span = min(len(observed), len(simulated), frames)
    if span < 10:
        return None
    gap = np.linalg.norm(observed[:span] - simulated[:span], axis=1)
    gap = gap[np.isfinite(gap)]
    return float(np.mean(gap)) if len(gap) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the opening to the observed path")
    parser.add_argument("--plays", type=int, default=20)
    parser.add_argument("--angle", type=float, default=6.0, help="degrees either side")
    parser.add_argument("--angle-step", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=0.25, help="fraction either side")
    parser.add_argument("--frames", type=int, default=120, help="two seconds at 60 fps")
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz"))):
        data = load_scan(path)
        result = analyse(data)
        fps = result["fps"]
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None) or not shot.success:
                    continue
                start = opening(shot, data["positions"], fps)
                if start is None:
                    continue
                layout, cue, velocity, frame = start
                observed = data["positions"][cue][frame:shot.end_frame + 1]

                measured_angle = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
                measured_speed = float(np.hypot(*velocity))
                as_measured = mismatch(observed,
                                       simulate(layout, cue, velocity, fps=fps).paths[cue],
                                       args.frames)
                if as_measured is None:
                    continue

                best = (as_measured, 0.0, 1.0, False)
                offsets = np.arange(-args.angle, args.angle + 1e-9, args.angle_step)
                for turn in offsets:
                    for scale in (1 - args.speed, 1.0, 1 + args.speed):
                        radians = np.radians(measured_angle + turn)
                        speed = measured_speed * scale
                        played = simulate(layout, cue,
                                          (speed * np.cos(radians), speed * np.sin(radians)),
                                          fps=fps)
                        error = mismatch(observed, played.paths[cue], args.frames)
                        if error is not None and error < best[0]:
                            best = (error, float(turn), float(scale), played.scored(cue))
                rows.append((as_measured, best[0], best[1], best[2], best[3]))
                print(f"  as measured {as_measured:6.0f} mm -> best fit {best[0]:6.0f} mm "
                      f"at {best[1]:+5.1f} deg, speed x{best[2]:.2f}"
                      f"{'  (and it scores)' if best[3] else ''}", flush=True)
                if len(rows) >= args.plays:
                    break
            if len(rows) >= args.plays:
                break
        if len(rows) >= args.plays:
            break

    if not rows:
        print("nothing to fit")
        return 1
    values = np.array([[r[0], r[1], r[2], r[3], float(r[4])] for r in rows])
    print(f"\n{len(rows)} scoring plays, fitted over the first {args.frames} frames")
    print(f"  path error as measured    median {np.median(values[:, 0]):5.0f} mm")
    print(f"  path error at best fit    median {np.median(values[:, 1]):5.0f} mm")
    print(f"  turn needed               median {np.median(np.abs(values[:, 2])):5.1f} deg"
          f"   (searched +/-{args.angle:.0f})")
    print(f"  best fit also scores      {values[:, 4].mean() * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
