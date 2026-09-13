"""Does the aiming map agree with what the players did?

For plays that scored, work out the angle the cue ball was actually sent at,
then sweep every angle from the same layout and ask where that one falls. If
the map is worth anything the players' lines should sit on or beside a scoring
one - they are professionals and the shot went in.

This is the end-to-end test of everything measured so far: the cloth, both
cushion curves, the collision, and the carry. Nothing else would put a scoring
line where a professional aimed.

    ~/.venvs/carom/bin/python tools/check_aiming.py --plays 40
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.aiming import probability, sweep  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
from validate_simulator import opening  # noqa: E402


def nearest_scoring(angles, outcomes, aimed):
    """How far the played angle was from the closest line that scores."""
    if not outcomes.any():
        return None
    gap = np.abs((angles[outcomes] - aimed + 180.0) % 360.0 - 180.0)
    return float(gap.min())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the aiming map against real plays")
    parser.add_argument("--plays", type=int, default=30)
    parser.add_argument("--step", type=float, default=0.5, help="angle step, degrees")
    parser.add_argument("--spread", type=float, default=0.8, help="aiming error, degrees")
    args = parser.parse_args(argv)

    chosen = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz"))):
        data = load_scan(path)
        result = analyse(data)
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None) or not shot.success:
                    continue
                start = opening(shot, data["positions"], result["fps"])
                if start is None:
                    continue
                chosen.append(start)
                if len(chosen) >= args.plays:
                    break
            if len(chosen) >= args.plays:
                break
        if len(chosen) >= args.plays:
            break

    print(f"{len(chosen)} scoring plays\n")
    print(f"{'played':>8} {'speed':>7} {'lines':>6} {'nearest scoring':>16} {'chance there':>13}")
    misses, chances = [], []
    for layout, cue, velocity, _frame in chosen:
        speed = float(np.hypot(*velocity))
        aimed = float(np.degrees(np.arctan2(velocity[1], velocity[0]))) % 360.0
        angles, scored = sweep(layout, cue, speeds=(speed,), angle_step_deg=args.step)
        outcomes = scored[speed]
        gap = nearest_scoring(angles, outcomes, aimed)
        chance = float(probability(outcomes, args.step, args.spread)[
            int(round(aimed / args.step)) % len(angles)])
        misses.append(np.nan if gap is None else gap)
        chances.append(chance)
        shown = "none" if gap is None else f"{gap:.1f} deg"
        print(f"{aimed:8.1f} {speed:7.0f} {int(outcomes.sum()):6d} {shown:>16} {chance:13.2f}")

    misses = np.array(misses, dtype=float)
    found = misses[np.isfinite(misses)]
    print(f"\nplays where the sweep found any scoring line at all: "
          f"{len(found)}/{len(misses)}")
    if len(found):
        print(f"   distance from the played angle to the nearest one: "
              f"median {np.median(found):.1f} deg, "
              f"within 2 deg on {(found <= 2).mean() * 100:.0f}%")
    print(f"   chance the map gives the line actually played: "
          f"median {np.median(chances):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
