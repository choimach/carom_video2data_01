"""Play the captured shots again in the simulator and see how far it drifts.

For each usable play: take the three balls where they actually stood, take the
cue ball's measured opening velocity, and let the simulator run. Then compare
what it produced against what the camera saw.

A simulator built on measured constants has no free parameters to tune, so this
is a test of the constants as much as of the code. Three things are compared,
in rising order of how hard they are: how many cushions the cue ball took, how
far it travelled in total, and where it actually was as time went on.

    ~/.venvs/carom/bin/python tools/validate_simulator.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate  # noqa: E402
from src.physics.stroke import _travel  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402

BALLS = ("white", "yellow", "red")
# Frames to leave alone after the ball first moves, and the longest baseline to
# measure the direction over. The cue is still on the ball for the first few and
# its centroid is smeared; after that, the longer the baseline the better, which
# is what a straight line through noisy points would predict. Measured against
# the rail the cue ball actually reached, on plays where a window could not
# overrun the first event: 41 mm at a 5-frame baseline, 35 at 12, 24 at 24.
OPENING_SKIP = 4
OPENING_SPAN = 24
# 이보다 짧으면 방향이라고 부를 수 없다. 두 점으로는 잡음이 각을
# 통째로 정한다. 3프레임이면 60fps에서 50 ms다.
MIN_OPENING_FRAMES = 3


def opening(shot, positions, fps):
    """Where the balls stood and how fast the cue ball left, at the first motion.

    The baseline is stretched to whatever the play allows and no further. A
    window that runs past the first cushion or the first contact measures a
    corner rather than a line, which is worse than measuring a short one: the
    same sweep that prefers 24 frames prefers 6 once the window is free to
    overrun, because then it usually has.
    """
    cue = positions[shot.cue_ball]
    events = (shot.verdict or {}).get("events") or []
    first_event = min((int(e.frame) for e in events), default=10 ** 6)

    for frame in range(shot.start_frame, min(shot.end_frame, shot.start_frame + 120)):
        here, ahead = cue[frame], cue[frame + OPENING_SKIP + 4]
        if not (np.isfinite(here).all() and np.isfinite(ahead).all()):
            continue
        if float(np.hypot(*(ahead - here))) / (OPENING_SKIP + 4) * fps < 300.0:
            continue

        # ★2026-09-26에 여기가 범인으로 드러났다. 옛 코드는
        #
        #     room = ... - (frame + OPENING_SKIP)
        #     span = int(np.clip(room, 4, OPENING_SPAN))
        #
        # 였고, **room이 4보다 작아도 4로 올렸다.** 그런데 실제로 쓸 수 있는
        # 창은 대개 1~6 프레임뿐이다 — `start_frame`은 검출이 "움직였다"고
        # 알아차린 때라 수구가 이미 1,500~5,200 mm/s로 구르고 있고, 첫 사건은
        # 그로부터 1~42 프레임 뒤다. 거기서 OPENING_SKIP(4)까지 건너뛰면
        # **재기도 전에 충돌을 지나간다.** 그러면 잰 것은 충돌 **후** 방향이고
        # 어디든 가리킨다.
        #
        # 그 대가 (12경기, 368판):
        #   시뮬이 아무 공도 못 맞힘 180판(49%) · 빗나간 판의 수직거리 중앙값
        #   161 mm(공 2.6개) · 78판은 1적구가 겨냥선 **뒤에** 있었다
        #   빗나간 판의 59%가 room<4, 44%가 room<0
        #
        # 그러므로 **건너뛰기도 창에 맞춰 줄인다.** 큐가 아직 공에 닿아 있는
        # 앞 몇 프레임은 버리고 싶지만, 버릴 것이 없으면 버리지 않는다.
        available = shot.start_frame + first_event - 2 - frame
        if available < MIN_OPENING_FRAMES:
            continue
        skip = int(min(OPENING_SKIP, max(0, available - MIN_OPENING_FRAMES)))
        span = int(np.clip(available - skip, MIN_OPENING_FRAMES, OPENING_SPAN))
        direction, speed = _travel(cue, frame + skip, frame + skip + span)
        if direction is None:
            continue
        layout = {}
        for colour in BALLS:
            point = positions[colour][frame]
            if not np.isfinite(point).all():
                return None
            layout[colour] = tuple(float(v) for v in point)
        return layout, shot.cue_ball, tuple(direction * speed * fps), frame
    return None


def travelled(path):
    good = path[np.isfinite(path).all(axis=1)]
    if len(good) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(good, axis=0), axis=1)))


def main(argv=None):
    limit = int(argv[0]) if argv else None
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:limit]:
        data = load_scan(path)
        result = analyse(data)
        fps = result["fps"]
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None):
                    continue
                start = opening(shot, data["positions"], fps)
                if start is None:
                    continue
                layout, cue, velocity, frame = start
                # The side the player actually put on, as measured from the
                # first rebound. Simulating every shot with none of it is
                # simulating a game nobody plays.
                side = shot.spin_x if shot.spin_x is not None else 0.0
                played = simulate(layout, cue, velocity, fps=fps, side_degrees=side)

                seen = data["positions"][cue][frame:shot.end_frame + 1]
                mine = played.paths[cue]
                overlap = min(len(seen), len(mine))
                if overlap < 30:
                    continue
                gap = np.linalg.norm(seen[:overlap] - mine[:overlap], axis=1)
                gap = gap[np.isfinite(gap)]
                real_cushions = sum(1 for e in (shot.verdict or {}).get("events") or []
                                    if e.kind == "cushion")
                # Count the simulated cushions over the same stretch of time the
                # camera watched. The simulator runs until the balls stop; the
                # play's own event list ends when the play does, so comparing
                # the two totals counts cushions the camera was never going to
                # see and makes the simulator look like it overshoots.
                sim_cushions = sum(1 for frame, _kind, _rail in played.cushions
                                   if frame <= overlap)
                # Where the cue ball first met a rail, and where it went next.
                # A three-cushion path is chaotic - an error at the first rail
                # is multiplied by every rail after it - so this is the part
                # that tests the physics rather than the patience.
                first_real = next((e for e in (shot.verdict or {}).get("events") or []
                                   if e.kind == "cushion"), None)
                first_sim = played.cushions[0] if played.cushions else None
                meeting = turning = np.nan
                if first_real is not None and first_sim is not None:
                    real_at = first_real.frame
                    sim_at = first_sim[0]
                    if real_at < len(seen) and sim_at < len(mine):
                        meeting = float(np.hypot(*(seen[real_at] - mine[sim_at])))
                    after_real = real_at + 8
                    after_sim = sim_at + 8
                    if after_real < len(seen) and after_sim < len(mine):
                        a = seen[after_real] - seen[real_at]
                        b = mine[after_sim] - mine[sim_at]
                        if np.isfinite(a).all() and np.hypot(*a) > 1 and np.hypot(*b) > 1:
                            cosine = float(np.dot(a, b) / (np.hypot(*a) * np.hypot(*b)))
                            turning = abs(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
                rows.append({
                    "speed": float(np.hypot(*velocity)) / 1000.0,
                    "cushions_real": real_cushions,
                    "cushions_sim": sim_cushions,
                    "travel_real": travelled(seen),
                    "travel_sim": travelled(mine[:overlap]),
                    "drift_1s": float(gap[min(len(gap) - 1, int(fps))]) if len(gap) else np.nan,
                    "drift_2s": float(gap[min(len(gap) - 1, int(2 * fps))]) if len(gap) else np.nan,
                    "frames": overlap,
                    "first_rail_mm": meeting,
                    "rebound_deg": turning,
                })

    if not rows:
        print("no plays to compare")
        return 1
    print(f"{len(rows)} plays replayed\n")

    cushions_real = np.array([r["cushions_real"] for r in rows])
    cushions_sim = np.array([r["cushions_sim"] for r in rows])
    same = cushions_real == cushions_sim
    print(f"cushions taken by the cue ball")
    print(f"   same count      {same.mean() * 100:.0f}%")
    print(f"   within one      {(np.abs(cushions_real - cushions_sim) <= 1).mean() * 100:.0f}%")
    print(f"   real median {np.median(cushions_real):.0f}, simulated {np.median(cushions_sim):.0f}")

    real = np.array([r["travel_real"] for r in rows])
    sim = np.array([r["travel_sim"] for r in rows])
    keep = real > 500
    print(f"\ndistance the cue ball covered")
    print(f"   simulated / real   median {np.median(sim[keep] / real[keep]):.2f}")

    print(f"\nthe first rail, before the path has anywhere to go wrong")
    for label, key, unit in (("where it met the rail", "first_rail_mm", "mm"),
                             ("which way it left", "rebound_deg", "deg")):
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            print(f"   {label:<22} median {np.median(values):5.0f} {unit}   (n={len(values)})")

    print(f"\nhow far the simulated cue ball was from the real one")
    for label, key in (("after 1 second", "drift_1s"), ("after 2 seconds", "drift_2s")):
        values = np.array([r[key] for r in rows])
        values = values[np.isfinite(values)]
        print(f"   {label:<16} median {np.median(values):6.0f} mm   "
              f"(a ball is {61.5:.0f} mm across)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
