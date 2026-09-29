"""시뮬 수구가 **언제** 느려지나 — 출발 · 1적구 앞 · 1적구 뒤의 속도를 영상과 견준다.

`tools/cushion_drift.py` (2026-09-29): 첫 쿠션을 나갈 때 시뮬 속도가 실제의 0.69배,
그런데 구간 감속도와 쿠션 반발은 거의 맞다. 그러면 첫 쿠션 **전에** 잃는다.
여기서 어디인지 가른다.

속도는 접촉 흐림을 피해 몇 프레임 떨어진 두 점의 거리/프레임. log(시뮬/실제).

    ~/.venvs/carom/bin/python tools/speed_probe.py       # 앞 25개 스캔
    ~/.venvs/carom/bin/python tools/speed_probe.py 61
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate_with_spin  # noqa: E402

# ★조언판(`sim.js`)과 같은 `spin.py` 모형으로 친다 (2026-09-29). 옛 `simulate()`는
# 상하 당점이 없어 1적구 충돌 뒤 수구 속도가 실제의 0.83배였다. 상단 0·1·2팁을
# 견주니 2팁이 충돌 앞뒤 속도를 가장 잘 맞췄다 (1.04배 · 0.94배, 앞 6경기).
# 판마다 맞춘 값이 아니라 **고정값**이다 — 당점은 영상에 안 찍힌다.
TOP_TIPS = 2.0


def simulate(layout, cue, velocity, fps=60.0, side_degrees=0.0):
    return simulate_with_spin(layout, cue, velocity, tips_side=0.0,
                              tips_vertical=TOP_TIPS, fps=fps)
from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402


def speed(path, a, b):
    if a < 0 or b >= len(path):
        return None
    q = path[[a, b]]
    return None if not np.isfinite(q).all() else np.hypot(*(q[1] - q[0])) / (b - a)


def main(argv):
    limit = int(argv[0]) if argv else 25
    got = {"출발 직후 (1~4프레임)": [], "1적구 맞기 직전": [],
           "1적구 맞은 직후 수구": [], "1적구 맞은 직후 1적구": []}
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
                side = shot.spin_x if shot.spin_x is not None else 0.0
                played = simulate(layout, cue, velocity, fps=fps, side_degrees=side)
                first = (shot.verdict or {}).get("first_ball")
                hit = next(((f, d) for f, kind, d in played.events if kind == "ball"), None)
                real_hit = next((e for e in (shot.verdict or {}).get("events") or []
                                 if e.kind == "ball"), None)
                if not first or not hit or hit[1] != first or real_hit is None:
                    continue
                seen = data["positions"][cue][frame:shot.end_frame + 1]
                mine = np.asarray(played.paths[cue])
                ra, sa = int(real_hit.frame), int(hit[0])
                obj_seen = data["positions"][first][frame:shot.end_frame + 1]
                obj_mine = np.asarray(played.paths[first])
                pairs = {
                    "출발 직후 (1~4프레임)": (speed(seen, 1, 4), speed(mine, 1, 4)),
                    "1적구 맞기 직전": ((speed(seen, ra - 6, ra - 2), speed(mine, sa - 6, sa - 2))
                                   if ra > 8 and sa > 8 else (None, None)),
                    "1적구 맞은 직후 수구": (speed(seen, ra + 3, ra + 8), speed(mine, sa + 3, sa + 8)),
                    "1적구 맞은 직후 1적구": (speed(obj_seen, ra + 3, ra + 8),
                                        speed(obj_mine, sa + 3, sa + 8)),
                }
                for name, (real, sim) in pairs.items():
                    if real and sim:
                        got[name].append(np.log(sim / real))
    for name, v in got.items():
        v = np.array(v)
        half = (np.percentile(v, 75) - np.percentile(v, 25)) / 2
        print(f"  {name:<16} n={len(v):4d}  log(시뮬/실제) {np.median(v):+.2f} ±{half:.2f}"
              f"  (배율 {np.exp(np.median(v)):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
