"""회전이 얼마나 빨리 죽는지를 영상에 맞춰 잡는다.

`spin.SPIN_FRICTION`은 이 프로젝트에서 유일하게 **재지 않고 둔** 상수다. 처음
넣을 때의 변명은 "1초 안에서는 0.01이든 0.04든 밀리미터까지 같은 답이 나온다"
였는데, 그건 1초짜리 비교를 하던 시절의 말이다. 3쿠션은 4~8초짜리 사건이고,
회전은 쿠션을 세 번 지나며 계속 살아 있다. 선수의 말: "처음 시작한 공의 회전은
스트로크 강도, 1적구와의 충돌, 쿠션과의 충돌, 진행 거리에 따라 변한다."

당점은 영상에 찍히지 않으므로 플레이마다 팁을 맞춰 준다. 그러면 팁 13가지 중
가장 잘 맞는 것을 고르는 셈이라 오차가 낙관적으로 나오는데, 감쇠값끼리
비교할 때는 모두 같은 이득을 받으므로 순서는 믿을 수 있다.

    ~/.venvs/carom/bin/python tools/fit_spin.py            # 기본 200개
    ~/.venvs/carom/bin/python tools/fit_spin.py --plays 400 --sweep 0.005 0.01 0.03
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics import spin  # noqa: E402
from src.physics.simulator import simulate_with_spin  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening, travelled  # noqa: E402

TIPS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def gather(limit_plays, limit_files=None):
    """영상에서 배치와 첫 속도를 꺼내 둔다. 한 번만 하고 재사용한다."""
    plays = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:limit_files]:
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
                seen = data["positions"][cue][frame:shot.end_frame + 1]
                if len(seen) < 60:
                    continue
                cushions = sum(1 for e in (shot.verdict or {}).get("events") or []
                               if e.kind == "cushion")
                plays.append({"layout": {k: np.array(v) for k, v in layout.items()},
                              "cue": cue, "velocity": velocity, "fps": fps,
                              "seen": np.asarray(seen, dtype=float),
                              "cushions": cushions})
                if len(plays) >= limit_plays:
                    return plays
    return plays


def score(plays, **overrides):
    """이 상수들로 모든 플레이를 다시 치고, 얼마나 어긋나는지 모은다."""
    was = {name: getattr(spin, name) for name in overrides}
    for name, value in overrides.items():
        setattr(spin, name, value)
    try:
        one, two, cushion_gap, ratio = [], [], [], []
        for play in plays:
            seen, fps = play["seen"], play["fps"]
            best = None
            for tips in TIPS:
                shot = simulate_with_spin(play["layout"], play["cue"], play["velocity"],
                                          tips_side=tips, fps=fps)
                mine = shot.paths[play["cue"]]
                overlap = min(len(seen), len(mine))
                if overlap < 30:
                    continue
                gap = np.linalg.norm(seen[:overlap] - mine[:overlap], axis=1)
                gap = gap[np.isfinite(gap)]
                if not len(gap):
                    continue
                # 팁은 1초 지점의 어긋남으로 고른다. 그 뒤는 고르는 데 쓰지
                # 않으므로, 2초 지점은 이 감쇠값이 실제로 얼마나 맞는지에 대한
                # 정직한 답이 된다.
                at_one = int(min(fps, len(gap) - 1))
                if best is None or gap[at_one] < best[0]:
                    at_two = int(min(2 * fps, len(gap) - 1))
                    sim_cushions = sum(1 for f, _k, _r in shot.cushions if f <= overlap)
                    # 이동거리는 반드시 같은 구간끼리. 시뮬 공이 먼저 서면
                    # overlap이 짧아지는데, 그때 실제 쪽을 끝까지 재면 시뮬이
                    # 실제보다 느린 것처럼 보이는 만큼 한 번 더 깎인다.
                    best = (gap[at_one], gap[at_two] if len(gap) > fps else np.nan,
                            sim_cushions, travelled(mine[:overlap]),
                            travelled(seen[:overlap]))
            if best is None:
                continue
            one.append(best[0])
            two.append(best[1])
            cushion_gap.append(best[2] - play["cushions"])
            if best[4] > 100:
                ratio.append(best[3] / best[4])
        return {"one": np.nanmedian(one), "two": np.nanmedian(two),
                "cushions": np.median(cushion_gap),
                "same": float(np.mean(np.array(cushion_gap) == 0)),
                "ratio": np.median(ratio), "n": len(one)}
    finally:
        for name, value in was.items():
            setattr(spin, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plays", type=int, default=200)
    parser.add_argument("--files", type=int, default=None)
    parser.add_argument("--constant", default="SPIN_FRICTION",
                        help="훑을 상수 이름 (spin.py 안의 것)")
    parser.add_argument("--sweep", type=float, nargs="*",
                        default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.15])
    args = parser.parse_args()

    plays = gather(args.plays, args.files)
    print(f"플레이 {len(plays)}개 — 실제 쿠션 중앙값 "
          f"{np.median([p['cushions'] for p in plays]):.0f}")
    print(f"{args.constant} (지금 {getattr(spin, args.constant)})\n")
    print("  값     1초 오차   2초 오차   쿠션차(중앙)  같은수  이동비  n")
    for value in args.sweep:
        got = score(plays, **{args.constant: value})
        print(f"{value:<7} {got['one']:7.0f} mm {got['two']:8.0f} mm "
              f"{got['cushions']:9.1f}     {got['same']:5.0%}  {got['ratio']:5.2f}  {got['n']}")
    print("\n회전이 죽는 데 걸리는 시간 = ω / (감쇠 · g / R). 3쿠션이 4~8초이므로")
    print("그 안에서 눈에 띄게 줄어야 실제 공이다.")


if __name__ == "__main__":
    main()
