"""시뮬레이터의 좌우 회전 부호가 영상과 같은 쪽인지 영상에 대고 판정한다.

선수: "당점결정을 좌우가 바뀌는 것 같아."

이건 눈으로 따질 일이 아니다. 화면은 y가 아래로 가고, 회전의 부호는 좌표계의
손잡이에 달려 있어서, 종이에 그려 놓고 따지면 서로 다른 답이 나온다. 대신
**영상이 이미 답을 알고 있다**: `english_side()`가 쿠션 반사각에서 어느 쪽
회전이었는지를 재 두었고(양수가 오른쪽, 위에서 보아 시계 방향), 680개 플레이에
그 값이 있다.

그러면 물음은 하나다 — 그 회전을 시뮬레이터에 **그대로** 넣을 때와 **뒤집어**
넣을 때, 어느 쪽이 영상에 가까운가. 뒤집은 쪽이 가까우면 부호가 반대인 것이고,
그건 화면에 적히는 당점의 좌우가 통째로 틀렸다는 뜻이다.

    ~/.venvs/carom/bin/python tools/check_side_sign.py --plays 120
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate_with_spin  # noqa: E402
from src.pipeline import analyse, english_side, load_scan  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402

# 재어 둔 것은 좌우 방향뿐이고 팁 수는 아니다. 실전 당점 2팁으로 놓고 본다.
TIPS = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plays", type=int, default=120)
    parser.add_argument("--files", type=int, default=8)
    args = parser.parse_args()

    same, flipped, none = [], [], []
    seen = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:args.files]:
        data = load_scan(path)
        result = analyse(data)
        fps = result["fps"]
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None):
                    continue
                events = (shot.verdict or {}).get("events") or []
                side = english_side(shot, data["positions"], events)
                if side is None or abs(side) < 1e-9:
                    continue
                start = opening(shot, data["positions"], fps)
                if start is None:
                    continue
                layout, cue, velocity, frame = start
                watched = np.asarray(data["positions"][cue][frame:shot.end_frame + 1],
                                     dtype=float)
                if len(watched) < 60:
                    continue

                def apart(tips):
                    shot_now = simulate_with_spin({k: np.array(v) for k, v in layout.items()},
                                                  cue, velocity, tips_side=tips, fps=fps)
                    mine = shot_now.paths[cue]
                    overlap = min(len(watched), len(mine))
                    gap = np.linalg.norm(watched[:overlap] - mine[:overlap], axis=1)
                    gap = gap[np.isfinite(gap)]
                    if not len(gap):
                        return np.nan
                    return float(gap[int(min(fps, len(gap) - 1))])

                # 재어 둔 쪽 그대로, 뒤집어서, 그리고 회전 없이.
                hand = 1.0 if side > 0 else -1.0
                same.append(apart(TIPS * hand))
                flipped.append(apart(-TIPS * hand))
                none.append(apart(0.0))
                seen += 1
                if seen >= args.plays:
                    break
            if seen >= args.plays:
                break
        if seen >= args.plays:
            break

    if not seen:
        print("회전 방향이 측정된 플레이를 찾지 못했습니다")
        return 1

    same, flipped, none = np.array(same), np.array(flipped), np.array(none)
    print(f"회전 방향이 측정된 플레이 {seen}개, 1초 뒤 위치 오차 중앙값\n")
    print(f"  영상이 잰 쪽 그대로   {np.nanmedian(same):6.0f} mm")
    print(f"  좌우를 뒤집어서       {np.nanmedian(flipped):6.0f} mm")
    print(f"  회전 없이             {np.nanmedian(none):6.0f} mm\n")

    better = int(np.sum(same < flipped))
    print(f"  플레이별로 그대로가 더 가까운 경우  {better}/{seen} ({better / seen:.0%})")
    if np.nanmedian(same) < np.nanmedian(flipped) and better > seen * 0.5:
        print("\n→ 부호는 맞습니다. 좌우가 바뀐 것은 시뮬레이터가 아니라 다른 곳입니다.")
    elif np.nanmedian(flipped) < np.nanmedian(same):
        print("\n→ 부호가 반대입니다. 화면에 적히는 당점의 좌우가 통째로 틀렸습니다.")
    else:
        print("\n→ 가르지 못했습니다. 이 자료로는 어느 쪽이라고 말할 수 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
