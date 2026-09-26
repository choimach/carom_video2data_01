"""스쿼트가 자료에 보이나 — 겨냥선에서 벗어난 거리가 무엇에 비례하나.

선수 (2026-09-26): *"1구까지 공이 구르면서 회전도 어느정도 작용해. 특히 먼 공의
경우는... 이런현상을 스쿼트라고 하고 겨냥할때 보정이 필요해."*

맞다, 그리고 **시뮬에 아예 없다**: `simulator.simulate()`의 `side_degrees`는
`bounce()`에서만 쓰인다 — 회전이 쿠션에서만 작용하고 수구는 1적구까지 완벽한
직선으로 간다. 스쿼트도 스워브도 0이다.

넣기 전에 **자료에 실제로 보이는지** 본다. 재는 것은 하나다:

> 영상에서 읽은 겨냥선에서 1적구 중심까지의 **수직거리**.
> 겨냥이 옳고 공이 곧게 간다면 이 값은 공 반지름 언저리여야 한다.

두 가지로 갈린다:

  · **거리에 비례한다** — 각을 잘못 읽으면 거리에 비례해 벌어진다. 스쿼트와
    구별이 안 된다 (스쿼트도 거리에 비례한다).
  · **회전량에 비례한다** — ★이것이 스쿼트의 지문이다. 스쿼트는 옆회전이
    있어야 생기고 그 크기에 비례한다. 각도 오차는 회전과 무관하다.

⚠️ **겨냥 읽기를 고친 뒤에 재야 한다.** 2026-09-26 이전의 `opening()`은
   겨냥 창이 충돌을 지나가서 절반이 충돌 **후** 방향을 읽고 있었다 (1적구를
   맞히는 비율 46%). 그 표본으로 잰 상관은 거리 +0.303 · 회전 −0.016이었는데,
   잡음이 반이라 믿을 값이 아니었다.

    ~/.venvs/carom/bin/python tools/check_squirt.py         # 스캔 전부
    ~/.venvs/carom/bin/python tools/check_squirt.py 12
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402

DIAMETER = 61.5


def collect(limit_files=None):
    rows = []
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
                layout, cue, velocity, _frame = start
                first = (shot.verdict or {}).get("first_ball")
                if not first or first == cue:
                    continue
                here = np.array(layout[cue], float)
                there = np.array(layout[first], float)
                speed = float(np.hypot(*velocity))
                if speed < 1e-6:
                    continue
                way = np.array(velocity, float) / speed
                to = there - here
                along = float(np.dot(to, way))
                if along <= 0:          # 1적구가 겨냥선 뒤 — 읽기가 실패한 판
                    continue
                rows.append({
                    "perp": abs(float(way[0] * to[1] - way[1] * to[0])),
                    "reach": float(np.linalg.norm(to)),
                    "spin": abs(float(shot.spin_x or 0.0)),
                    "speed": speed,
                    "thickness": shot.thickness,
                })
    return rows


def main(argv):
    limit = int(argv[0]) if argv else None
    rows = collect(limit)
    if len(rows) < 50:
        print(f"잴 판이 모자랍니다 ({len(rows)}개)."); return 1
    perp = np.array([r["perp"] for r in rows])
    reach = np.array([r["reach"] for r in rows])
    spin = np.array([r["spin"] for r in rows])
    print(f"{len(rows)}판 — 겨냥선에서 1적구 중심까지의 수직거리 "
          f"(공 지름 {DIAMETER} mm)\n")
    print(f"  중앙값 {np.median(perp):.1f} mm · 지름 안 {np.mean(perp < DIAMETER):.0%}")

    print("\n  1적구까지 거리별")
    for lo, hi in [(0, 500), (500, 900), (900, 1400), (1400, 4000)]:
        take = (reach >= lo) & (reach < hi)
        if take.sum() < 15:
            continue
        print(f"    {lo:4d}~{hi:<5d}{take.sum():4d}판 · 수직거리 중앙 "
              f"{np.median(perp[take]):6.1f} mm · 지름 밖 "
              f"{np.mean(perp[take] > DIAMETER):.0%}")

    print("\n  회전량별 (|spin_x|, 영상이 쿠션 반사각에서 잰 값)")
    for lo, hi in [(0, 10), (10, 25), (25, 45), (45, 200)]:
        take = (spin >= lo) & (spin < hi)
        if take.sum() < 15:
            continue
        print(f"    {lo:3d}~{hi:<4d}{take.sum():4d}판 · 수직거리 중앙 "
              f"{np.median(perp[take]):6.1f} mm · 지름 밖 "
              f"{np.mean(perp[take] > DIAMETER):.0%}")

    print(f"\n  거리와의 상관   {np.corrcoef(reach, perp)[0, 1]:+.3f}"
          "   (각을 잘못 읽어도 이렇게 된다 — 스쿼트와 구별 안 됨)")
    r_spin = float(np.corrcoef(spin, perp)[0, 1])
    print(f"  ★회전량과의 상관 {r_spin:+.3f}"
          "   (스쿼트의 지문 — 각도 오차는 회전과 무관하다)")
    # 거리를 나눠 각도로 바꾸면 거리 효과가 빠진다. 남는 것이 회전 효과다.
    angle = np.degrees(np.arctan2(perp, reach))
    print(f"  각도로 바꾼 뒤 회전량과의 상관 {np.corrcoef(spin, angle)[0, 1]:+.3f}")
    print(f"  (참고) 수직거리를 각도로 보면 중앙값 {np.median(angle):.2f}도")

    print("\n★회전량과의 상관이 0에 가까우면 지금 보이는 어긋남은 스쿼트가 아니라")
    print("  겨냥 읽기의 각도 오차다. 스쿼트를 넣어도 이 숫자는 안 줄어든다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
