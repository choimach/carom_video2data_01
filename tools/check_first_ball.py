"""충돌 후 1적구가 얼마나 굴러가나 — 시뮬과 영상을 견준다.

선수 (2026-09-26): *"좋은데 충돌후 1적구가 움직이는 거리가 너무 긴것 같아."*

1적구의 최종 자리는 이제 순위에 쓰려는 값이다 (`rest`, 포지션 플레이). 그
자리가 틀리면 "다음 배치가 쉬운가"도 같이 틀린다. 그래서 먼저 잰다.

⚠️ **창 길이를 맞춘다.** `validate_simulator.py`는 실제를 창 전체에서, 시뮬을
   겹치는 구간까지만 재서 비가 저절로 작아진다 (HANDOFF의 "이동거리 0.65"
   항목). 여기서는 **같은 프레임 수**로 잰다.

⚠️ **멈추는 것까지 봤는지는 묻지 않는다.** 처음에는 물었고 12경기에서 판이
   **7개**만 남았다 (1적구가 멈추기 전에 카메라가 끊기는 판이 63%다 — 득점하면
   바로 화면을 돌린다). 그런데 **같은 창에서 재므로 물을 필요가 없다**: 카메라가
   60프레임에서 끊겼으면 실제 60프레임과 시뮬 60프레임을 견주면 되고, 그것이
   오히려 "너무 빨리 가나"에 정확히 맞는 비교다. `--stopped`로 옛 조건을 켤 수
   있게 남겨 둔다.

⚠️ **시뮬이 같은 공을 먼저 맞힌 판만 쓴다.** 안 그러면 "시뮬이 안 맞힌 공의
   이동거리 0"이 섞여 비가 0으로 눌린다 — 처음에 그렇게 재서 0.00이 나왔다.
   그리고 그 자체가 큰 값이다: **시뮬이 아무 공도 못 맞히는 판이 49%다.**

무엇을 보나:
  · 치우침 (중앙값 비) — 한쪽으로 쏠렸으면 상수가 틀렸다. 고칠 수 있다.
  · 흩어짐 — 판마다 다르면 모르는 것(당점 따위)이 있다.
  · 두께별 — 두껍게 맞히면 1적구가 많이 간다. 두께에 따라 어긋나면
    **에너지가 넘어가는 비율**이 틀린 것이다.

    ~/.venvs/carom/bin/python tools/check_first_ball.py        # 스캔 전부
    ~/.venvs/carom/bin/python tools/check_first_ball.py 8      # 앞 8개만
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening, travelled  # noqa: E402

STILL_MM = 6.0      # 마지막 8프레임에 이만큼도 안 움직이면 멈춘 것으로 본다


def collect(limit_files=None, stopped=False):
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
                layout, cue, velocity, frame = start
                first = (shot.verdict or {}).get("first_ball")
                if not first or first == cue:
                    continue
                side = shot.spin_x if shot.spin_x is not None else 0.0
                played = simulate(layout, cue, velocity, fps=fps, side_degrees=side)
                # ★시뮬이 **같은 공을 먼저 맞혔을 때만** 견준다. 안 그러면
                # "시뮬이 안 맞힌 공의 이동거리 0"이 섞여 비가 0으로 눌린다 —
                # 처음에 그렇게 재서 시뮬/실제가 0.00으로 나왔다.
                mine_first = next((detail for _f, kind, detail in played.events
                                   if kind == "ball"), None)
                if mine_first != first:
                    continue
                seen = data["positions"][first][frame:shot.end_frame + 1]
                mine = played.paths.get(first)
                if mine is None:
                    continue
                overlap = min(len(seen), len(mine))
                if overlap < 20:
                    continue
                real = seen[:overlap]
                if not np.isfinite(real).all():
                    continue
                rolling = travelled(real[-8:]) > STILL_MM
                if stopped and rolling:
                    continue
                far_real = travelled(real)
                far_sim = travelled(mine[:overlap])
                if far_real < 30:        # 거의 안 움직인 판은 비가 폭발한다
                    continue
                rows.append({
                    "real": float(far_real),
                    "sim": float(far_sim),
                    "ratio": float(far_sim / far_real),
                    "thickness": shot.thickness,
                    "frames": overlap,
                    "rolling": bool(rolling),
                })
    return rows


def band(rows, pick, edges, name):
    print(f"\n  {name}별")
    values = np.array([pick(r) for r in rows], float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = [r for r, v in zip(rows, values) if np.isfinite(v) and lo <= v < hi]
        if len(take) < 20:
            continue
        ratio = np.array([r["ratio"] for r in take])
        print(f"    {lo:.2f}~{hi:.2f}  시뮬/실제 중앙값 {np.median(ratio):5.2f}"
              f"  (실제 {np.median([r['real'] for r in take]):4.0f} mm,"
              f" 시뮬 {np.median([r['sim'] for r in take]):4.0f} mm, {len(take)}판)")


def main(argv):
    stopped = "--stopped" in argv
    rest = [a for a in argv if not a.startswith("--")]
    limit = int(rest[0]) if rest else None
    rows = collect(limit, stopped)
    if not rows:
        print("잴 판이 없습니다."); return 1
    ratio = np.array([r["ratio"] for r in rows])
    real = np.array([r["real"] for r in rows])
    sim = np.array([r["sim"] for r in rows])
    print(f"{len(rows)}판 — 1적구가 멈추는 것까지 본 판만, 같은 창에서 잰다\n")
    print(f"  실제 굴러간 거리  중앙값 {np.median(real):5.0f} mm"
          f"  (25% {np.percentile(real, 25):.0f} · 75% {np.percentile(real, 75):.0f})")
    print(f"  시뮬 굴러간 거리  중앙값 {np.median(sim):5.0f} mm"
          f"  (25% {np.percentile(sim, 25):.0f} · 75% {np.percentile(sim, 75):.0f})")
    mid = float(np.median(ratio))
    half = float(np.percentile(ratio, 75) - np.percentile(ratio, 25)) / 2.0
    print(f"\n  시뮬/실제  중앙값 {mid:.2f} · 흩어짐 ±{half:.2f}")
    print(f"  시뮬이 더 멀리 보낸 판 {np.mean(ratio > 1):.0%}")
    verdict = ("★시뮬이 너무 멀리 보낸다 — 선수 말이 맞다" if mid > 1.15
               else ("★시뮬이 너무 짧게 보낸다" if mid < 0.87
                     else "치우침이 크지 않다"))
    print(f"  {verdict}")
    band(rows, lambda r: r["thickness"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.01], "두께")
    for flag, name in ((False, "1적구가 멈추는 것까지 본 판"), (True, "카메라가 먼저 끊긴 판")):
        take = [r for r in rows if r["rolling"] is flag]
        if len(take) >= 20:
            got = np.array([r["ratio"] for r in take])
            print(f"\n  {name}  시뮬/실제 중앙값 {np.median(got):.2f}  ({len(take)}판)")
    print("\n★치우침(중앙값이 1에서 떨어진 정도)은 상수를 고치면 준다.")
    print("  흩어짐은 판마다 모르는 것이 다르다는 뜻이고 자료로는 안 준다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
