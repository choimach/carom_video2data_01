"""프로 샷을 그대로 다시 쳐서, 공이 **멈춘 자리**를 영상과 견준다 — 물리의 끝 자리 자.

`tools/rest_error.py`는 물리를 재지 못한다 (2026-09-29): 열거가 찾은 줄은 경로 이름만
프로와 같고 강도·당점이 다르다. 여기서는 프로가 **실제로 친** 겨냥과 속도를
`validate_simulator.opening()`(되풀기)으로 읽어 `spin.py`로 끝까지 친다.

실제 끝 자리: 같은 이닝의 **다음 샷이 시작할 때**의 세 공 자리 (득점해서 같은 선수가
이어 친 판). 같은 스캔, 같은 틀이라 거울 맞추기가 필요 없다.

당점은 영상에 안 찍힌다. 두 가지로 친다:

* **고정** — 상단 2팁, 좌우 0 (`cushion_drift.py`가 충돌 앞뒤 속도로 고른 값)
* **첫 쿠션으로 고름** — 당점 격자에서 첫 쿠션 자리가 가장 맞는 것. ★끝 자리는
  고르는 데 쓰지 않으므로 끝 자리 오차는 정직한 답이다 (`check_after_cushion.js`와 같은 방식).

잣대: 공이 하나도 안 움직였다고 치기.

    ~/.venvs/carom/bin/python tools/replay_rest.py            # 스캔 전부
    ~/.venvs/carom/bin/python tools/replay_rest.py 5          # 앞 5개만
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate_with_spin  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402

BALLS = ("white", "yellow", "red")
FIXED = (0.0, 2.0)   # (좌우, 상하) 팁
GRID = [(s, v) for s in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0) for v in (-1.0, 0.0, 1.0, 2.0, 3.0)]


def rest_of(shot):
    return {c: np.asarray(shot.paths[c][-1], float) for c in BALLS}


def gap(rest, real):
    """세 공의 평균 거리 (mm), 그리고 공별 거리."""
    each = {c: float(np.hypot(*(rest[c] - real[c]))) for c in BALLS}
    return float(np.mean(list(each.values()))), each


def first_ball(played):
    return next((d for _f, kind, d in played.events if kind == "ball"), None)


def first_cushion(shot):
    return next((f for f, kind, _d in shot.events if kind == "cushion"), None)


def collect(limit_files=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:limit_files]:
        data = load_scan(path)
        result = analyse(data)
        fps = result["fps"]
        pos = data["positions"]
        for inning in result["innings"]:
            shots = inning.shots
            for shot, after in zip(shots, shots[1:]):
                if shot.inferred or getattr(shot, "rejections", None):
                    continue
                if after.cue_ball != shot.cue_ball or not shot.success:
                    continue
                # ★1적구를 같은 공으로 맞힌 판만 (2026-09-30). 안 거르면 시뮬이 1적구를
                # 빗나가 적구가 제자리에 선 판이 섞여, "안 움직였다"보다 나빠 보였다.
                events = (shot.verdict or {}).get("events") or []
                aimed = next((e.detail for e in events if e.kind == "ball"), None)
                if aimed is None:
                    continue
                real = {c: pos[c][after.start_frame] for c in BALLS}
                if not all(np.isfinite(v).all() for v in real.values()):
                    continue
                start = opening(shot, pos, fps)
                if start is None:
                    continue
                layout, cue, velocity, frame = start
                stay = {c: np.asarray(layout[c], float) for c in BALLS}

                fixed = simulate_with_spin(layout, cue, velocity, tips_side=FIXED[0],
                                           tips_vertical=FIXED[1], fps=fps)
                if first_ball(fixed) != aimed:
                    continue

                # 첫 쿠션 자리로 당점을 고른다 (끝 자리는 안 본다)
                real_first = next((e for e in events if e.kind == "cushion"), None)
                seen = pos[cue][frame:shot.end_frame + 1]
                chosen = None
                if real_first is not None and int(real_first.frame) < len(seen) \
                        and np.isfinite(seen[int(real_first.frame)]).all():
                    target = seen[int(real_first.frame)]
                    best = None
                    for s, v in GRID:
                        played = simulate_with_spin(layout, cue, velocity, tips_side=s,
                                                    tips_vertical=v, fps=fps)
                        at = first_cushion(played)
                        if at is None or first_ball(played) != aimed:
                            continue
                        miss = float(np.hypot(*(np.asarray(played.paths[cue][at]) - target)))
                        if best is None or miss < best[0]:
                            best = (miss, played)
                    chosen = best[1] if best else None

                # 시각별 수구 거리 — 실제 궤적이 찍혀 있는 동안만
                timeline = {}
                if chosen is not None:
                    mine = np.asarray(chosen.paths[cue])
                    for sec in (0.5, 1, 2, 3, 4, 6):
                        i = int(sec * fps)
                        if i < len(seen) and np.isfinite(seen[i]).all():
                            timeline[sec] = float(np.hypot(*(mine[min(i, len(mine) - 1)] - seen[i])))
                row = {"stay": gap(stay, real), "timeline": timeline, "fixed": gap(rest_of(fixed), real),
                       "chosen": gap(rest_of(chosen), real) if chosen else None,
                       "cue": cue, "moved": gap(stay, real)[0]}
                rows.append(row)
    return rows


def summary(name, values):
    v = np.array([x for x in values if x is not None], float)
    print(f"  {name:<22} n={len(v):4d}  중앙값 {np.median(v):5.0f} mm"
          f"  (25% {np.percentile(v, 25):4.0f} · 75% {np.percentile(v, 75):4.0f})")


def main(argv):
    limit = int(argv[0]) if argv else None
    rows = collect(limit)
    print(f"득점 뒤 이어 친 판 {len(rows)}개 — 세 공의 멈춘 자리, 공당 평균 거리\n")
    summary("공이 안 움직였다", [r["stay"][0] for r in rows])
    summary("다시 치기 · 고정 당점", [r["fixed"][0] for r in rows])
    summary("다시 치기 · 첫 쿠션 당점", [r["chosen"][0] if r["chosen"] else None for r in rows])

    print("\n공별 (첫 쿠션 당점):")
    for who in ("수구", "적구 둘"):
        vals = []
        for r in rows:
            if not r["chosen"]:
                continue
            each = r["chosen"][1]
            if who == "수구":
                vals.append(each[r["cue"]])
            else:
                vals.extend(each[c] for c in BALLS if c != r["cue"])
        summary(who, vals)

    print("\n수구, 친 뒤 몇 초에 (첫 쿠션 당점, 영상에 찍혀 있는 동안):")
    for sec in (0.5, 1, 2, 3, 4, 6):
        vals = [r["timeline"][sec] for r in rows if sec in r.get("timeline", {})]
        if len(vals) >= 5:
            summary(f"{sec}초", vals)

    ok = [r for r in rows if r["chosen"]]
    better = np.mean([r["chosen"][0] < r["stay"][0] for r in ok])
    print(f"\n  '안 움직였다'보다 가까운 판 {better:.0%}")
    print("  테이블 대각선 3,170 mm · 공 지름 61.5 mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
