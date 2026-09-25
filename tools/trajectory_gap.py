"""궤적의 오차가 **치우침인가 흩어짐인가** — 자료를 더 모아서 될 일인지의 답.

선수가 물었다 (2026-09-25): *"문제는 모델이 얼마나 신빙성있는 궤적을 보여주나
하는것이잖아? 그러려면 데이터가 얼마나 더 필요해?"*

답은 오차의 **모양**에 달려 있다. 같은 크기의 오차라도:

* **치우침** (부호가 한쪽, 중앙값이 흩어짐만큼 큼) — 상수나 식이 틀렸다.
  판을 늘리면 상수가 정확해지므로 **치우침은 줄어든다.**
* **흩어짐** (부호가 반반, 중앙값 ≈ 0) — 판마다 우리가 **모르는 것**이 다르다는
  뜻이다. ★**판을 늘려도 각 판의 오차는 그대로다.** 판을 늘리면 상수의
  불확실성만 줄고, 판마다의 잔차는 안 준다. 줄이려면 그 모르는 것을 알아내야
  한다 — 여기서는 **당점**이 유력하다 (영상에 안 찍힌다).

★그래서 **자료를 무한히 모았을 때의 이득은 치우침만큼**이다. 흩어짐은 남는다.

`validate_simulator.py`는 **절댓값**만 재서 둘을 못 가른다. 여기서는 부호를
살려서 잰다. 재는 곳은 **첫 쿠션** — 경로가 어긋날 기회가 생기기 전이라
물리만 시험하는 자리다.

⚠️ CLAUDE.md §7의 덫: 이동거리로 감쇠를 다시 잡지 말 것. 영상 창이 닫힐 때
   아직 구르던 공을 "다 간 거리"로 읽어 테이블이 2.5배 느려진 적이 있다.
   여기서 이동거리는 **참고로만** 찍고 맞추는 데 쓰지 않는다.

    ~/.venvs/carom/bin/python tools/trajectory_gap.py         # 스캔 전부
    ~/.venvs/carom/bin/python tools/trajectory_gap.py 10      # 앞 10개만
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


def signed_turn(a, b):
    """b가 a에서 몇 도 돌아갔나 — **부호를 살려서** (반시계가 +)."""
    cross = a[0] * b[1] - a[1] * b[0]
    dot = float(np.dot(a, b))
    return float(np.degrees(np.arctan2(cross, dot)))


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
                layout, cue, velocity, frame = start
                side = shot.spin_x if shot.spin_x is not None else 0.0
                played = simulate(layout, cue, velocity, fps=fps, side_degrees=side)
                seen = data["positions"][cue][frame:shot.end_frame + 1]
                mine = played.paths[cue]
                if min(len(seen), len(mine)) < 30:
                    continue
                first_real = next((e for e in (shot.verdict or {}).get("events") or []
                                   if e.kind == "cushion"), None)
                first_sim = played.cushions[0] if played.cushions else None
                if first_real is None or first_sim is None:
                    continue
                real_at, sim_at = int(first_real.frame), int(first_sim[0])
                if real_at + 8 >= len(seen) or sim_at + 8 >= len(mine):
                    continue
                if not np.isfinite(seen[real_at]).all() or not np.isfinite(seen[real_at + 8]).all():
                    continue
                a = seen[real_at + 8] - seen[real_at]        # 실제로 나간 방향
                b = mine[sim_at + 8] - mine[sim_at]          # 시뮬이 나간 방향
                if np.hypot(*a) <= 1 or np.hypot(*b) <= 1:
                    continue
                rows.append({
                    # 쿠션에 닿기까지 걸린 시간의 차 (프레임) — 부호 있음
                    "when": sim_at - real_at,
                    # 쿠션을 만난 자리의 차 — ⚠️ 절댓값이라 중앙값이 0일 수
                    # 없다. 이 줄의 "비"는 치우침의 증거로 읽지 말 것.
                    "where": float(np.hypot(*(seen[real_at] - mine[sim_at]))),
                    # ★나간 각의 차, 부호를 살려서
                    "turn": signed_turn(a, b),
                    "side": float(side),
                    "travel_real": travelled(seen),
                    "travel_sim": travelled(mine[:min(len(seen), len(mine))]),
                })
    return rows


def spread(name, values, unit):
    """치우침(중앙값)과 흩어짐(사분위 범위의 절반)을 나란히."""
    v = np.array([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return
    mid = float(np.median(v))
    half = float(np.percentile(v, 75) - np.percentile(v, 25)) / 2.0
    ratio = abs(mid) / (half + 1e-9)
    verdict = ("★치우침 — 상수/식을 고쳐야 한다" if ratio > 1.0
               else ("반반" if ratio > 0.4
                     else "흩어짐 — 판을 늘려도 안 준다 (숨은 변수)"))
    print(f"  {name:<24} 중앙값 {mid:+8.1f} {unit:<4} · 흩어짐 ±{half:6.1f} "
          f"· 비 {ratio:4.2f}  {verdict}")


def main(argv):
    limit = int(argv[0]) if argv else None
    rows = collect(limit)
    if not rows:
        print("비교할 판이 없습니다."); return 1
    print(f"{len(rows)}판을 다시 쳐 봤다 — 재는 곳은 **첫 쿠션**\n")
    print("치우침(중앙값)이 흩어짐보다 크면 자료가 아니라 식이 문제다")
    spread("첫 쿠션에 닿은 시각", [r["when"] for r in rows], "프레임")
    spread("나간 각", [r["turn"] for r in rows], "도")
    spread("쿠션을 만난 자리(절댓값)", [r["where"] for r in rows], "mm")

    turn = np.array([r["turn"] for r in rows])
    print(f"\n  나간 각이 한쪽으로 몰렸나: + {np.mean(turn > 0):.0%} · − {np.mean(turn < 0):.0%}"
          f"  (반반이면 잡음, 한쪽이면 치우침)")

    side = np.array([r["side"] for r in rows])
    ok = np.isfinite(side) & np.isfinite(turn)
    if ok.sum() > 20:
        print(f"  회전을 준 정도와 각 오차의 상관 {np.corrcoef(side[ok], turn[ok])[0,1]:+.3f}"
              f"   (크면 회전 모형이 범인이다)")

    real = np.array([r["travel_real"] for r in rows], float)
    sim = np.array([r["travel_sim"] for r in rows], float)
    ok = np.isfinite(real) & np.isfinite(sim) & (real > 0)
    print(f"\n  (참고, 맞추는 데 쓰지 말 것) 이동거리 시뮬/실제 중앙값 "
          f"{np.median(sim[ok]/real[ok]):.2f}")

    print("\n★판을 늘려서 줄어드는 것은 **치우침**뿐이다 — 상수가 정확해지니까.")
    print("  흩어짐은 판마다 다른 숨은 변수이고, 판을 늘려도 그대로다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
