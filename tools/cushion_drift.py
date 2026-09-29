"""끝 자리 오차가 **어디서** 불어나나 — 쿠션마다 쪼개 잰다.

`tools/rest_error.py` (2026-09-29): 시뮬의 끝 자리는 공당 696 mm 틀리고, "공이 안
움직였다"고 치면 743 mm다. 첫 쿠션은 고쳤는데(`trajectory_gap.py`) 멈추는 자리는
아니다. 그 사이 어디서 무너지는지를 본다.

쿠션 k마다 셋을 가른다:

* **물려받은 것** — 쿠션에 들어올 때 이미 틀린 방향 (앞 쿠션들의 잔재)
* **쿠션이 더한 것** — (실제 나간 각 − 실제 들어온 각) − (시뮬의 같은 것).
  들어온 방향이 틀려도 이것은 쿠션 모형만 시험한다.
* **속도** — 쿠션 반발(나감/들어옴 속도비)과, 쿠션 사이 구간의 감속
  (다음 쿠션에 들어올 때 / 이번 쿠션을 나갈 때). ★둘 다 **영상 창 안의 국소
  속도비**라 CLAUDE.md §7의 덫(창이 닫힐 때 아직 구르던 공)에 안 걸린다.
  그래도 이것으로 상수를 **맞추지는** 않는다 — 어디가 틀렸는지 찾는 자다.

k번째까지 쿠션 이름이 실제와 시뮬에서 같게 이어진 판만 k에서 잰다. 이름이
갈라지는 비율도 함께 찍는다 — 그 자리가 경로가 어긋나는 곳이다.

수구만 본다. 1적구를 같은 공으로 맞힌 판만 쓴다 (`trajectory_gap.py`와 같다).

    ~/.venvs/carom/bin/python tools/cushion_drift.py          # 스캔 전부
    ~/.venvs/carom/bin/python tools/cushion_drift.py 5        # 앞 5개만
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
from tools.trajectory_gap import signed_turn  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402

K = 5
# 접촉 흐림을 피해 접촉 NEAR~FAR 프레임 앞뒤로 방향·속도를 잰다. 3~8·6~12·9~16을
# 견줘 반발 치우침은 셋 다 −0.17~−0.22로 같았고, 6~12가 흩어짐이 가장 작았다
# (±0.35 → ±0.10, 2026-09-29).
NEAR, FAR = 6, 12


def around(path, at):
    """접촉 자리 at 앞뒤의 방향 벡터와 속도(mm/프레임)."""
    if at - FAR < 0 or at + FAR >= len(path):
        return None
    pts = path[[at - FAR, at - NEAR, at + NEAR, at + FAR]]
    if not np.isfinite(pts).all():
        return None
    d_in, d_out = pts[1] - pts[0], pts[3] - pts[2]
    v_in, v_out = np.hypot(*d_in) / (FAR - NEAR), np.hypot(*d_out) / (FAR - NEAR)
    if v_in <= 0.2 or v_out <= 0.2:
        return None
    return d_in, d_out, v_in, v_out


def collect(limit_files=None):
    rows = {k: [] for k in range(1, K + 1)}
    reach = {k: [0, 0] for k in range(1, K + 1)}   # [이름이 같게 이어진 판, 갈라진 판]
    short = {k: [0, 0, 0] for k in range(1, K + 1)}  # k번째가 없다: [실제만, 시뮬만, 둘 다]
    shots = 0
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
                first_ball = (shot.verdict or {}).get("first_ball")
                mine_first = next((d for _f, kind, d in played.events if kind == "ball"), None)
                if not first_ball or mine_first != first_ball:
                    continue
                seen = data["positions"][cue][frame:shot.end_frame + 1]
                mine = np.asarray(played.paths[cue])
                real = [e for e in (shot.verdict or {}).get("events") or [] if e.kind == "cushion"]
                sim = played.cushions
                shots += 1
                prev = None
                for k in range(1, K + 1):
                    if len(real) < k or len(sim) < k:
                        short[k][(len(real) < k) + 2 * (len(sim) < k) - 1] += 1
                        break
                    if real[k - 1].detail != sim[k - 1][2]:
                        reach[k][1] += 1
                        break
                    reach[k][0] += 1
                    ra, sa = int(real[k - 1].frame), int(sim[k - 1][0])
                    r, s = around(seen, ra), around(mine, sa)
                    if r is None or s is None:
                        prev = None
                        continue
                    row = {
                        "where": float(np.hypot(*(seen[ra] - mine[sa]))),
                        "when": sa - ra,
                        "in": signed_turn(r[0], s[0]),       # 물려받은 방향 오차
                        "out": signed_turn(r[1], s[1]),      # 나간 방향 오차
                        # 쿠션이 더한 것: 각 쪽의 꺾임을 견준다
                        "bend": signed_turn(r[0], r[1]) - signed_turn(s[0], s[1]),
                        # 반발: 속도비의 로그 차 (+면 시뮬이 덜 잃는다)
                        "bounce": float(np.log(s[3] / s[2]) - np.log(r[3] / r[2])),
                        "bounce_real": float(np.log(r[3] / r[2])),
                        "bounce_sim": float(np.log(s[3] / s[2])),
                        "leg": None,
                        # 나가는 속도 자체 (+면 시뮬이 빠르다)
                        "speed": float(np.log(s[3] / r[3])),
                        "decel": None,
                    }
                    # 앞 쿠션을 나간 뒤 여기 들어올 때까지의 감속 (+면 시뮬이 덜 느려진다)
                    if prev is not None:
                        row["leg"] = float(np.log(s[2] / prev[1]) - np.log(r[2] / prev[0]))
                        # 구간의 평균 감속도, mm/s² (+면 시뮬이 더 세게 선다). 속도비와
                        # 달리 속도 크기에 안 묶인다 — 구름 감속은 속도와 거의 무관하다.
                        real_dec = (prev[0] - r[2]) * fps * fps / max(ra - prev[2], 1)
                        sim_dec = (prev[1] - s[2]) * fps * fps / max(sa - prev[3], 1)
                        row["decel"] = float(sim_dec - real_dec)
                        row["decel_real"] = float(real_dec)
                    rows[k].append(row)
                    prev = (r[3], s[3], ra, sa)
    return rows, reach, short, shots


def mid_half(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) < 5:
        return None
    return float(np.median(v)), float(np.percentile(v, 75) - np.percentile(v, 25)) / 2.0, len(v)


def main(argv):
    limit = int(argv[0]) if argv else None
    rows, reach, short, shots = collect(limit)
    print(f"1적구를 같게 맞힌 판 {shots}개 — 수구의 쿠션 k번째에서\n")
    print("   k  이름 같음  갈라짐 |  자리 mm  |  시각 프레임 | 물려받은 각 | 나간 각 "
          "| ★쿠션이 더한 각 | 반발 log비 | 구간 감속 log비"
          " | 나가는 속도 log비 | 감속도 차 mm/s² | 실제 감속도 | 실제 반발 | 시뮬 반발")
    for k in range(1, K + 1):
        same, split = reach[k]
        cells = []
        for key, digits in (("where", 0), ("when", 1), ("in", 1), ("out", 1),
                            ("bend", 1), ("bounce", 2), ("leg", 2),
                            ("speed", 2), ("decel", 0), ("decel_real", 0),
                            ("bounce_real", 2), ("bounce_sim", 2)):
            got = mid_half([r.get(key) for r in rows[k]])
            if got is None:
                cells.append("—")
            elif key == "where":
                cells.append(f"{got[0]:5.0f}")
            else:
                cells.append(f"{got[0]:+.{digits}f} ±{got[1]:.{digits}f}")
        n = len(rows[k])
        print(f"  {k}  {same:6d}  {split:6d}  | " + " | ".join(cells) + f"   (n={n})")
    print("\n  k번째 쿠션이 없어서 멈춘 판:  실제에 없음 · 시뮬에 없음 · 둘 다")
    for k in range(1, K + 1):
        print(f"    {k}   {short[k][0]:5d}  {short[k][1]:5d}  {short[k][2]:5d}")
    print("\n  각은 도, 중앙값 ±사분위 범위의 절반. 자리는 절댓값 중앙값.")
    print("  '쿠션이 더한 각'이 k마다 ±로 크면 쿠션 모형(또는 당점)이, '물려받은 각'만")
    print("  크면 앞에서 쌓인 것이, 속도비가 한쪽으로 치우치면 반발·마찰 상수가 범인이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
