"""두껍게 맞힌 뒤 수구가 시뮬에서 너무 느린가 — 충돌 직후 속도를 영상과 견준다 (2026-10-18).

선수의 배치 (씨앗 599488784): 흰공을 두껍게(0.65) 상단 2시 강도 5로 치면 시뮬의 수구는 장-단-장을
다 돌고 빨간공 280 mm 앞에 선다. 선수는 "강도 5 정도로" 그 길을 친다. 10-09의 "온 두께로 밀어치면
수구가 더 가야"와 같은 쪽이다. 선수 값은 느낌이므로 맞추지 않는다 — **프로 영상으로 잰다.**

자: 충돌 **직후** 수구 속도 / 충돌 **직전** 수구 속도 (`r`). 비율이라 속도 읽기의 치우침이 지워지고,
첫 쿠션 전 짧은 창만 보므로 영상 창이 일찍 닫히는 것과 상관없다. 당점은 영상에 안 찍히므로
시뮬은 상하 당점 여럿으로 쳐서 **범위**를 낸다 — 실제 r이 그 범위 위로 자주 나가면 시뮬이 느린 것이다.

두께는 양쪽 모두 같은 식으로 잰다: 충돌 직전 수구 방향과 충돌 직후 1적구 방향 사이 각 θ,
두께 = 1 − sin θ (`ref/terms.md`의 1 − e/D).

    ~/.venvs/carom/bin/python tools/check_thick_speed.py            # 스캔 전부
    ~/.venvs/carom/bin/python tools/check_thick_speed.py 5          # 앞 5개만
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.physics.simulator import simulate_with_spin  # noqa: E402
from src.physics.table_calibration import TABLE_LENGTH_MM as L, TABLE_WIDTH_MM as W  # noqa: E402
from src.pipeline import analyse, load_scan  # noqa: E402
from tools.validate_simulator import opening  # noqa: E402

BALLS = ("white", "yellow", "red")
VERTICAL = (-2.0, 0.0, 1.0, 2.0, 3.0)   # 상하 당점 (팁), 좌우 0
RAIL_MM = 60.0                          # 공 중심이 쿠션에서 이만큼 안이면 쿠션 근처로 친다
BEFORE = 4                              # 충돌 전 속도: 프레임 수
SKIP = 2                                # 충돌 뒤 건너뛸 프레임 (공이 붙어 있는 동안)
AFTER_S = 0.25                          # 충돌 뒤 창 길이 (초) — 첫 쿠션 전까지로 줄인다
LATER = (0.5, 1.0, 1.5, 2.0, 3.0)       # 충돌 뒤 굴러간 길이를 재는 시각 (초)


def near_rail(p):
    return p[0] < RAIL_MM or p[0] > L - RAIL_MM or p[1] < RAIL_MM or p[1] > W - RAIL_MM


def measure(cue_path, obj_path, contact, fps, span=None):
    """충돌 프레임 `contact` 둘레에서 (r, 두께, 창 길이). 못 재면 None."""
    cue_path = np.asarray(cue_path, float)
    obj_path = np.asarray(obj_path, float)
    if contact - BEFORE - 1 < 0:
        return None
    pre = cue_path[contact - BEFORE - 1:contact]
    if not np.isfinite(pre).all():
        return None
    v_before = np.hypot(*(pre[-1] - pre[0])) / (len(pre) - 1) * fps
    heading = (pre[-1] - pre[0]) / max(np.hypot(*(pre[-1] - pre[0])), 1e-9)
    start = contact + SKIP
    if span is None:
        span = int(round(AFTER_S * fps))
        for k in range(start, min(start + span + 1, len(cue_path))):
            if not np.isfinite(cue_path[k]).all() or near_rail(cue_path[k]):
                span = k - start - 1
                break
    if span < 3 or start + span >= len(cue_path) or start + 4 >= len(obj_path):
        return None
    post = cue_path[start:start + span + 1]
    if not np.isfinite(post).all():
        return None
    v_after = np.hypot(*(post[-1] - post[0])) / span * fps
    o = obj_path[contact + 1:contact + 5]
    if not np.isfinite(o).all() or np.hypot(*(o[-1] - o[0])) < 5:
        return None
    od = (o[-1] - o[0]) / np.hypot(*(o[-1] - o[0]))
    cos = float(np.clip(heading @ od, -1, 1))
    if cos <= 0:
        return None
    thick = 1.0 - np.sqrt(1 - cos * cos)
    return {"r": v_after / max(v_before, 1e-9), "thick": float(thick), "span": span, "v_before": float(v_before)}


def collect(limit=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:limit]:
        data = load_scan(path)
        result = analyse(data)
        fps = result["fps"]
        pos = data["positions"]
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred:
                    continue
                events = (shot.verdict or {}).get("events") or []
                aimed = next((e.detail for e in events if e.kind == "ball"), None)
                if aimed is None:
                    continue
                cue = shot.cue_ball
                lo, hi = shot.start_frame, min(shot.end_frame, shot.start_frame + 240)
                d = np.hypot(*(pos[cue][lo:hi] - pos[aimed][lo:hi]).T)
                if not np.isfinite(d).any():
                    continue
                contact = lo + int(np.nanargmin(d))
                if np.nanmin(d) > 90:
                    continue
                real = measure(pos[cue], pos[aimed], contact, fps)
                if real is None:
                    continue
                start = opening(shot, pos, fps)
                if start is None:
                    continue
                layout, cue_, velocity, frame = start
                horizon = (contact - frame) / fps + AFTER_S + 0.3
                sims = {}
                for v in VERTICAL:
                    played = simulate_with_spin(layout, cue_, velocity, tips_side=0.0, tips_vertical=v,
                                                fps=fps, max_seconds=horizon)
                    hit = next((f for f, kind, det in played.events if kind == "ball"), None)
                    first = next((det for f, kind, det in played.events if kind == "ball"), None)
                    if hit is None or first != aimed:
                        continue
                    m = measure(played.paths[cue_], played.paths[aimed], hit, fps, span=real["span"])
                    if m is None or abs(m["thick"] - real["thick"]) > 0.1:
                        continue
                    if any(kind == "cushion" and hit < f <= hit + SKIP + real["span"] for f, kind, _ in played.events):
                        continue
                    sims[v] = m["r"]
                if not sims:
                    continue
                # 충돌 뒤 몇 초 동안 수구가 **굴러간 길이** (호 길이) — 방향이 달라도 견줄 수 있다.
                # 짧은 창(r)이 맞아도, 느려진 공의 쿠션 손실·구름 저항이 틀리면 여기서 벌어진다.
                travel = {}
                if 2.0 in sims:
                    played = simulate_with_spin(layout, cue_, velocity, tips_side=0.0, tips_vertical=2.0,
                                                fps=fps, max_seconds=(contact - frame) / fps + max(LATER) + 0.2)
                    hit = next(f for f, kind, _ in played.events if kind == "ball")
                    mine = np.asarray(played.paths[cue_], float)
                    seen = np.asarray(pos[cue][contact:shot.end_frame + 1], float)
                    # 추적 떨림은 호 길이를 늘린다 — 0.1초 간격으로 솎아 잰다 (양쪽 같게)
                    STEP = max(1, int(round(fps / 10)))
                    for sec in LATER:
                        k = int(round(sec * fps)) // STEP * STEP
                        if k >= len(seen) or not np.isfinite(seen[:k + 1]).all():
                            break
                        real_len = float(np.hypot(*np.diff(seen[:k + 1:STEP], axis=0).T).sum())
                        seg = mine[hit:hit + k + 1]
                        sim_len = float(np.hypot(*np.diff(seg[::STEP], axis=0).T).sum()) if len(seg) > 1 else 0.0
                        travel[sec] = (real_len, sim_len)
                rows.append({"scan": os.path.basename(path), "thick": real["thick"], "r": real["r"],
                             "v_before": real["v_before"], "span": real["span"], "sim": sims, "travel": travel})
        print(f"  {os.path.basename(path)}  누적 {len(rows)}판", file=sys.stderr)
    return rows


def main(argv):
    limit = int(argv[0]) if argv else None
    rows = collect(limit)
    out = os.path.join(ROOT, "data", "thick_speed.json")
    json.dump(rows, open(out, "w"), ensure_ascii=False)
    print(f"{len(rows)}판 -> {out}\n")
    print("두께        판   실제 r 중앙   시뮬 r 중앙 (상하 " + " / ".join(f"{v:+g}" for v in VERTICAL) + ")"
          "   실제 > 시뮬 최대   실제 < 시뮬 최소")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        b = [x for x in rows if lo <= x["thick"] < hi]
        if not b:
            continue
        sim_med = []
        for v in VERTICAL:
            vals = [x["sim"][str(v)] if str(v) in x["sim"] else x["sim"].get(v) for x in b]
            vals = [q for q in vals if q is not None]
            sim_med.append(f"{np.median(vals):.2f}" if vals else "  - ")
        above = np.mean([x["r"] > max(x["sim"].values()) for x in b])
        below = np.mean([x["r"] < min(x["sim"].values()) for x in b])
        print(f"{lo:.1f}~{min(hi, 1):.1f}  {len(b):5d}   {np.median([x['r'] for x in b]):.2f}        "
              + " / ".join(sim_med) + f"        {above:5.0%}            {below:5.0%}")
    travel_table(rows)


def travel_table(rows):
    print("\n충돌 뒤 수구가 굴러간 길이, 시뮬(상단 2팁) ÷ 실제 — 중앙값 (판 수). 1보다 작으면 시뮬이 덜 간다")
    print("두께       " + "".join(f"{s:>12}" for s in LATER))
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 1.01)):
        b = [x for x in rows if lo <= x["thick"] < hi]
        cells = []
        for sec in LATER:
            vals = [t[1] / t[0] for x in b for k, t in x.get("travel", {}).items()
                    if float(k) == sec and t[0] > 50]
            cells.append(f"{np.median(vals):.2f} ({len(vals)})" if vals else "-")
        print(f"{lo:.2f}~{min(hi, 1):.2f}  " + "".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1:])
