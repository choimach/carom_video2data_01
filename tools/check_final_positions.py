"""How well do we know where the balls stopped?

Safety and position play are both judged on where the balls came to rest, so
this is the ruler for them (2026-10-05).

Each play records `final_mm` at the frame its window closed. The window closes
when every ball is below REST_SPEED_MS (0.25 m/s) for 0.35 s - right for
splitting plays, but a ball at 0.2 m/s still rolls on. This follows the scan
past the window to where the balls really stood still (each ball within
STILL_MM over STILL_SECONDS) and compares:

  final_mm <-> true rest   how wrong the stored end position is

The next play's layout is NOT a usable answer: in 2026-10-05 runs the plays the
numbering called adjacent often had another, unscanned play between them.

    ~/.venvs/carom/bin/python tools/check_final_positions.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pipeline import load_scan  # noqa: E402

COLOURS = ("white", "yellow", "red")
STILL_MM = 5.0
STILL_SECONDS = 0.5


def true_rest(pos, live, start, stop, fps):
    """First stretch after `start` where all three balls are seen and still.

    Returns (frame, {colour: (x, y)}) or (None, None) if the camera never
    showed them still before `stop`.
    """
    span = max(2, int(STILL_SECONDS * fps))
    xy = np.stack([pos[c][start:stop] for c in COLOURS], axis=1)       # (n, 3, 2)
    ok = live[start:stop] & np.isfinite(xy).all(axis=(1, 2))
    n = len(ok)
    i = 0
    while i + span <= n:
        if not ok[i:i + span].all():
            i += 1
            continue
        win = xy[i:i + span]
        if (np.linalg.norm(win - win[0], axis=2) <= STILL_MM).all():
            med = np.median(win, axis=0)
            return start + i, {c: tuple(med[k]) for k, c in enumerate(COLOURS)}
        i += 1
    return None, None


def still_at(pos, frame, fps):
    """Balls that had not moved in the half second before `frame`."""
    w = int(0.5 * fps)
    out = []
    for c in COLOURS:
        seg = pos[c][max(0, frame - w):frame + 1]
        seg = seg[np.isfinite(seg).all(axis=1)]
        if len(seg) > 3 and np.linalg.norm(seg - seg[0], axis=1).max() < STILL_MM:
            out.append(c)
    return out


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json"))):
        name = os.path.basename(path)[:-5]
        scan = os.path.join(ROOT, "data", "scans", name + ".npz")
        if name == "index" or not os.path.exists(scan):
            continue
        s = load_scan(scan)
        pos, live, fps = s["positions"], s["live"], s["fps"]
        plays = sorted(json.load(open(path, encoding="utf-8"))["plays"], key=lambda p: int(p["start_frame"]))
        starts = [int(p["start_frame"]) for p in plays]
        for a in plays:
            end, fa = int(a["end_frame"]), a.get("final_mm") or {}
            if not all(c in fa for c in COLOURS):
                continue
            later = [x for x in starts if x > end]
            stop = min(len(live), end + int(60 * fps), later[0] if later else len(live))
            frame, rest = true_rest(pos, live, end, stop, fps)
            still = still_at(pos, end, fps)
            cut = frame is not None and (~live[end:frame]).sum() / fps >= 0.5
            # After a cut another play may have happened unseen. Balls that were
            # already still when the window closed must not have moved.
            agrees = None if frame is None or not still else all(math.dist(fa[c], rest[c]) <= 20 for c in still)
            rows.append(dict(
                complete=str(a.get("complete")) == "True" and str(a.get("inferred")) != "True",
                found=frame is not None, cut=cut, still=len(still), agrees=agrees,
                err=max(math.dist(fa[c], rest[c]) for c in COLOURS) if rest else None,
                late=(frame - end) / fps if frame is not None else None,
            ))

    n = len(rows)
    found = [r for r in rows if r["found"]]
    direct = [r for r in found if not r["cut"]]
    after = [r for r in found if r["cut"]]
    print(f"플레이 {n}")
    print(f"다음 판 전에 세 공이 멈춘 것이 보인다: {len(found) / n:.0%}  "
          f"(카메라가 끊기지 않고: {len(direct) / n:.0%} · 중계가 다른 화면으로 갔다 돌아와서: {len(after) / n:.0%})")
    for k in (0, 1, 2):
        rk = [r for r in after if r["still"] == k]
        if rk and k:
            print(f"  끊긴 뒤, 창이 닫힐 때 멈춰 있던 공 {k}개: {len(rk)}판 — 그 공이 그대로인 판 "
                  f"{np.mean([bool(r['agrees']) for r in rk]):.0%} (아니면 사이에 놓친 판)")
        elif rk:
            print(f"  끊긴 뒤, 창이 닫힐 때 세 공 다 움직이던 판: {len(rk)} — 이 방법으로는 확인할 수 없다")
    trusted = direct + [r for r in after if r["agrees"]]
    e = np.array([r["err"] for r in trusted])
    print(f"믿을 수 있는 진짜 멈춤: {len(trusted) / n:.0%}")
    print(f"  거기서 저장된 final_mm의 오차 (가장 먼 공): 중앙값 {np.median(e):.0f} mm · ≤20 {np.mean(e <= 20):.0%}"
          f" · >100 {np.mean(e > 100):.0%} · >300 {np.mean(e > 300):.0%}")
    for name, sel in (("complete", lambda r: r["complete"]), ("멈춤 못 봄", lambda r: not r["complete"])):
        rs = [r for r in rows if sel(r)]
        d = [r for r in rs if r["found"] and not r["cut"]]
        if d:
            de = np.array([r["err"] for r in d])
            print(f"  [{name}] {len(rs)}판, 카메라가 멈출 때까지 머문 판 {len(d) / len(rs):.0%} — 저장 오차 중앙값 "
                  f"{np.median(de):.0f} mm, 창이 닫히고 {np.median([r['late'] for r in d]):.1f}초 뒤에 멈춤")


if __name__ == "__main__":
    sys.exit(main())
