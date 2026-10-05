"""How well do we know where the balls stopped?

Safety and position play are both judged on where the balls came to rest, so
this is the ruler for them (2026-10-05).

`final_mm` is where the balls were when the play's window closed - below
0.25 m/s, still rolling. `rest_mm` (src/pipeline.py `settle`) follows the scan
until they really stood still; `rest_basis` says how it was seen:
  watched   the camera stayed until they stopped
  returned  seen still after a cut, checked (balls already still did not
            move, moving balls could reach, no new stroke)
  None      not known

The next play's layout is NOT a usable answer: plays the numbering calls
adjacent often had another, unscanned play between them.

    ~/.venvs/carom/bin/python tools/check_final_positions.py
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLOURS = ("white", "yellow", "red")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset", "*.json"))):
        name = os.path.basename(path)[:-5]
        if name == "index":
            continue
        event = re.sub(r"^soop_\d+_", "", name)
        for p in json.load(open(path, encoding="utf-8"))["plays"]:
            if str(p.get("inferred")) == "True":
                continue
            fa, rest = p.get("final_mm") or {}, p.get("rest_mm")
            usable = not p.get("rejected_for")
            err = None
            if rest and all(c in fa for c in COLOURS):
                err = max(math.dist(fa[c], rest[c]) for c in COLOURS)
            rows.append(dict(event=event, basis=p.get("rest_basis"), usable=usable, err=err,
                             complete=str(p.get("complete")) == "True", scored=str(p.get("success")) == "True"))

    n = len(rows)
    print(f"플레이 {n} (화면에 나온 것)")
    for label, sel in (("전체", lambda r: True), ("쓸 수 있는 판", lambda r: r["usable"])):
        rs = [r for r in rows if sel(r)]
        w = sum(r["basis"] == "watched" for r in rs)
        t = sum(r["basis"] == "returned" for r in rs)
        print(f"  {label:<10} {len(rs):5d}판 — 멈춘 자리를 안다 {(w + t) / len(rs):4.0%}"
              f"  (끝까지 봄 {w / len(rs):.0%} · 돌아와서 봄 {t / len(rs):.0%})")
    for label, sel in (("득점", lambda r: r["scored"]), ("실패", lambda r: not r["scored"])):
        rs = [r for r in rows if r["usable"] and sel(r)]
        print(f"  쓸 수 있는 {label} {len(rs):5d}판 — 안다 {np.mean([r['basis'] is not None for r in rs]):.0%}")
    e = np.array([r["err"] for r in rows if r["err"] is not None])
    print(f"\n저장된 final_mm과 멈춘 자리의 차이 (가장 먼 공): 중앙값 {np.median(e):.0f} mm · ≤20 {np.mean(e <= 20):.0%}"
          f" · >100 {np.mean(e > 100):.0%} · >300 {np.mean(e > 300):.0%}")
    for b in ("watched", "returned"):
        eb = np.array([r["err"] for r in rows if r["basis"] == b and r["err"] is not None])
        print(f"  {b:<9} 중앙값 {np.median(eb):5.0f} mm · ≤20 {np.mean(eb <= 20):.0%}")

    by = defaultdict(list)
    for r in rows:
        by[r["event"]].append(r["basis"] is not None)
    print("\n대회별 (중계 연출이 다르다):")
    for ev, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {ev:<12} {len(v):5d}판  안다 {np.mean(v):4.0%}")


if __name__ == "__main__":
    sys.exit(main())
