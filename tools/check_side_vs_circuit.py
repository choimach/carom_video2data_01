"""회전 방향과 도는 방향이 실제로 같이 가는지 영상에서 센다.

선수: "꼭 그런 건 아니야. 하지만 일반적으로 공이 당구대에서 왼쪽으로 돌아야
할 때는 왼쪽 당점, 오른쪽으로 돌 때는 오른쪽 당점을 주는 게 일반적이야.
물론 아닌 경우도 없지는 않아."

이건 규칙이 아니라 **경향**이다 — 본인이 그렇게 말했고, 규칙으로 굳히면 안
된다. 대신 두 가지에 쓴다.

* 얼마나 강한 경향인지 세어 둔다. "일반적"이 70%인지 95%인지는 다른 이야기다.
* **부호 검사에 쓴다.** 좌우 회전 부호가 뒤집혀 있으면 이 수가 반대로 나온다.
  `tools/check_side_sign.py`가 위치 오차로 판정하는 것을, 이쪽은 선수의 감각
  으로 판정한다 — 서로 다른 자로 같은 것을 재는 셈이라, 둘이 어긋나면 어딘가
  더 볼 곳이 있다는 뜻이다.

`english_side()`는 양수가 오른쪽(위에서 보아 시계 방향), `circuit_sign()`은
양수가 오른쪽으로 도는 것이다. 경향이 사실이라면 둘의 부호가 대체로 같아야
한다.

    ~/.venvs/carom/bin/python tools/check_side_vs_circuit.py
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import analyse, circuit_sign, english_side, load_scan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=None)
    args = parser.parse_args()

    agree, total = 0, 0
    by_route = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))[:args.files]:
        data = load_scan(path)
        result = analyse(data)
        for inning in result["innings"]:
            for shot in inning.shots:
                if shot.inferred or getattr(shot, "rejections", None):
                    continue
                events = (shot.verdict or {}).get("events") or []
                side = english_side(shot, data["positions"], events)
                circuit = circuit_sign(shot, data["positions"])
                if side is None or circuit is None or abs(side) < 1e-9:
                    continue
                same = (side > 0) == (circuit > 0)
                agree += same
                total += 1
                route = getattr(shot, "route", None) or "(이름 없음)"
                keep = by_route.setdefault(route, Counter())
                keep["same" if same else "opposite"] += 1

    if not total:
        print("둘 다 측정된 플레이를 찾지 못했습니다")
        return 1

    share = agree / total
    print(f"회전 방향과 도는 방향이 둘 다 측정된 플레이 {total}개\n")
    print(f"  같은 쪽 (도는 쪽으로 회전)   {agree:>4}  {share:.0%}")
    print(f"  반대 쪽                      {total - agree:>4}  {1 - share:.0%}\n")

    rows = sorted(by_route.items(), key=lambda kv: -sum(kv[1].values()))
    print("유형별")
    for route, count in rows:
        n = sum(count.values())
        if n < 10:
            continue
        print(f"  {route:<10} n={n:>4}   같은 쪽 {count['same'] / n:.0%}")

    print()
    if share > 0.6:
        print(f"→ 선수의 말대로입니다 — {share:.0%}가 도는 쪽으로 회전을 줍니다.")
        print("  경향이지 규칙은 아니므로, 순위에 넣지 않고 검사에만 씁니다.")
    elif share < 0.4:
        print(f"→ 거꾸로 나옵니다 ({share:.0%}). 부호가 어딘가 뒤집혀 있습니다 —")
        print("  english_side()이거나 circuit_sign()이거나, 둘 중 하나입니다.")
    else:
        print(f"→ 반반입니다 ({share:.0%}). 경향이라고 부를 만한 것이 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
