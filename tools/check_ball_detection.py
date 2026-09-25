"""경기마다 세 공이 몇 %의 프레임에서 잡히나 — 수율이 어디서 새는지의 첫 자.

2026-09-25. 선수가 물었다: *"모델을 개선하려면 효율을 높여야해. 지금 속도로는
너무나도 많은 file을 다운로드 해야해."* 재 보니 **내려받기가 문제가 아니었다.**

  받은 경기 61개 (경기당 7~15 GB) · 점수판이 말하는 있었을 샷 경기당 131개
  스캐너가 찾은 판 5,577 · 쓸 수 있는 판 3,069 (55%) · 모델에 들어간 판 2,466

그리고 **여덟 경기는 찾은 판이 570개인데 쓸 수 있는 판이 1개**다. 판마다 조금씩
나쁜 것이 아니라 **경기 단위로 전멸**한다 — 원인이 하나라는 뜻이다.

이 도구가 그 하나를 가리킨다. 영상이 아니라 `data/scans/*.npz`만 읽으므로
경기당 1초면 된다.

⚠️ **빠진 자리는 0이 아니라 NaN이다** (`src/pipeline.py`의 `np.full(..., np.nan)`).
   0으로 알고 세면 모든 경기가 100%로 나온다 — 2026-09-25에 한 번 그랬다.

    ~/.venvs/carom/bin/python tools/check_ball_detection.py
    ~/.venvs/carom/bin/python tools/check_ball_detection.py --bad   # 낮은 것만
"""

import collections
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANS = os.path.join(ROOT, "data", "scans")
BALL_ORDER = ("white", "yellow", "red")     # src/pipeline.py와 같은 차례
LABEL = {"white": "흰", "yellow": "노", "red": "빨"}


def rates(path):
    """(테이블이 보이는 비율, 공별 검출률, 셋 다 잡힌 비율)."""
    scan = np.load(path, allow_pickle=True)
    track = scan["track"]
    live = track[:, 0] == 1.0
    if live.sum() == 0:
        return 0.0, None, 0.0
    seen = track[live]
    have = [~np.isnan(seen[:, 1 + 2 * i]) for i in range(len(BALL_ORDER))]
    all_three = have[0] & have[1] & have[2]
    return live.mean(), [h.mean() for h in have], all_three.mean()


def main(argv):
    only_bad = "--bad" in argv
    rows = []
    for path in sorted(glob.glob(os.path.join(SCANS, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        table, have, three = rates(path)
        if have is None:
            continue
        rows.append((name, table, have, three))
    if not rows:
        print("스캔이 없습니다."); return 1

    head = "".join(f"{LABEL[c]:>6}" for c in BALL_ORDER)
    print(f"{'경기':<34}{'테이블':>7}{head}{'셋다':>7}")
    for name, table, have, three in sorted(rows, key=lambda r: r[3]):
        if only_bad and three >= 0.85:
            continue
        bar = "".join(f"{h:>6.0%}" for h in have)
        mark = "  ←쓸모없음" if three < 0.05 else ("  ←샌다" if three < 0.85 else "")
        print(f"{name[:34]:<34}{table:>7.0%}{bar}{three:>7.0%}{mark}")

    three = [r[3] for r in rows]
    print(f"\n{len(rows)}경기 · 셋 다 잡히는 비율 중앙값 {np.median(three):.0%}"
          f" · 최소 {min(three):.0%} · 최대 {max(three):.0%}")
    print(f"  5% 밑 (통째로 버려짐) {sum(1 for t in three if t < 0.05)}경기"
          f" · 85% 밑 {sum(1 for t in three if t < 0.85)}경기")

    group = collections.defaultdict(list)
    for name, _t, _h, three_ in rows:
        found = re.search(r"_([A-Z]{2,6})\d{4}$", name)
        group[found.group(1) if found else "기타"].append(three_)
    print("\n대회(제작사)별 — 방송이 다르면 임계값도 달라야 한다는 증거다")
    for key, vals in sorted(group.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {key:<8}{np.mean(vals):>6.0%}  ({len(vals)}경기)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
