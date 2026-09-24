"""쿠션 반사 곡선을 영상에서 직접 잰다 — spin.py의 REBOUND_OUT에 대고.

파이프라인이 수구의 쿠션 자리를 이미 알고 있으므로, 이웃한 세 점으로 들어온
각과 나간 각을 바로 읽을 수 있다. 회전이 섞여 있어 한 건씩은 믿을 수 없지만
수천 건의 중앙값은 곡선 자체를 말해 준다.

2026-09-25 실측 (6,115건):

    들어온 각    나간 각(영상)   지금 표    차이
      0~10도       23.2도      12.5도   +10.7도
     10~20도       38.3도      29.4도    +8.9도
     20~30도       44.9도      39.9도    +5.0도
     30~40도       50.2도      47.8도    +2.4도
     40~50도       57.9도      54.5도    +3.4도
     50~60도       64.7도      62.2도    +2.4도
     60~70도       71.1도      69.8도    +1.3도
     70~80도       78.0도      77.0도    +1.0도
     80~90도       78.7도      85.4도    −6.7도

★**얕게 들어올 때 우리 표가 10도 넘게 좁다.** 얕은 입사는 3쿠션에서 가장 흔한
자리이므로 이 구간이 가장 무겁다.

⚠️ 영상 쪽 값에는 **회전이 섞여 있다.** 표는 무회전 기준이므로 그대로 갈아
끼우면 회전 효과를 두 번 세게 된다. 회전이 적은 건만 골라 다시 재거나,
`tools/check_after_cushion.js`로 갈아 끼운 결과를 확인해야 한다.

    ~/.venvs/carom/bin/python tools/measure_rebound.py
"""

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, W = 2844.0, 1422.0
INWARD = {"left": (1, 0), "right": (-1, 0), "bottom": (0, 1), "top": (0, -1)}


def rail_of(p):
    x, y = p
    gap = {"left": x, "right": L - x, "bottom": y, "top": W - y}
    return min(gap, key=gap.get)


def main():
    data = json.load(open(os.path.join(ROOT, "build", "app-data.json"), encoding="utf-8"))
    pairs = []
    for play in data["plays"]:
        rails = play.get("rails")
        if not rails or len(rails) < 3 or not play.get("hit"):
            continue
        points = [play["hit"]] + rails
        for k in range(1, len(points) - 1):
            a, b, c = (np.array(points[i], float) for i in (k - 1, k, k + 1))
            normal = np.array(INWARD[rail_of(points[k])], float)
            coming, going = b - a, c - b
            if np.linalg.norm(coming) < 80 or np.linalg.norm(going) < 80:
                continue
            coming /= np.linalg.norm(coming)
            going /= np.linalg.norm(going)
            if float(np.dot(coming, normal)) > 0:      # 쿠션에서 멀어지며 들어오면 버린다
                continue
            pairs.append((math.degrees(math.acos(min(1, abs(float(np.dot(coming, normal)))))),
                          math.degrees(math.acos(min(1, abs(float(np.dot(going, normal))))))))

    arr = np.array(pairs)
    print(f"영상에서 잰 쿠션 반사 {len(arr)}건\n")
    table_in = [0, 6.7, 17.8, 27.2, 38.0, 46.4, 56.0, 66.1, 79.3, 90]
    table_out = [0, 16.7, 33.7, 41.8, 50.1, 55.4, 63.0, 70.6, 80.1, 90]
    print("들어온 각   나간 각(영상)   지금 표    차이")
    for lo in range(0, 90, 10):
        keep = (arr[:, 0] >= lo) & (arr[:, 0] < lo + 10)
        if keep.sum() < 15:
            continue
        got = float(np.median(arr[keep, 1]))
        want = float(np.interp(lo + 5, table_in, table_out))
        print(f"  {lo:>2}~{lo+10:<2}도 (n={int(keep.sum()):>4})   {got:>5.1f}도    "
              f"{want:>5.1f}도   {got - want:+5.1f}도")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
