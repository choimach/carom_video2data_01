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

★**장쿠션과 단쿠션이 다르다** (선수가 먼저 말했다, 2026-09-25: "고무의 두께는
같지만 총 길이가 다르므로 반사 특징이 같지 않다"). 0~10도에서 장 14.7도 · 단
26.6도로 **12도 차이**이고, 두 곡선이 20~30도에서 교차한다. 우리는 표 한 벌로
둘 다 쓰고 있다. 자세한 것은 `ref/terms.md`의 "쿠션의 성질".

⚠️⚠️ **이것은 "진짜 반사각"이 아니라 효과적인 곡선이다** (2026-09-25에 알았다).

여기서는 **쿠션 점에서 쿠션 점까지** 긴 구간으로 각을 읽는다. 파이프라인의
`stroke.sidespin()`은 쿠션 **바로 앞뒤 짧은 구간**의 추적 경로로 읽는다.
같은 첫 쿠션 2,064건에서 둘을 맞대면 긴 구간이 체계적으로 **6~7도 넓다**:

    입사각      긴 구간     짧은 구간
     0~10도    +20.3도      13.6도
    10~20도    +16.8도      11.3도
    30~40도    +15.3도      10.9도
    50~60도    +10.8도       8.8도

그 차이가 **쿠션을 지난 뒤의 곡선**이다. 회전이 남은 공은 쿠션 뒤에도 계속
휘므로, 긴 구간으로 읽으면 나간 각이 더 넓어 보인다.

그러므로 이 곡선을 `REBOUND_OUT`에 넣는 것은 **쿠션 뒤 곡선을 반사각에
흡수시키는 것**이다. 원리로는 틀렸지만 **끝에서 끝까지의 궤적 예측은 19%
좋아진다** — 우리 물리가 쿠션 뒤 곡선을 덜 내기 때문이다. 제대로 고치려면
쿠션 뒤 곡선을 따로 모델링하고, 표에는 짧은 구간 곡선을 넣어야 한다.

⚠️ **"회전이 적은 것만 고르기"는 이 자료로 안 된다.** `spin_x`는 *거울 각*에서
얼마나 빗나갔나이고, 무회전 반사도 이미 거울에서 크게 벌어진다. 그래서
|spin_x|가 작은 것을 고르면 무회전이 아니라 **역회전**(자연스러운 벌어짐을
상쇄한 것)을 고르게 된다. 실제로 그렇게 뽑은 곡선을 넣어 보니 3쿠션 오차가
274 → 497~611 mm로 크게 나빠졌다.

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
