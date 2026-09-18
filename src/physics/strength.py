"""스트로크 강도 — 선수가 실제로 말할 수 있는 세기의 단위.

밀리미터 매 초는 큐를 잡은 사람이 쓸 수 있는 말이 아니다. 당구대 위에서
쓸 수 있는 말은 "공이 얼마나 굴러가느냐"이고, 자는 테이블 자신이다.

    강도 1 = 수구가 장축 거리(2844 mm)만큼 굴러가는 세기
    강도 n = 그 n 배

기준 샷은 하나로 고정한다. 그러지 않으면 같은 세기가 배치마다 다른 숫자가
된다. **긴 쿠션과 나란히, 짧은 쿠션 한가운데에서, 무회전 중단으로 곧게 친
수구**의 총 이동거리다. 이 공은 짧은 쿠션 사이를 오가므로 강도 1은 반대
쿠션에 닿아서 서는 세기, 강도 2는 왕복, 강도 3은 한 번 더 건너간 세기가 된다.

비스듬히 치면 쿠션을 더 많이 먹고 그만큼 덜 간다 — 강도 5로 친 공이 실제
배치에서 4.2 장축밖에 못 가는 것은 오차가 아니라 쿠션이 먹은 것이다.
강도는 **큐가 공에 준 것**의 이름이지, 공이 간 거리의 이름이 아니다.

프로 1335개 플레이의 초속을 이 자로 읽으면:

    25%  강도 3.2     50%  강도 4.4     75%  강도 5.6     90%  강도 7.3

실전은 강도 3~7에 들어간다.

이 표는 물리에 매달려 있다. 마찰을 손보면 같은 이름이 다른 세기를 가리키게
되므로 `measure()`를 다시 돌려 받아 적어야 하고, `tests/test_strength.py`가
어긋남을 잡는다. 실제로 2026-09-19에 천을 영상에 맞춰 다시 잡았을 때
강도 3은 2670 mm/s에서 1767 mm/s로 바뀌었다 — 테이블이 빨라진 만큼 같은
거리를 덜 세게 쳐도 간다.
"""

import numpy as np

LONG_RAIL_MM = 2844.0

# 기준 샷을 초속 별로 돌려 얻은 이동거리(장축 몇 개)를 강도로 되읽은 표.
# `measure()`가 이 표를 다시 만들고, tests/test_strength.py가 물리가 바뀌면
# 표가 틀어졌다고 알려 준다 — 강도는 물리에 매달린 값이라 물리만 고치고 표를
# 두면 같은 이름이 다른 세기를 가리키게 된다.
_STRENGTH = np.array([0.14, 0.32, 0.57, 1.00, 1.37, 1.84, 2.45, 3.08, 3.84, 4.46, 5.07, 5.75, 6.65, 7.89, 8.79, 9.89, 10.73, 11.91, 13.57, 15.15])
_SPEED_MM_S = np.array([300, 450, 600, 800, 1000, 1200, 1500, 1800, 2200, 2600, 3000, 3500, 4000, 4800, 5600, 6800, 8000, 10000, 14000, 20000], dtype=float)


def speed_for(strength):
    """강도를 초속(mm/s)으로. 표 밖은 양 끝 기울기로 이어 붙인다."""
    strength = float(strength)
    if strength <= _STRENGTH[0]:
        return float(_SPEED_MM_S[0] * strength / _STRENGTH[0])
    return float(np.interp(strength, _STRENGTH, _SPEED_MM_S))


def strength_of(speed):
    """초속(mm/s)을 강도로 — 프로가 친 세기를 이 자로 읽을 때 쓴다."""
    speed = float(speed)
    if speed <= _SPEED_MM_S[0]:
        return float(_STRENGTH[0] * speed / _SPEED_MM_S[0])
    return float(np.interp(speed, _SPEED_MM_S, _STRENGTH))


def measure(speeds=None, max_seconds=40.0):
    """기준 샷을 실제로 돌려 (초속, 강도) 표를 다시 만든다.

    표를 손으로 고치지 말고 이것을 돌려서 받아 적는다.
    """
    from src.physics import simulator
    from src.physics.spin import BALL_RADIUS_MM
    from src.physics.table_calibration import TABLE_WIDTH_MM

    speeds = _SPEED_MM_S if speeds is None else np.asarray(speeds, dtype=float)
    out = []
    for speed in speeds:
        # 다른 두 공은 테이블 밖으로 치워 둔다: 기준은 수구 혼자다.
        layout = {"white": np.array([BALL_RADIUS_MM + 20.0, TABLE_WIDTH_MM / 2.0]),
                  "yellow": np.array([-9000.0, -9000.0]),
                  "red": np.array([-9500.0, -9500.0])}
        shot = simulator.simulate_with_spin(layout, "white", (speed, 0.0),
                                            max_seconds=max_seconds)
        path = np.asarray(shot.paths["white"])
        gone = float(np.hypot(*np.diff(path, axis=0).T).sum())
        out.append((float(speed), gone / LONG_RAIL_MM))
    return out
