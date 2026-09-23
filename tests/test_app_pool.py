"""조언판의 이웃 풀이 유형을 통째로 빠뜨리지 않는지.

한때 `OUTCOME_IN_NAME`(대회전·되돌아오기·횡단)이 풀에서 290판을 통째로
빼고 있었다. 그 목록의 이유는 **유형 이름 맞히기 실험**에서 "이름이 결과를
담고 있어 채점할 수 없다"는 것이지, 이 풀과는 상관이 없다.

값은 조용히 틀렸다. 화면은 세 유형 후보에 늘 "프로 0명"을 주고, 학습 쪽은
같은 후보에 표를 주었다 — 가중치가 배운 분포와 화면이 먹이는 분포가 달랐다.
숫자는 멀쩡해 보이고 순위만 틀린다. 시험으로만 잡힌다.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "model.json"
POOL = ROOT / "build" / "app-data.json"


@pytest.mark.skipif(not (MODEL.exists() and POOL.exists()),
                    reason="자료가 아직 없습니다")
def test_no_route_is_missing_from_the_pool():
    rows = json.loads(MODEL.read_text(encoding="utf-8"))
    common = {r for r, n in Counter(x["route"] for x in rows if x.get("route")).items()
              if n >= 10}
    pool = {p["route"] for p in json.loads(POOL.read_text(encoding="utf-8"))["plays"]}
    missing = sorted(common - pool)
    assert not missing, f"프로가 열 판 넘게 친 유형이 풀에 없습니다: {missing}"
