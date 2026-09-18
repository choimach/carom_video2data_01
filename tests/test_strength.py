"""강도가 물리에 매달려 있는지 지킨다.

강도는 이름이 아니라 측정값이다 — "강도 3"은 수구가 장축 세 개만큼 굴러가는
세기이고, 그 대응은 시뮬레이터가 정한다. 마찰 하나만 손대도 같은 이름이 다른
세기를 가리키게 되므로, 표와 물리가 어긋나면 여기서 걸린다.

JS 쪽 표도 같은 값이어야 한다. 앱은 그 표로 세기를 고르고 화면에 적는다.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.physics import strength

SIM_JS = Path(__file__).resolve().parent.parent / "src" / "visualization" / "assistant" / "sim.js"


def test_the_table_still_matches_the_simulator():
    for speed, measured in strength.measure():
        listed = strength.strength_of(speed)
        assert abs(listed - measured) < 0.02, (
            f"{speed:.0f} mm/s: 표는 강도 {listed:.2f}, 실제로 돌려 보니 {measured:.2f} — "
            "물리가 바뀌었으면 strength.measure()를 돌려 표를 다시 적으세요")


def test_strength_one_is_one_long_rail():
    """정의 그 자체: 강도 1은 장축 한 번이다."""
    speed = strength.speed_for(1.0)
    (_, gone), = strength.measure([speed])
    assert abs(gone - 1.0) < 0.05, f"강도 1이 장축 {gone:.2f}개를 갔습니다"


def test_it_reads_back_the_way_it_was_written():
    for value in (0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0):
        assert abs(strength.strength_of(strength.speed_for(value)) - value) < 0.01


def test_professionals_play_between_two_and_five():
    """이 자가 실전 범위에 맞는지. 맞지 않으면 눈금이 잘못 잡힌 것이다."""
    import numpy as np
    data = json.loads((Path(__file__).resolve().parent.parent / "build" / "app-data.json")
                      .read_text(encoding="utf-8")) if (
        Path(__file__).resolve().parent.parent / "build" / "app-data.json").exists() else None
    if data is None:
        pytest.skip("build/app-data.json이 없습니다 — tools/app_data.py를 먼저 돌리세요")
    speeds = [p["speed"] for p in data["plays"] if p.get("speed")]
    read = np.array([strength.strength_of(v) for v in speeds])
    # 눈금이 게임이 벌어지는 자리에 놓였는지만 본다. 좁게 잡으면 물리를
    # 고칠 때마다 이 선이 먼저 걸려서, 정작 확인해야 할 것을 가린다.
    middle = float(np.percentile(read, 50))
    assert 2.5 < middle < 6.5, f"프로의 중앙값이 강도 {middle:.1f}입니다"


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_the_javascript_carries_the_same_table():
    listed = re.search(r"const STRENGTH = \[(.*?)\];", SIM_JS.read_text(encoding="utf-8"),
                       re.S).group(1)
    js_strengths = [float(x) for x in listed.replace("\n", "").split(",")]
    assert js_strengths == pytest.approx(list(strength._STRENGTH)), "JS 표가 파이썬과 다릅니다"

    script = f"""
    const fs = require('fs');
    const SIM = eval(fs.readFileSync({json.dumps(str(SIM_JS))}, 'utf8') + '\\nSIM;');
    console.log(JSON.stringify([1, 2, 2.5, 3.5, 4.5, 5.5, 7].map((s) => SIM.speedFor(s))));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    for value, js_speed in zip([1, 2, 2.5, 3.5, 4.5, 5.5, 7], json.loads(done.stdout)):
        assert js_speed == pytest.approx(strength.speed_for(value), rel=1e-9), \
            f"강도 {value}: JS는 {js_speed:.0f}, 파이썬은 {strength.speed_for(value):.0f} mm/s"
