"""득점 확률이 파이썬과 조언판에서 같은 값인지 붙들어 둔다.

CLAUDE.md §5: 같은 모형이 두 언어에 있으면 **한쪽만 고쳤을 때 조용히 어긋나고,
증상은 "그럴듯한 틀린 숫자" 하나뿐이다.** 확률은 화면에 퍼센트로 나가므로
어긋나도 아무도 모른다 — 64%가 61%가 되어도 그럴듯하다.

파이썬 쪽: `tools/score_probability.py`의 `features()`와 무게 (data/probability_weights.json)
JS 쪽:   `src/visualization/assistant/assistant.html`의 `chanceOf()`

두 축의 **차례와 모양**이 같아야 한다.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PAGE = os.path.join(ROOT, "src", "visualization", "assistant", "assistant.html")
WEIGHTS = os.path.join(ROOT, "data", "probability_weights.json")

BRANCHES = [
    {"room": 3.0, "lines": 40, "thickness": 0.6, "strength": 3, "rails": 3,
     "side": 0.5, "rate": 0.8, "chosen": 12},
    {"room": 0.6, "lines": 12, "thickness": 0.35, "strength": 4, "rails": 3,
     "side": 1.5, "rate": 0.55, "chosen": 4},
    {"room": 0.05, "lines": 2, "thickness": 0.1, "strength": 7, "rails": 5,
     "side": -3, "rate": 0.33, "chosen": 0},
    # 빈 값들 — 조언판에서 실제로 온다 (이웃이 하나도 없는 후보가 45%다)
    {"room": 0, "lines": 0, "thickness": 0, "strength": 0, "rails": 0,
     "side": 0, "rate": None, "chosen": 0},
]


def python_chance(branch, table):
    from score_probability import features
    x = features(branch)
    mean = [float(v) for v in table["mean"]]
    spread = [float(v) for v in table["spread"]]
    total = float(table["bias"])
    for i, value in enumerate(x):
        total += ((value - mean[i]) / spread[i]) * float(table["weights"][i])
    return 1.0 / (1.0 + math.exp(-total))


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 없습니다")
@pytest.mark.skipif(not os.path.exists(WEIGHTS),
                    reason="tools/score_probability.py를 먼저 돌리세요")
def test_chance_matches_between_python_and_the_page(tmp_path):
    table = json.load(open(WEIGHTS, encoding="utf-8"))
    page = open(PAGE, encoding="utf-8").read()
    found = re.search(r"function chanceOf\(branch\) \{[\s\S]*?\n  \}", page)
    assert found, "assistant.html에서 chanceOf()를 찾지 못했습니다"

    script = tmp_path / "chance.js"
    script.write_text(
        f"const CHANCE = {json.dumps(table)};\n"
        + found.group(0).replace("function chanceOf", "function chanceOf")
        + "\nconsole.log(JSON.stringify("
        + json.dumps(BRANCHES)
        + ".map(chanceOf)));\n",
        encoding="utf-8",
    )
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    theirs = json.loads(out.stdout)
    mine = [python_chance(b, table) for b in BRANCHES]
    assert len(theirs) == len(mine)
    for branch, a, b in zip(BRANCHES, mine, theirs):
        assert a == pytest.approx(b, abs=1e-9), f"{branch}: 파이썬 {a} vs 조언판 {b}"


@pytest.mark.skipif(not os.path.exists(WEIGHTS),
                    reason="tools/score_probability.py를 먼저 돌리세요")
def test_the_page_does_not_carry_copied_weights():
    """무게는 build_assistant.py가 넣는다. 베끼면 다시 재고도 옛 숫자가 나간다."""
    page = open(PAGE, encoding="utf-8").read()
    assert "__CHANCE__" in page, "assistant.html이 __CHANCE__ 자리를 잃었습니다"
    table = json.load(open(WEIGHTS, encoding="utf-8"))
    assert str(table["weights"][0]) not in page, "무게가 손으로 베껴져 있습니다"


@pytest.mark.skipif(not os.path.exists(WEIGHTS),
                    reason="tools/score_probability.py를 먼저 돌리세요")
def test_the_fit_is_worth_showing():
    """가르지 못하는 숫자를 확률이라고 화면에 내보내지 않는다."""
    table = json.load(open(WEIGHTS, encoding="utf-8"))
    assert table["auc"] > 0.58, f"AUC {table['auc']} — 동전 던지기와 다를 바 없습니다"
    assert table["plays"] > 500
