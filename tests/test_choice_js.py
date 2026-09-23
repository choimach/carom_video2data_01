"""배운 점수의 특징 계산이 두 언어에서 같은지.

`tools/learn_choices.py`가 맞춘 가중치를 조언판이 쓴다. 특징의 차례나 모양이
한 쪽에서만 바뀌면 **가중치가 엉뚱한 값에 곱해진다** — 화면은 멀쩡히 순위를
내놓고, 그 순위는 아무 근거가 없다. 조용한 실패라 시험으로만 잡힌다.

node가 없으면 건너뛴다.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.learn_choices import NAMES, features

CHOICE_JS = Path(__file__).resolve().parent.parent / "src" / "visualization" / "assistant" / "choice.js"
WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "choice_weights.json"

BRANCHES = [
    {"room": 0.25, "thickness": 0.1, "strength": 3.5, "rails": 3,
     "pushed": 500, "lines": 1, "side": 0.0, "up": 0.0, "chosen": 0, "rate": 0.5},
    {"room": 4.0, "thickness": 0.55, "strength": 5.5, "rails": 4,
     "pushed": 4200, "lines": 80, "side": 1.73, "up": 1.0, "chosen": 7, "rate": 0.62},
    {"room": 12.5, "thickness": 0.95, "strength": 7.0, "rails": 6,
     "pushed": 9000, "lines": 300, "side": -3.0, "up": -2.0, "chosen": 1, "rate": 0.33},
    {"room": 1.0, "thickness": 0.3, "strength": 2.0, "rails": 5,
     "pushed": 0, "lines": 12, "side": 0.0, "up": 2.0, "chosen": 0, "rate": 0.5},
]


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_both_languages_build_the_same_features():
    script = f"""
    const CHOICE = require({json.dumps(str(CHOICE_JS))});
    const branches = {json.dumps(BRANCHES)};
    console.log(JSON.stringify(branches.map((b) => CHOICE.features(b))));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    theirs = json.loads(done.stdout)

    for branch, got in zip(BRANCHES, theirs):
        mine = features(branch)
        assert got == pytest.approx(mine, rel=1e-9), \
            f"{branch} → 파이썬 {mine} · JS {got}"


@pytest.mark.skipif(not shutil.which("node"), reason="node가 없습니다")
def test_the_names_line_up_with_the_weights():
    """차례가 어긋나면 가중치가 엉뚱한 특징에 곱해진다."""
    script = f"""
    const CHOICE = require({json.dumps(str(CHOICE_JS))});
    console.log(JSON.stringify(CHOICE.NAMES));
    """
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == NAMES, "특징 이름의 차례가 두 쪽에서 다릅니다"

    if WEIGHTS.exists():
        learned = json.loads(WEIGHTS.read_text(encoding="utf-8"))
        assert learned["names"] == NAMES, "저장된 가중치의 이름이 지금 코드와 다릅니다"
        assert len(learned["weights"]) == len(NAMES)


def test_the_page_takes_its_weights_from_the_file():
    """손으로 베껴 두면 다시 재고도 화면이 안 바뀐다."""
    page = (CHOICE_JS.parent / "assistant.html").read_text(encoding="utf-8")
    assert "__WEIGHTS__" in page, "가중치 자리가 사라졌습니다"
    assert "learnedScore" in page and "CHOICE.score" in page
