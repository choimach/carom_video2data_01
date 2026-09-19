"""가상 테이블 장부가 계속 읽히는지 지킨다.

조언판의 **결과 복사**가 붙이는 ```carom-feedback 덩어리와
`tools/record_feedback.py`가 읽는 형식은 같아야 한다. 한쪽만 바꾸면 붙여넣기가
조용히 "raw"로 떨어지고, 배치도 후보도 없는 글뭉치만 쌓인다 — 그러면 장부가
있는데도 되먹임은 없는, 있기 전과 같은 상태가 된다.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "src" / "visualization" / "assistant" / "assistant.html"
LEDGER = ROOT / "data" / "table_feedback.json"

from tools.record_feedback import BLOCK, add, same_round  # noqa: E402


def test_the_page_still_writes_the_block():
    page = PAGE.read_text(encoding="utf-8")
    assert '"```carom-feedback"' in page, "결과 복사가 더 이상 덩어리를 붙이지 않습니다"
    # 장부가 쓸모 있으려면 배치와 후보가 반드시 들어 있어야 한다.
    for field in ("layout:", "candidates:", "comment:", "picked:"):
        assert field in page, f"덩어리에 {field}가 없습니다"


def test_a_pasted_block_is_read_back_whole():
    one = {"at": "2026-09-19T12:00:00", "cue": "white",
           "layout": {"white": [700, 400], "yellow": [1500, 900], "red": [2300, 500]},
           "picked": "3", "candidates": [{"key": "3", "route": "뒤돌리기"}],
           "comment": "이건 옆돌리기야"}
    pasted = f"사람이 읽는 글\n\n```carom-feedback\n{json.dumps(one, ensure_ascii=False)}\n```\n"
    got = add(pasted)
    assert got == [one]


def test_a_paste_without_a_block_is_kept_rather_than_dropped():
    got = add("배치 (mm): 흰공 (700, 400)\n\n의견:\n이건 옆돌리기야")
    assert len(got) == 1 and "raw" in got[0], "덩어리 없는 붙여넣기를 버렸습니다"


def test_the_same_paste_twice_is_one_record():
    one = {"layout": {"white": [1, 2]}, "comment": "같은 말", "picked": "1"}
    assert same_round(one, dict(one))
    assert not same_round(one, dict(one, comment="다른 말"))


def test_the_ledger_on_disk_still_parses():
    kept = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert kept["rounds"], "장부가 비어 있습니다"
    # 무엇이 빠졌는지도 장부가 알고 있어야 한다.
    assert any(r.get("lost") for r in kept["rounds"]), \
        "잃어버린 기록에 대한 빈칸이 사라졌습니다"
