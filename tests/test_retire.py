"""지운 영상을 다시 받지 않는지.

지우기와 받기는 짝이다. `tools/retire_videos.py`가 스캔 끝난 영상을 지우면
`tools/collect.py`는 그것을 "아직 안 받은 것"으로 본다 — 실제로 2026-09-20에
39경기 347 GB를 지운 직후 받을 목록이 4개에서 43개로 뛰었다. 그대로 두고
`keep_up.sh`를 한 번 더 돌렸으면 **지운 것을 전부 다시 받았을 것이다.**

한쪽만 만들어 두면 다음 라운드가 되돌린다.
"""

import json

from tools import collect, retire_videos


def test_a_retired_match_is_not_wanted_again(tmp_path, monkeypatch):
    screening = [
        {"usable": True, "vod_id": "111", "event": "A", "url": "https://x/111"},
        {"usable": True, "vod_id": "222", "event": "B", "url": "https://x/222"},
        {"usable": False, "vod_id": "333", "event": "C", "url": "https://x/333"},
    ]
    ledger = {"matches": {"soop_111_A": {"vod_id": "111", "url": "https://x/111"}}}

    screening_file = tmp_path / "_screening.json"
    screening_file.write_text(json.dumps(screening), encoding="utf-8")
    retired_file = tmp_path / "videos.json"
    retired_file.write_text(json.dumps(ledger), encoding="utf-8")

    monkeypatch.setattr(collect, "SCREENING", str(screening_file))
    monkeypatch.setattr(collect, "RETIRED", str(retired_file))
    monkeypatch.setattr(collect, "UNREADABLE", str(tmp_path / "nothing.json"))

    ids = [row["vod_id"] for row in collect.wanted()]
    assert "111" not in ids, "지운 경기를 다시 받으려 합니다"
    assert "222" in ids, "아직 안 받은 경기가 목록에서 빠졌습니다"
    assert "333" not in ids, "쓸 수 없다고 판정된 경기가 목록에 있습니다"


def test_no_ledger_means_nothing_is_retired(tmp_path, monkeypatch):
    """장부가 없을 때 전부 지운 것으로 읽으면 아무것도 안 받게 된다."""
    monkeypatch.setattr(collect, "RETIRED", str(tmp_path / "없는파일.json"))
    assert collect.retired() == set()


def test_the_ledger_keeps_an_address_for_every_retired_match():
    """주소 없이 지우면 영영 못 되찾는다 — 실제 장부를 본다."""
    import os
    if not os.path.exists(retire_videos.LEDGER):
        return
    kept = json.load(open(retire_videos.LEDGER, encoding="utf-8"))
    without = [name for name, row in kept.get("matches", {}).items() if not row.get("url")]
    assert not without, f"주소 없이 지운 경기가 있습니다: {without}"
