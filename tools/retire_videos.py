"""스캔이 끝난 영상을 URL만 남기고 지운다.

선수가 처음부터 정한 방식이다: *"완료된 경기 video는 url정보만 남기고 지우면
된다. 최종 data 크기는 300MB를 넘지 않을 듯."* 한 경기가 7~15 GB이고 스캔
결과(`data/scans/*.npz`)는 1 MB도 안 되므로, 영상을 들고 있을 이유는 다시 스캔할
일이 있을 때뿐이다.

지우기 전에 반드시 **세** 가지를 확인한다:

* **스캔이 끝났는가.** 안 끝난 영상을 지우면 30분짜리 다운로드를 다시 해야 한다.
* ★**플레이가 실제로 나왔는가.** 스캔이 "끝났다"는 것과 "쓸모가 있었다"는 것은
  다르다. 2026-09-15에 적어 둔 규칙 — *"수율이 낮은 경기는 검출기를 고칠 때까지
  남겨 둔다. 다시 스캔하려면 영상이 있어야 하고 SOOP VOD는 사라질 수 있다"* —
  을 읽지 않고 2026-09-20에 **0판짜리 다섯 경기 47 GB를 지웠다** (AKR2025 하나와
  T3WC2026 넷). 주소는 남겼으니 되찾을 수는 있지만, 그게 규칙이 있었던 이유다.
* **URL이 git에 남았는가.** 지금 URL은 `data/_screening.json`에만 있는데 `data/`는
  git에 올라가지 않는다. 그래서 이 도구가 먼저 `data/videos.json`에 옮겨 적고
  (`git add -f`로 추적), 그 다음에 지운다. **주소를 잃으면 영상을 잃는 것이다.**

기본은 **말만 하고 지우지 않는다.** 실제로 지우려면 `--delete`.

    ~/.venvs/carom/bin/python tools/retire_videos.py
    ~/.venvs/carom/bin/python tools/retire_videos.py --delete
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VIDEOS = os.path.join(ROOT, "data", "videos")
MODEL = os.path.join(ROOT, "data", "model.json")
SCANS = os.path.join(ROOT, "data", "scans")
SCREENING = os.path.join(ROOT, "data", "_screening.json")
LEDGER = os.path.join(ROOT, "data", "videos.json")

NOTE = ("내려받았던 경기의 주소. 영상 본체는 스캔이 끝나면 지우고 이것만 남긴다 - "
        "경기당 7~15 GB이고 스캔 결과는 1 MB도 안 된다. 다시 필요하면 여기 url로 "
        "받는다. data/는 git에 안 올라가므로 이 파일만 git add -f로 추적한다.")


def ledger():
    if not os.path.exists(LEDGER):
        return {"note": NOTE, "matches": {}}
    kept = json.load(open(LEDGER, encoding="utf-8"))
    kept.setdefault("note", NOTE)
    kept.setdefault("matches", {})
    return kept


def plays_per_match():
    """경기마다 모델 자료에 남은 플레이 수. 0이면 아직 지우지 않는다."""
    if not os.path.exists(MODEL):
        return {}
    rows = json.load(open(MODEL, encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("plays", rows.get("rows", []))
    return Counter(r["match"] for r in rows)


def screened():
    if not os.path.exists(SCREENING):
        return {}
    return {str(r["vod_id"]): r for r in json.load(open(SCREENING, encoding="utf-8"))
            if r.get("vod_id")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="실제로 지운다")
    args = parser.parse_args()

    known = screened()
    scans = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SCANS, "*.npz"))}
    yields = plays_per_match()
    kept = ledger()

    ready, held = [], []
    for path in sorted(glob.glob(os.path.join(VIDEOS, "*.mp4"))):
        stem = os.path.basename(path)[:-4]
        size = os.path.getsize(path) / 2 ** 30
        vod = stem.split("_")[1] if stem.startswith("soop_") else None
        row = known.get(vod or "")
        why = []
        if stem not in scans:
            why.append("스캔 안 됨")
        elif not yields.get(stem):
            # 검출기를 고치면 다시 스캔해야 하고, 그러려면 영상이 있어야 한다.
            why.append("플레이 0판 (검출기 고칠 때까지 보관)")
        if not (row and row.get("url")):
            why.append("주소 모름")
        (held if why else ready).append((stem, size, path, row, why))

    # 지울 것의 주소를 먼저 적는다. 적기 전에는 아무것도 지우지 않는다.
    for stem, size, _path, row, _why in ready:
        kept["matches"][stem] = {
            "vod_id": row.get("vod_id"), "url": row.get("url"),
            "title": row.get("title"), "event": row.get("event"),
            "duration": row.get("duration"), "gigabytes": round(size, 2),
            "scanned": True, "plays": yields.get(stem, 0), "retired": str(date.today()),
        }
    if ready:
        json.dump(kept, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"주소 {len(ready)}개를 {LEDGER}에 적었습니다 "
              f"(장부 전체 {len(kept['matches'])}개)\n")

    print(f"지워도 되는 것 {len(ready)}개 · {sum(r[1] for r in ready):.0f} GB")
    for stem, size, _p, _r, _w in ready[:5]:
        print(f"   {stem:<34} {size:5.1f} GB")
    if len(ready) > 5:
        print(f"   … 그리고 {len(ready) - 5}개")

    if held:
        print(f"\n남겨 두는 것 {len(held)}개 · {sum(r[1] for r in held):.1f} GB")
        for stem, size, _p, _r, why in held:
            print(f"   {stem:<34} {size:5.1f} GB  ← {', '.join(why)}")

    if not args.delete:
        print("\n(말만 했습니다. 실제로 지우려면 --delete)")
        return 0

    freed = 0.0
    for stem, size, path, _r, _w in ready:
        os.remove(path)
        freed += size
        # yt-dlp가 남긴 부스러기도 함께.
        for extra in glob.glob(os.path.join(VIDEOS, stem + ".*")):
            if not extra.endswith(".mp4"):
                os.remove(extra)
    print(f"\n{len(ready)}개 지웠습니다 · {freed:.0f} GB 회수")
    print(f"주소는 {LEDGER}에 있습니다 — git add -f 하는 것을 잊지 마세요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
