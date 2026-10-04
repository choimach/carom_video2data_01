"""폰 앱에서 보낸 판단을 Firestore에서 가져와 장부(`data/table_feedback.json`)에 쌓는다.

폰 앱(https://carom001-board.web.app)의 "결과 보내기"는 `users/{uid}/feedback/{id}`에
기록을 쓴다 (`src/visualization/mobile/cloud.js`). 그 기록의 `record`는 조언판의
"결과 복사"가 만드는 ```carom-feedback 덩어리와 같은 것이다 — `tools/record_feedback.py`가
받는 그것.

    ~/.venvs/carom/bin/python tools/pull_feedback.py           # 새 것만 보여 준다
    ~/.venvs/carom/bin/python tools/pull_feedback.py --save    # 장부에 쌓는다

관리자 열쇠(서비스 계정)는 **이 저장소에 두지 않는다.** 기존 폰 앱 저장소의 것을 그
자리에서 읽는다 (`CAROM_SERVICE_ACCOUNT`로 바꿀 수 있다).
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "data", "table_feedback.json")
KEY = os.environ.get("CAROM_SERVICE_ACCOUNT", "/mnt/d/Data/Billiard/carom_bot_01/serviceAccount.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", action="store_true", help="새 기록을 장부에 쌓는다")
    parser.add_argument("--count", action="store_true", help="새 기록 수만 찍는다 (tools/state.sh)")
    args = parser.parse_args(argv)

    from google.cloud import firestore
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(KEY)
    db = firestore.Client(project="carom001", credentials=creds)

    ledger = json.load(open(LEDGER, encoding="utf-8"))
    have = {(r.get("at"), r.get("seed")) for r in ledger["rounds"]}
    fresh = []
    for doc in db.collection_group("feedback").stream():
        data = doc.to_dict()
        record = data.get("record") or {}
        if (record.get("at"), record.get("seed")) in have:
            continue
        fresh.append((data.get("createdAt"), doc.reference.path, data.get("build"), record))
    fresh.sort(key=lambda row: str(row[0]))

    if args.count:
        print(len(fresh))
        return 0
    print(f"Firestore의 새 기록 {len(fresh)}개 (장부에는 {len(ledger['rounds'])}판)")
    for created, path, build, record in fresh:
        verdicts = record.get("verdicts") or {}
        print(f"\n  {record.get('at')} · 씨앗 {record.get('seed')} · 수구 {record.get('cue')}"
              f" · 판단 {len(verdicts)}개 · 빌드 {build}")
        for key in sorted(set(verdicts) | set(record.get("notes") or {}) | set(record.get("renames") or {})):
            extra = " ".join((record.get("notes") or {}).get(key, []))
            name = (record.get("renames") or {}).get(key)
            print(f"    {verdicts.get(key, '-'):5s} {key}  {extra}{'  이름→' + name if name else ''}")
        if record.get("better"):
            print(f"    더 나은 공략: {record['better']}")
        if record.get("comment"):
            print("    의견: " + record["comment"].replace("\n", "\n          "))
    # 사진 (2026-10-04): users/{uid}/photos 의 feedbackId로 짝을 지어 data/photos/feedback/ 에 저장
    photos = {}
    for doc in db.collection_group("photos").stream():
        d = doc.to_dict()
        photos.setdefault(d.get("feedbackId"), []).append((d.get("slot", 0), d.get("jpeg")))
    for _created, path, build, record in fresh:
        got = photos.get(path.rsplit("/", 1)[-1], [])
        if got:
            print(f"  {record.get('at')} — 사진 {len(got)}장")
    if args.save and fresh:
        import base64
        out_dir = os.path.join(ROOT, "data", "photos", "feedback")
        os.makedirs(out_dir, exist_ok=True)
        for _created, path, build, record in fresh:
            saved = []
            for slot, jpeg in sorted(photos.get(path.rsplit("/", 1)[-1], [])):
                if not jpeg or "," not in jpeg:
                    continue
                name = f"{(record.get('at') or 'x').replace(':', '')}_{slot}.jpg"
                with open(os.path.join(out_dir, name), "wb") as handle:
                    handle.write(base64.b64decode(jpeg.split(",", 1)[1]))
                saved.append(os.path.join("data", "photos", "feedback", name))
            record["photo_files"] = saved or None
        for _created, path, build, record in fresh:
            # 통째로 담는다 — record_feedback.py와 같다. 후보 목록(candidates)이 있어야 판단을
            # 순위에 대고 채점할 수 있다 (2026-10-03까지 빼고 담았다).
            keep = dict(record)
            keep["note"] = f"폰 앱에서 보냄 (빌드 {build}, {path})"
            ledger["rounds"].append(keep)
        with open(LEDGER, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=1)
        print(f"\n{len(fresh)}개를 장부에 쌓았습니다 → {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
