"""가상 테이블에서 받은 판단을 장부에 쌓는다.

이 프로젝트에서 가장 값진 데이터는 영상이 아니라 **선수가 직접 본 배치와 그에
대한 판단**이다. 영상은 프로가 무엇을 골랐는지만 알려 주고, 고르지 않은 길은
남기지 않는다. 가상 테이블에서는 조언판이 열 가지를 늘어놓고 그가 "이건
옆돌리기야", "뒤돌리기가 정답이야", "두께는 1/2~2/3에서 잡아"라고 말한다 —
**틀린 답과 맞는 답이 함께 있는 유일한 자리**다.

그런데 그것이 대화에 한 번 나타났다가 사라지고 있었다. 이 장부가 그 구멍이다.

조언판의 **결과 복사**는 사람이 읽는 글 아래에 ```carom-feedback 덩어리를
붙인다. 그 붙여넣기를 통째로 이 도구에 넣으면 된다:

    ~/.venvs/carom/bin/python tools/record_feedback.py < 붙여넣기.txt
    ~/.venvs/carom/bin/python tools/record_feedback.py --show

`data/labels.json`과 같은 성격이다 — **규칙이 아니라 채점표다.** 어떤 순위
계산이든 이것에 대고 점수를 매긴다. 여기 담긴 것을 규칙으로 굳히지 않는다:
"나라고 당구를 배우는 입장에서 어떻게 규칙이라고 확정을 하겠어."
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "data", "table_feedback.json")
BLOCK = re.compile(r"```carom-feedback\s*(\{.*?\})\s*```", re.S)

NOTE = ("가상 테이블에서 선수가 직접 준 판단. 규칙이 아니라 채점표다 — "
        "순위 계산이 이것에 대고 점수를 받는다.")


def load():
    if not os.path.exists(LEDGER):
        return {"note": NOTE, "rounds": []}
    kept = json.load(open(LEDGER, encoding="utf-8"))
    kept.setdefault("note", NOTE)
    kept.setdefault("rounds", [])
    return kept


def save(kept):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    json.dump(kept, open(LEDGER, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def same_round(a, b):
    """같은 배치에 같은 말이면 같은 기록이다. 붙여넣기를 두 번 해도 한 번만 쌓인다."""
    return (a.get("layout") == b.get("layout")
            and a.get("comment") == b.get("comment")
            and a.get("picked") == b.get("picked"))


def add(text):
    found = BLOCK.findall(text)
    if not found:
        # 덩어리가 없는 옛 붙여넣기도 버리지 않는다. 사람이 읽는 글 그대로
        # 담아 두면 나중에 손으로 옮길 수 있다.
        if text.strip():
            return [{"raw": text.strip()}]
        return []
    return [json.loads(chunk) for chunk in found]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="장부를 요약해 보여준다")
    parser.add_argument("paste", nargs="?", help="붙여넣기가 담긴 파일 (없으면 표준입력)")
    args = parser.parse_args()

    kept = load()
    if args.show:
        rounds = kept["rounds"]
        print(f"{LEDGER}\n기록 {len(rounds)}개\n")
        for i, one in enumerate(rounds, 1):
            if "raw" in one:
                print(f"{i:>3}. (옛 붙여넣기 그대로) {one['raw'][:70]}…")
                continue
            balls = " ".join(f"{c}({x},{y})" for c, (x, y) in one.get("layout", {}).items())
            print(f"{i:>3}. {one.get('at', '?')} · 수구 {one.get('cue')} · {balls}")
            print(f"     후보 {len(one.get('candidates', []))}개"
                  f" · 고른 것 {one.get('picked') or '없음'}")
            if one.get("comment"):
                print(f"     의견: {one['comment']}")
        return 0

    text = open(args.paste, encoding="utf-8").read() if args.paste else sys.stdin.read()
    fresh = add(text)
    if not fresh:
        print("담을 것이 없습니다 — 결과 복사 전체를 넣어 주세요", file=sys.stderr)
        return 1

    added = 0
    for one in fresh:
        if any(same_round(one, old) for old in kept["rounds"]):
            continue
        kept["rounds"].append(one)
        added += 1
    save(kept)
    print(f"{added}개 담았습니다 (장부 전체 {len(kept['rounds'])}개) -> {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
