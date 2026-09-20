"""순위 계산이 선수의 판단에 대고 몇 점인지 매긴다.

오늘까지 순위를 여러 번 고쳤다 — 두께를 넣고, 당점을 극좌표로 바꾸고, 강도
눈금을 다시 잡았다 — 그런데 **나아졌는지 잴 방법이 없었다.** 좋아졌다는 말을
들은 것이 전부였고, 그건 측정이 아니다.

`data/table_feedback.json`이 그 자를 준다. 한 판마다 조언판이 순서대로 늘어놓은
후보가 있고, 그중 무엇이 맞고 무엇이 아닌지 선수가 눌러 둔 것이 있다. 그러면
물을 수 있다: **맞다고 하신 것이 내 목록에서 몇 번째에 있었나.**

    ~/.venvs/carom/bin/python tools/table_check.py

`tools/rules_check.py`가 유형 규칙에 하는 일과 같다. 이 숫자를 올리는 변경만
개선이라고 부른다. 여기 담긴 판단을 규칙으로 굳히지는 않는다 — 채점표다.
"""

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "data", "table_feedback.json")
# 이 아래로는 숫자가 흔들려서 개선인지 운인지 가릴 수 없다.
ENOUGH = 30


def rounds():
    if not os.path.exists(LEDGER):
        return []
    kept = json.load(open(LEDGER, encoding="utf-8"))
    return [r for r in kept.get("rounds", [])
            if not r.get("lost") and not r.get("raw") and r.get("candidates")]


def score(one):
    """한 판의 점수. 후보는 이미 순위 순서로 들어 있다."""
    verdicts = one.get("verdicts") or {}
    order = [c.get("key") for c in one["candidates"]]
    good = [i for i, key in enumerate(order) if verdicts.get(key) == "good"]
    bad = [i for i, key in enumerate(order) if verdicts.get(key) == "bad"]
    if not good and not bad:
        return None
    out = {"n": len(order), "good": len(good), "bad": len(bad)}
    if good:
        out["best_rank"] = min(good) + 1          # 맞다고 하신 것 중 가장 위
        # 맞는 것보다 위에 있는 틀린 것의 수 — 순위가 헛디딘 횟수.
        out["bad_above"] = sum(1 for i in bad if i < min(good))
    if order:
        out["top_is_good"] = verdicts.get(order[0]) == "good"
        out["top_is_bad"] = verdicts.get(order[0]) == "bad"
    return out


def middle(values):
    if not values:
        return None
    ordered = sorted(values)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[half])
    return (ordered[half - 1] + ordered[half]) / 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="판마다 보여준다")
    args = parser.parse_args()

    kept = rounds()
    scored = [(one, score(one)) for one in kept]
    usable = [(one, got) for one, got in scored if got]
    print(f"장부 {LEDGER}")
    print(f"기록 {len(kept)}판 · 그중 판단이 붙은 것 {len(usable)}판\n")

    if not usable:
        print("아직 채점할 것이 없습니다. 조언판에서 후보마다 맞다/아니다를 눌러")
        print("결과 복사한 다음 tools/record_feedback.py에 넣어 주세요.")
        return 0

    ranks = [got["best_rank"] for _one, got in usable if "best_rank" in got]
    above = [got["bad_above"] for _one, got in usable if "bad_above" in got]
    tops = [got for _one, got in usable if "top_is_good" in got]
    right = sum(1 for g in tops if g["top_is_good"])
    wrong = sum(1 for g in tops if g["top_is_bad"])

    print("순위가 선수의 판단에 대고 받는 점수")
    if ranks:
        print(f"  맞다고 하신 것의 자리   중앙값 {middle(ranks):.1f}번째"
              f" (1번째가 {sum(1 for r in ranks if r == 1)}/{len(ranks)}판)")
    if above:
        print(f"  그 위에 있던 틀린 것    중앙값 {middle(above):.1f}개")
    if tops:
        print(f"  1번 추천               맞다 {right}판 · 아니다 {wrong}판"
              f" · 판단 없음 {len(tops) - right - wrong}판")

    # 판 단위 점수는 한 판에 숫자 하나씩만 쌓인다. 그런데 한 판에 후보가 스무
    # 개쯤 있고 거기 전부 판단이 붙으므로, **유형별로 세면 훨씬 빨리 보인다** —
    # 두 판이 스물한 개의 판단이 된다. 어느 유형을 우리가 과대평가하는지는 이
    # 쪽에서 먼저 드러난다.
    by_route = {}
    for one, _got in usable:
        verdicts = one.get("verdicts") or {}
        for branch in one["candidates"]:
            say = verdicts.get(branch.get("key"))
            if not say:
                continue
            tally = by_route.setdefault(branch.get("route", "?"), Counter())
            tally[say] += 1
            tally["rank"] += one["candidates"].index(branch) + 1
    if by_route:
        print("\n유형별로 맞다/아니다 (판단이 붙은 후보 "
              f"{sum(t['good'] + t['bad'] for t in by_route.values())}개)")
        rows = sorted(by_route.items(),
                      key=lambda kv: -(kv[1]["bad"] / max(kv[1]["good"] + kv[1]["bad"], 1)))
        for route, tally in rows:
            n = tally["good"] + tally["bad"]
            print(f"  {route:<10} 맞다 {tally['good']:>2} · 아니다 {tally['bad']:>2}"
                  f"   ({tally['bad'] / n:.0%} 아니다, 평균 자리 {tally['rank'] / n:.1f})")

    done = [one for one in kept if one.get("played")]
    if done:
        went_in = sum(1 for one in done if one["played"].get("scored"))
        print(f"\n실제로 쳐보신 것 {len(done)}판 — 들어감 {went_in} · 빗나감 {len(done) - went_in}")
    else:
        print("\n실제로 쳐보신 기록은 아직 없습니다 — 그것만이 선호가 아니라 결과라,")
        print("순위를 훈련시키는 데 쓸 수 있는 유일한 자료입니다.")

    if len(usable) < ENOUGH:
        print(f"\n⚠️ {len(usable)}판으로는 숫자가 흔들립니다. {ENOUGH}판까지는 이 값이")
        print("   올랐다고 개선이라고 부르지 않습니다.")

    if args.verbose:
        print()
        for one, got in usable:
            balls = " ".join(f"{c}{tuple(xy)}" for c, xy in (one.get("layout") or {}).items())
            print(f"  {one.get('at', '?')} {balls}")
            print(f"    후보 {got['n']}개 · 맞다 {got['good']} · 아니다 {got['bad']}"
                  + (f" · 맞는 것이 {got['best_rank']}번째" if "best_rank" in got else ""))
            if one.get("comment"):
                print(f"    의견: {one['comment'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
