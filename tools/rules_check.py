"""Score every naming rule against the plays the player named himself.

He is learning the game too, so nothing he says is a rule - it is a hypothesis,
and `data/labels.json` is what decides between hypotheses. This prints the
current rule's score and, beside it, every rule that has been tried and dropped,
so a new idea has to beat them on the record rather than on how good it sounds.

Run it before changing how routes are named, and again after. A rule that scores
worse than the one in place is not an improvement no matter how well it argues.

    ~/.venvs/carom/bin/python tools/rules_check.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LABELS = os.path.join(ROOT, "data", "labels.json")
DATASET = os.path.join(ROOT, "data", "dataset")
SHORT = ("left", "right")


def labelled():
    """The named plays, each with what the pipeline measured for it."""
    named = json.load(open(LABELS, encoding="utf-8"))["labels"]
    wanted = {(row["match"], row["inning"], row["shot"]): row["route"] for row in named}
    out = []
    for key, route in wanted.items():
        match, inning, shot = key
        path = os.path.join(DATASET, f"{match}.json")
        if not os.path.exists(path):
            continue
        for play in json.load(open(path, encoding="utf-8"))["plays"]:
            if play["inning"] == inning and play["shot_number"] == shot:
                out.append((route, play))
                break
    return out


def by_family(play):
    """The rule in place: the face struck against the circuit round the table."""
    rails = (play.get("route") or {}).get("rails") or []
    if not rails or play.get("struck_side") is None or play.get("circuit") is None:
        return None
    same = (play["struck_side"] < 0) == (play["circuit"] > 0)
    short = rails[0] in SHORT
    return ("앞돌리기" if short else "옆돌리기") if same else ("빗겨치기" if short else "뒤돌리기")


def by_drift(play):
    """Dropped: how far the second cushion carried along the line of the shot."""
    rails = (play.get("route") or {}).get("rails") or []
    away = play.get("away_mm")
    if not rails or away is None:
        return None
    short = rails[0] in SHORT
    if away < 300.0:
        return "앞돌리기" if short else "옆돌리기"
    return "빗겨치기" if short else "뒤돌리기"


def by_rails(play):
    """Dropped: the rail sequence alone."""
    rails = (play.get("route") or {}).get("rails") or []
    if not rails:
        return None
    return "뒤돌리기" if rails[0] in SHORT else "옆돌리기"


TURNS = ("뒤돌리기", "옆돌리기", "앞돌리기", "빗겨치기")
RULES = (("맞힌 면 vs 도는 방향 (지금)", by_family),
         ("2쿠션 진행 거리 (버림)", by_drift),
         ("쿠션 순서 (버림)", by_rails))


def main():
    rows = labelled()
    turns = [(route, play) for route, play in rows if route in TURNS]
    print(f"선수가 이름 붙인 플레이 {len(rows)}개 · 그중 네 가지 돌리기 {len(turns)}개\n")
    for name, rule in RULES:
        right = sum(1 for route, play in turns if rule(play) == route)
        print(f"  {name:26s} {right:2d}/{len(turns)}  {right / len(turns) * 100:3.0f}%")

    print("\n지금 규칙이 틀리는 것:")
    for route, play in turns:
        got = by_family(play)
        if got != route:
            rails = "-".join((play.get("route") or {}).get("rails") or [])
            print(f"  {route} → {got}   면 {play.get('struck_side'):.0f} · "
                  f"도는 방향 {play.get('circuit')} · 쿠션 {rails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
