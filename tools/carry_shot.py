"""이웃 프로가 친 **샷**을 이 배치로 옮겨 보고, 실제와 얼마나 가까운지 잰다.

선수가 정한 차례 (2026-09-21): **1 > 2 >> 3 > 4 >> 5**.
① 프로라면 무엇을 골랐을까 → ② 수구의 진행 방향 → ③ 적구들의 진행 →
④ 두께·강도·당점 → ⑤ 경로 이름.

지금 코드는 ①을 **9지선다 분류**로 풀고 있다 — 후보를 `유형|1적구|면`으로 묶고
이웃이 유형에 투표한다. 실측 40%, 최빈값만 찍어도 30.8%인 문제다. 그리고 그
길목에서 ②③④가 쓸 정보를 버린다.

①이 실제로 묻는 것은 **"어떤 샷이냐"**다. 그래서 여기서는 이름을 거치지 않고
이웃의 샷 자체를 옮긴다:

    이웃이 준 것 = (가까운 공이냐 먼 공이냐, 어느 면, 몇 두께, 얼마나 세게)
    이 배치에서 그 두께로 그 면을 맞히는 겨냥을 계산 → 쳐 본다 → 근처를 다듬는다

절대 각도를 옮기지 않는 것이 요점이다. 배치가 다르면 각도는 뜻이 없지만
"먼 공의 오른쪽 면을 1/2 두께로 강도 4.5" 는 어느 배치에서든 뜻이 있다.

재는 자도 이름이 아니라 **궤적**이다 — ①과 ②를 한꺼번에 재는 셈이라 차례에 맞다.
프로가 실제로 지나간 길과 우리가 내놓은 길이 몇 mm 떨어져 있는가.

    ~/.venvs/carom/bin/python tools/carry_shot.py --plays 120

⚠️ 당점(팁)은 `build/app-data.json`에 없다. 이웃에게서 옮길 수 없어 탐색으로
채운다 — ④의 한 조각이 비어 있는 셈이고, 내보내기를 고쳐야 한다.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "build", "app-data.json")
SIM_JS = os.path.join(ROOT, "build", "sim.js")
K = 40


def nearest(point, F, matches, not_match, k=K):
    """같은 경기는 이웃으로 쓰지 않는다 — 같은 선수의 같은 버릇을 보고 맞혔다고
    하게 된다."""
    gap = np.linalg.norm(F - point, axis=1)
    gap[matches == not_match] = np.inf
    return np.argsort(gap)[:k]


def carry(play, neighbours, plays):
    """이웃들이 말하는 샷을, 어느 공·어느 면·몇 두께·얼마나 세게로 모은다."""
    votes = {}
    for rank, i in enumerate(neighbours):
        other = plays[i]
        if other.get("face") is None:
            continue
        # 가까이 있는 이웃일수록 무겁게. 순위로만 줄여도 충분하다.
        weight = 1.0 / (1.0 + rank)
        key = (bool(other["near"]), other["face"])
        seat = votes.setdefault(key, {"weight": 0.0, "thick": [], "speed": []})
        seat["weight"] += weight
        if other.get("thick") is not None:
            seat["thick"].append(other["thick"])
        if other.get("speed"):
            seat["speed"].append(other["speed"])
    out = []
    for (near, face), seat in sorted(votes.items(), key=lambda kv: -kv[1]["weight"]):
        out.append({
            "near": near, "face": face, "weight": round(seat["weight"], 3),
            "thickness": float(np.median(seat["thick"])) if seat["thick"] else None,
            "speed": float(np.median(seat["speed"])) if seat["speed"] else None,
            "n": len(seat["thick"]),
        })
    return out


def run_js(script):
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(done.stderr[-2000:])
    return json.loads(done.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plays", type=int, default=120)
    args = parser.parse_args()

    data = json.load(open(DATA, encoding="utf-8"))
    plays = data["plays"]
    F = np.array([p["f"] for p in plays], dtype=float)
    matches = np.array([p["match"] for p in plays])

    jobs = []
    step = max(1, len(plays) // args.plays)
    for i in range(0, len(plays), step):
        play = plays[i]
        if play.get("face") is None or not play.get("path"):
            continue
        near = nearest(F[i], F, matches, play["match"])
        jobs.append({
            "layout": {"cue": play["cue"], "balls": play["balls"]},
            "path": play["path"],
            "carried": carry(play, near, plays),
            "truth": {"near": bool(play["near"]), "face": play["face"],
                      "thickness": play.get("thick"), "speed": play.get("speed")},
        })
        if len(jobs) >= args.plays:
            break

    print(f"플레이 {len(jobs)}개 · 이웃 {K}명 (같은 경기는 빼고)\n")
    print(json.dumps({"jobs": jobs}, ensure_ascii=False),
          file=open(os.path.join(ROOT, "build", "_carry_jobs.json"), "w", encoding="utf-8"))
    print(f"-> build/_carry_jobs.json  (다음: node tools/carry_shot.js)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
