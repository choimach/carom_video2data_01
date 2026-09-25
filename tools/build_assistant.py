"""Build the advice page: the template plus the plays it ranks against.

The page is two files and they have to travel together — `assistant.html` holds
a `__DATA__` placeholder where the exported plays go, and `sim.js` is the
physics the browser runs to draw a route in *this* layout. Publishing the built
page without `sim.js` beside it leaves a table and no lines.

    ~/.venvs/carom/bin/python tools/app_data.py --out build/app-data.json
    ~/.venvs/carom/bin/python tools/build_assistant.py
    node --check build/sim.js      # before publishing, every time

Why the check: the page is a single script, and a stray brace makes the whole
thing silently do nothing - which has cost this project a session already.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "visualization" / "assistant"
PLACEHOLDER = "__DATA__"
# 배운 가중치는 손으로 베끼지 않는다. 베끼면 tools/learn_choices.py를 다시
# 돌리고도 화면이 그대로인 채로 "좋아졌다"고 말하게 된다.
WEIGHTS_MARK = "__WEIGHTS__"
WEIGHTS_FILE = ROOT / "data" / "choice_weights.json"
# 같은 이유로 득점 확률의 무게도 베끼지 않는다 (tools/score_probability.py).
CHANCE_MARK = "__CHANCE__"
CHANCE_FILE = ROOT / "data" / "probability_weights.json"


def build(data_path, out_dir):
    data = json.loads(data_path.read_text(encoding="utf-8"))
    template = (SOURCE / "assistant.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"{SOURCE / 'assistant.html'} 안에 {PLACEHOLDER}가 없습니다")

    if WEIGHTS_MARK in template:
        if not WEIGHTS_FILE.exists():
            raise SystemExit(f"{WEIGHTS_FILE}가 없습니다 — tools/learn_choices.py를 먼저 돌리세요")
        learned = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        template = template.replace(WEIGHTS_MARK, json.dumps(learned["weights"]))
        print(f"배운 가중치 {len(learned['weights'])}개 "
              f"({learned.get('rounds', '?')}판으로 맞춘 것)")

    if CHANCE_MARK in template:
        if not CHANCE_FILE.exists():
            raise SystemExit(f"{CHANCE_FILE}가 없습니다 — tools/score_probability.py를 먼저 돌리세요")
        chance = json.loads(CHANCE_FILE.read_text(encoding="utf-8"))
        template = template.replace(CHANCE_MARK, json.dumps(
            {k: chance[k] for k in ("names", "mean", "spread", "weights", "bias",
                                    "auc", "plays", "scored")}))
        print(f"득점 확률 가중치 {len(chance['weights'])}개 "
              f"(AUC {chance.get('auc')}, {chance.get('plays')}판)")

    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "carom_assistant.html"
    page.write_text(template.replace(PLACEHOLDER, json.dumps(data, ensure_ascii=False,
                                                             separators=(",", ":"))),
                    encoding="utf-8")
    for side in ("sim.js", "route.js", "choice.js"):
        shutil.copy(SOURCE / side, out_dir / side)

    # The physics is the part a browser cannot complain about usefully, so run
    # node over it here if node is around.
    try:
        for side in ("sim.js", "route.js", "choice.js"):
            subprocess.run(["node", "--check", str(out_dir / side)], check=True)
    except FileNotFoundError:
        print("node가 없어 문법 검사는 건너뜁니다", file=sys.stderr)

    print(f"{page} — 플레이 {len(data['plays'])}개, {page.stat().st_size / 1024:.0f} KB")
    print(f"{out_dir / 'sim.js'}, {out_dir / 'route.js'}, {out_dir / 'choice.js'}"
          " — 함께 올려야 합니다")
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "build" / "app-data.json")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"{args.data}가 없습니다 — 먼저 tools/app_data.py를 돌리세요")
    build(args.data, args.out)


if __name__ == "__main__":
    main()
