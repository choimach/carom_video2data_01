#!/usr/bin/env bash
# keep_up.sh가 끝나기를 기다렸다가 HANDOFF ①~④를 차례로 돌린다 (2026-10-10).
# ⑤ 재학습부터는 전·후를 사람이 견줘야 하므로 여기서 멈춘다.
#
#   nohup bash tools/after_scan.sh <keep_up의 pid> > data/_after_scan.log 2>&1 &
#
# ★pgrep으로 기다리지 않는다 — 자기 명령줄을 찾고 영원히 기다린다 (CLAUDE.md §3).
# pid를 받아 kill -0으로만 본다.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=~/.venvs/carom/bin/python
PID=${1:?keep_up pid}
LOG=${KEEP_UP_LOG:-data/_keep_up_20261009.log}
say() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

say "keep_up($PID)를 기다린다"
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
if ! tail -3 "$LOG" | grep -q "=== 끝 ==="; then
  say "keep_up이 끝 표시 없이 멈췄다 — 여기서 그만"; tail -5 "$LOG"; exit 1
fi
say "keep_up 끝"

run() { say "── $1 시작"; shift; "$@" || { say "실패: $*"; exit 1; }; say "── 끝"; }
run "① 영상 지우기" $PY tools/retire_videos.py --delete
run "② 열거 (조각 3)" bash tools/enumerate_all.sh 3
run "③ 영어 이름" node tools/alternatives_english.js
run "④ 전: check_verdicts" node tools/check_verdicts.js --save data/_verdicts_before_more2.json
say "=== ①~④ 끝 — ⑤부터는 HANDOFF ==="
