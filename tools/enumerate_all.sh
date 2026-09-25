#!/usr/bin/env bash
# 열거를 코어 수만큼 나눠 돌리고 합친다.
#
# 16코어인데 한 코어만 쓰느라 여섯 시간을 기다린 적이 있다 (2026-09-25). 그날
# 그 여섯 시간을 두 번 썼고 한 번은 통째로 버렸다. 시뮬레이터는 판마다 완전히
# 독립이므로 나누는 데 아무 조건이 없다.
#
#   bash tools/enumerate_all.sh          # 코어 수 − 2 만큼
#   bash tools/enumerate_all.sh 8
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-$(( $(nproc) - 2 ))}
[ "$N" -lt 1 ] && N=1
echo "조각 $N개로 나눠 돌립니다 (코어 $(nproc)개)"
rm -f data/alternatives.part*.jsonl
for i in $(seq 0 $((N - 1))); do
  node --max-old-space-size=2000 tools/enumerate_alternatives.js --shard "$i/$N" \
    > "data/_alternatives.part$i.log" 2>&1 &
done
wait
cat data/alternatives.part*.jsonl > data/alternatives.jsonl
rm -f data/alternatives.part*.jsonl
echo "합쳤습니다 — $(wc -l < data/alternatives.jsonl)판 -> data/alternatives.jsonl"
