#!/usr/bin/env bash
# 열거를 코어 수만큼 나눠 돌리고 합친다.
#
# 16코어인데 한 코어만 쓰느라 여섯 시간을 기다린 적이 있다 (2026-09-25). 그날
# 그 여섯 시간을 두 번 썼고 한 번은 통째로 버렸다. 시뮬레이터는 판마다 완전히
# 독립이므로 나누는 데 아무 조건이 없다.
#
#   nohup bash tools/enumerate_all.sh > ~/.cache/carom_enum/run.log 2>&1 &
#   bash tools/enumerate_all.sh 8
#
# ★`/mnt/d`에서 돌리지 않는다 (2026-09-26). 조각 열넷이 9p 마운트를 두드리다
# **EIO**로 전부 넘어졌다 — 쓰다가도, 스크립트를 읽다가도. 필요한 다섯 파일을
# ext4(`$ENUM_WORK`, 기본 `~/.cache/carom_enum`)에 복사해 거기서 돌리고, 끝나면
# 합친 장부 하나만 저장소로 가져온다. 작업 폴더는 세션·재부팅을 넘어 남으므로
# 다시 돌리면 이어서 한다.
#
# ★감시가 있다 (2026-09-26). 같은 날 조각 하나가 배치 하나에서 로그 없이 몇
# 시간 서 있었다. 조각마다 지금 무슨 배치를 하는지 `_status<i>`에 적고, 그
# 파일이 STALL초 넘게 그대로면 죽이고 그 배치를 `alternatives.skip.txt`에 적고
# 다시 띄운다. 이유 없이 죽은 조각도 다시 띄운다 (RETRIES번까지).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK=${ENUM_WORK:-$HOME/.cache/carom_enum}
N=${1:-$(( $(nproc) - 2 ))}
[ "$N" -lt 1 ] && N=1
# ★메모리 (2026-09-26). 이 WSL은 **15 GB**뿐이다. 힙 한도 2000 MB로 조각 14개를
# 띄웠더니 조각마다 1.4 GB까지 불어 OOM이 조각을 죽였다. 쓰는 양이 아니라 GC가
# 게을러서다 — 한도 500이면 최대 360 MB로 똑같이 돈다. 한도를 낮추고, 조각 수도
# 남은 메모리로 한 번 더 자른다. 600으로 내렸더니 **정말로 590 MB가 사는** 배치가
# 있어 힙 OOM(134)이 났다 — 1000으로 둔다.
HEAP=${HEAP:-1000}
FIT=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / (HEAP + 200) ))
[ "$N" -gt "$FIT" ] && { echo "메모리가 모자라 조각을 $N → $FIT개로 줄입니다"; N=$FIT; }
[ "$N" -lt 1 ] && N=1
STALL=${STALL:-900}        # 배치 평균이 50~100초다. 15분이면 멈춘 것이다.
RETRIES=${RETRIES:-3}

say() { echo "[$(date +%H:%M:%S)] $*"; }

# id로 중복을 지우며 합친다. 덮어쓰지 않는다 — 이미 있는 판이 먼저다.
merge() {
  cat "$@" 2>/dev/null | python3 -c '
import json, sys
seen = set()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        got = json.loads(line)["id"]
    except Exception:
        continue
    if got in seen:
        continue
    seen.add(got)
    print(line)
'
}

mkdir -p "$WORK/tools" "$WORK/build" "$WORK/src/visualization/assistant" "$WORK/data"
cp "$REPO/tools/enumerate_alternatives.js" "$WORK/tools/"
cp "$REPO/build/sim.js" "$WORK/build/"
cp "$REPO/src/visualization/assistant/route.js" "$WORK/src/visualization/assistant/"
cp "$REPO/data/model.json" "$WORK/data/"
# 장부: 저장소 것 + 작업 폴더에 남은 것(지난번에 죽은 조각 포함)을 합친다.
merge "$REPO/data/alternatives.jsonl" "$WORK/data/alternatives.jsonl" \
  "$WORK"/data/alternatives.part*.jsonl > "$WORK/data/_ledger.tmp"
mv "$WORK/data/_ledger.tmp" "$WORK/data/alternatives.jsonl"
rm -f "$WORK"/data/alternatives.part*.jsonl
[ -f "$REPO/data/alternatives.skip.txt" ] && \
  sort -u "$REPO/data/alternatives.skip.txt" "$WORK/data/alternatives.skip.txt" 2>/dev/null \
  > "$WORK/data/_skip.tmp" && mv "$WORK/data/_skip.tmp" "$WORK/data/alternatives.skip.txt"
say "작업 폴더 $WORK · 장부 $(wc -l < "$WORK/data/alternatives.jsonl")판 · 조각 ${N}개 (코어 $(nproc)개)"

declare -A PID TRIES FINISHED CRASHED
launch() {
  local i=$1
  : > "$WORK/data/_status$i"
  node --max-old-space-size="$HEAP" "$WORK/tools/enumerate_alternatives.js" \
    --shard "$i/$N" --status "$WORK/data/_status$i" ${LIMIT:+--limit $LIMIT} \
    >> "$WORK/data/_part$i.log" 2>&1 &
  PID[$i]=$!
}
for i in $(seq 0 $((N - 1))); do
  TRIES[$i]=0; FINISHED[$i]=0; : > "$WORK/data/_part$i.log"; launch "$i"
done

# pid를 직접 들고 본다 — pgrep으로 기다리면 자기 자신을 찾는다.
while :; do
  left=0
  for i in $(seq 0 $((N - 1))); do
    [ "${FINISHED[$i]}" = 1 ] && continue
    pid=${PID[$i]}
    if kill -0 "$pid" 2>/dev/null; then
      left=$((left + 1))
      age=$(( $(date +%s) - $(stat -c %Y "$WORK/data/_status$i") ))
      if [ "$age" -gt "$STALL" ]; then
        stuck=$(head -1 "$WORK/data/_status$i")
        say "조각 $i: ${age}초째 배치 '$stuck'에 서 있다 — 죽이고 건너뛴다"
        kill "$pid"; sleep 2; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        [ -n "$stuck" ] && echo "$stuck" >> "$WORK/data/alternatives.skip.txt"
        launch "$i"
      fi
      continue
    fi
    wait "$pid"; code=$?
    if [ "$code" = 0 ]; then
      FINISHED[$i]=1; say "조각 $i 끝 ($(tail -1 "$WORK/data/_part$i.log"))"
    elif [ "$code" = 1 ] && grep -q "여유가 전부 0" "$WORK/data/_part$i.log"; then
      # 안전장치가 멈춘 것이다. 다시 띄워도 같다 — 전부 멈추고 사람이 본다.
      say "조각 $i: 여유가 전부 0이라 멈췄다. 전부 세운다."
      for j in "${!PID[@]}"; do kill "${PID[$j]}" 2>/dev/null; done
      exit 1
    elif [ -n "$(head -1 "$WORK/data/_status$i")" ] \
         && [ "${CRASHED[$i]:-}" = "$(head -1 "$WORK/data/_status$i")" ]; then
      # 같은 배치에서 두 번 죽었다 — 조각을 버리지 말고 그 배치만 버린다.
      stuck=$(head -1 "$WORK/data/_status$i")
      say "조각 $i: 배치 '$stuck'에서 두 번 죽었다 (코드 $code) — 건너뛰고 다시 띄운다"
      echo "$stuck" >> "$WORK/data/alternatives.skip.txt"
      launch "$i"; left=$((left + 1))
    elif [ "${TRIES[$i]}" -lt "$RETRIES" ]; then
      CRASHED[$i]=$(head -1 "$WORK/data/_status$i")
      TRIES[$i]=$(( TRIES[$i] + 1 ))
      say "조각 $i가 코드 $code로 죽었다 — 다시 띄운다 (${TRIES[$i]}/$RETRIES)"
      tail -3 "$WORK/data/_part$i.log" | sed 's/^/    /'
      launch "$i"; left=$((left + 1))
    else
      FINISHED[$i]=1; say "조각 $i: $RETRIES번 다시 띄웠는데 또 죽었다 — 포기"
    fi
  done
  [ "$left" = 0 ] && break
  sleep "${POLL:-60}"
done

merge "$WORK/data/alternatives.jsonl" "$WORK"/data/alternatives.part*.jsonl \
  > "$WORK/data/_ledger.tmp"
mv "$WORK/data/_ledger.tmp" "$WORK/data/alternatives.jsonl"
rm -f "$WORK"/data/alternatives.part*.jsonl
# 저장소에는 한 번만, 통째로 쓴다. (시험할 때는 DRY=1로 저장소를 건드리지 않는다.)
[ -n "${DRY:-}" ] && { say "DRY — 저장소에 쓰지 않음 ($(wc -l < "$WORK/data/alternatives.jsonl")판)"; exit 0; }
cp "$WORK/data/alternatives.jsonl" "$REPO/data/alternatives.jsonl"
[ -s "$WORK/data/alternatives.skip.txt" ] && \
  cp "$WORK/data/alternatives.skip.txt" "$REPO/data/alternatives.skip.txt"
say "합쳤습니다 — $(wc -l < "$REPO/data/alternatives.jsonl")판 -> data/alternatives.jsonl" \
  "· 멈춰서 건너뛴 배치 $(cat "$WORK/data/alternatives.skip.txt" 2>/dev/null | wc -l)개"
