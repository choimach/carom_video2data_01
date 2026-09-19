#!/usr/bin/env bash
# 새 경기를 받아서 ML 자료까지 **끝까지** 밀어 넣는다.
#
# 여태 이것이 없었다. 도구는 다 있었지만 잇는 것이 없어서, 스크리닝이 새 경기
# 열한 개를 찾아 놓고도 아무도 받지 않았고, model.json은 하루 묵은 채였다.
# "1500판 모으고 주당 1000씩"이 안 되고 있던 이유가 이것이다.
#
# 단계마다 앞 단계가 끝나야 다음이 돈다. 앞서 한 번, 기다리는 고리를 pgrep으로
# 짜다가 **그 고리가 제 명령줄을 붙잡아** 13시간을 멈춰 있었다. 그래서 여기서는
# 기다리지 않는다 - 그냥 차례로 돌린다. 각 단계는 이미 한 것을 건너뛰므로
# 언제 다시 돌려도 안전하다.
#
#     bash tools/keep_up.sh              # 받기부터 끝까지
#     bash tools/keep_up.sh --no-collect # 이미 받은 것만 처리
#
# 로그는 data/_keep_up.log.

set -u
cd "$(dirname "$0")/.." || exit 1
PY=~/.venvs/carom/bin/python
LOG=data/_keep_up.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

step() {                       # step "이름" 명령...
  local name="$1"; shift
  say "── $name 시작"
  if "$@" >> "$LOG" 2>&1; then
    say "── $name 끝"
  else
    say "!! $name 실패 (코드 $?) — 여기서 멈춥니다. $LOG를 보세요"
    exit 1
  fi
}

say "=== keep_up 시작 ==="
say "지금: 영상 $(ls data/videos/*.mp4 2>/dev/null | wc -l)개 · 스캔 $(ls data/scans/*.npz 2>/dev/null | wc -l)개"

if [ "${1:-}" != "--no-collect" ]; then
  # 스크리닝을 통과했는데 아직 없는 경기를 받는다. 이미 있는 것은 건너뛴다.
  step "1/5 내려받기" $PY tools/collect.py
fi

# 받은 영상 중 아직 스캔하지 않은 것만. 비싼 단계이고, 캐시는 나중에 판정이
# 바뀌어도 그대로 쓴다.
step "2/5 스캔" $PY tools/scan_all.py

# 스캔에서 플레이를 뽑아 경기별 JSON과 궤적으로. 싸므로 매번 전부 다시 한다.
step "3/5 플레이 뽑기" $PY tools/export_all.py

# 모델이 배워도 되는 것만 걸러 경기 단위로 train/test를 나눈다.
step "4/5 모델 자료" $PY tools/model_dataset.py

# 조언판이 읽는 묶음과 페이지.
step "5/5 조언판 자료" $PY tools/app_data.py --out build/app-data.json
step "5/5 조언판 만들기" $PY tools/build_assistant.py

say "=== 끝 ==="
say "지금: 영상 $(ls data/videos/*.mp4 2>/dev/null | wc -l)개 · 스캔 $(ls data/scans/*.npz 2>/dev/null | wc -l)개"
$PY - <<'PY' | tee -a "$LOG"
import json
d = json.load(open('build/app-data.json', encoding='utf-8'))
m = json.load(open('data/model.json', encoding='utf-8'))
rows = m if isinstance(m, list) else m.get('plays', m.get('rows'))
print(f"플레이: 조언판 {len(d['plays'])}개 · 모델 자료 {len(rows)}개")
PY
