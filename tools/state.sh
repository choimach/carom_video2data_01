#!/usr/bin/env bash
# 이 프로젝트가 지금 어디까지 와 있는지, 한 번의 호출로.
#
# 왜 있는가: 하루 동안 "무엇이 돌고 있나 / 어디까지 됐나"를 알아내는 데만 같은
# 명령을 예닐곱 번씩 되풀이했다. 세션이 압축되면 그 답이 사라지므로 또 되풀이
# 했다. 그게 이 프로젝트에서 토큰을 가장 많이 먹는 단일 원인이다.
#
# 출력은 스무 줄 남짓으로 묶어 둔다. 세션을 열 때, 그리고 긴 작업 뒤에 한 번.
#
#     bash tools/state.sh

cd "$(dirname "$0")/.." || exit 1
PY=~/.venvs/carom/bin/python

age() {                      # 파일이 얼마나 묵었나
  [ -e "$1" ] || { echo "없음"; return; }
  local secs=$(( $(date +%s) - $(stat -c %Y "$1") ))
  if   [ $secs -lt 3600 ]  ; then echo "$((secs/60))분 전"
  elif [ $secs -lt 86400 ] ; then echo "$((secs/3600))시간 전"
  else echo "$((secs/86400))일 전"; fi
}

echo "══ 돌고 있는 것 ══"
# 자기 자신을 잡지 않도록 대괄호 한 글자 트릭을 쓴다. pgrep -f로 기다리는
# 고리를 짜면 자기를 발견하고 멈춘다 - CLAUDE.md 2절 참조.
running=$(ps -ef \
  | grep -E "[c]ollect\.py|[s]can_all|[e]xport_all|[m]odel_dataset|[e]numerate_alt|[k]eep_up|[s]creen_run|[y]t-dlp|[f]it_spin" \
  | grep -v "shell-snapshots" \
  | awk '{ $1=$2=$3=$4=$5=$6=$7=""; sub(/^ +/, ""); print }' \
  | sed -E 's#^\S*/##; s#^(python[0-9.]*|node|bash) ##' | cut -c1-62 | sed 's/^/  /')
[ -n "$running" ] && echo "$running" || echo "  (없음)"

echo
echo "══ 파이프라인 ══"
printf "  영상   %4s개 %6s\n" "$(ls data/videos/*.mp4 2>/dev/null | wc -l)" "$(du -sh data/videos 2>/dev/null | cut -f1)"
printf "  스캔   %4s개\n" "$(ls data/scans/*.npz 2>/dev/null | wc -l)"
printf "  플레이 %4s개  (model.json, %s)\n" \
  "$($PY -c "
import json
m=json.load(open('data/model.json',encoding='utf-8'))
r=m if isinstance(m,list) else m.get('plays', m.get('rows'))
print(len(r))" 2>/dev/null || echo '?')" "$(age data/model.json)"
printf "  조언판 %4s개  (app-data.json, %s)\n" \
  "$($PY -c "
import json; print(len(json.load(open('build/app-data.json',encoding='utf-8'))['plays']))" 2>/dev/null || echo '?')" "$(age build/app-data.json)"
printf "  받을 것 %3s개  (스크리닝 통과, 미수신)\n" \
  "$($PY -c "
import sys, os; sys.path.insert(0,'.')
from tools.collect import wanted, target
print(sum(1 for e in wanted() if not os.path.exists(target(e))))" 2>/dev/null || echo '?')"

echo
echo "══ 채점표 ══"
printf "  선수가 붙인 유형 이름  %3s개\n" \
  "$($PY -c "import json; print(len(json.load(open('data/labels.json',encoding='utf-8'))['labels']))" 2>/dev/null || echo 0)"
printf "  가상 테이블 판단       %3s판  (%s)\n" \
  "$($PY -c "
import json; d=json.load(open('data/table_feedback.json',encoding='utf-8'))
print(sum(1 for r in d['rounds'] if r.get('candidates')))" 2>/dev/null || echo 0)" "$(age data/table_feedback.json)"
printf "  버린 길 (프로 배치)    %3s판  (%s)\n" \
  "$(wc -l < data/alternatives.jsonl 2>/dev/null || echo 0)" "$(age data/alternatives.jsonl)"

echo
echo "══ 진행 중인 일 ══"
if [ -s HANDOFF.md ]; then sed -n '/^## 지금/,$p' HANDOFF.md | head -14; else echo "  (HANDOFF.md 비어 있음)"; fi

echo
echo "══ 마지막 커밋 ══"
git log --oneline -3 2>/dev/null | sed 's/^/  /'
