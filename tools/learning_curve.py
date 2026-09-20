"""몇 판을 모아야 하는가 — 추측하지 말고 곡선을 그려 본다.

"10,000판은 있어야 한다" 같은 말은 세어 보지 않으면 그냥 숫자다. 여기서는 참고
자료의 크기를 키워 가며 **고정된 test 경기**에 대고 점수를 재서, 어디서 평평해
지는지 본다.

2026-09-21에 재 본 결과, **두 곡선이 정반대였다**:

* **순위 식** (`tools/learn_choices.py`의 가중치 아홉 개) — 100판에서 21.6%,
  1,329판에서 21.9%. **완전히 평평하다.** 자료를 열 배로 늘려도 이 식은 좋아지지
  않는다. 한계는 자료가 아니라 식의 모양이다.
* **이웃 찾기** (kNN) — 100판 31.3% → 800판 34.8% → 1,335판 **40.0%**.
  **아직 가파르게 오르는 중이고 평평해질 낌새가 없다.** 자료가 값을 하는 곳은
  여기다.

그러므로 "몇 판이 필요한가"의 답은 **무엇을 위해서냐에 따라 다르다.**

## 2026-09-15에 세운 추정 (대화 원본에 있던 것)

선수가 "최소 play수와 optimal play수"를 물었을 때 내놓은 표다. 그때는 세어 본
것이 아니라 계산으로 잡은 **추정치**였고, 본인도 "정확한 값은 학습 곡선을 그려서
확인해야 한다"고 적어 두었다.

| 단계 | 플레이 | 경기 | 할 수 있는 것 |
|---|---|---|---|
| 최소 | 1,500 | 25 | 흔한 배치에서 경로 유형별 성공률 ±10% |
| 권장 | 5,000 | 85 | 모델을 학습하고 검증 데이터로 제대로 평가 |
| 최적 | 10,000+ | 170+ | 세밀한 배치 구분, 드문 배치, 선수별 차이 |

근거: 배치를 30~50유형 × 유형당 경로 3~5가지 ≈ 150칸, 칸당 25개면 성공률을
±10%로 말할 수 있다.

**6일 뒤 실제로 재 보니** 곡선의 모양은 맞았고 (이웃 찾기는 계속 오른다) 기대치는
과했다. 1,335판에서 40%, 두 배마다 2점이면 10,000판은 **44%쯤**이다. "믿을 수
있는 모델"이라기보다 "네 판에 한 번 더 맞히는 정도"다.

그때 같이 적어 둔 경고 하나는 정확히 들어맞았다:

> *"프로는 좋은 경로만 고르기 때문에 '안 좋은 경로를 쳤을 때'의 데이터가 거의
> 없습니다. 이 부분은 플레이 수를 늘려도 잘 해결되지 않고, **시뮬레이터로
> 보완**해야 합니다."*

그것이 `tools/enumerate_alternatives.js`가 된 것이고, 버린 길 30,167개가 거기서
나왔다.

★**이 표는 대화 원본에만 있었다.** 2026-09-21에 "10,000이라고 했었냐"는 물음에
메모와 ref/만 뒤지고 "기록에 없다"고 답했다가 틀렸다. 원본은
`~/.claude/projects/.../\*.jsonl`에 있고, 거기까지 찾아야 한다.

    ~/.venvs/carom/bin/python tools/learning_curve.py
"""
import sys, json; sys.path.insert(0,'/mnt/d/Data/Billiard/carom_video2data_01')
import numpy as np
from collections import Counter

d = json.load(open('build/app-data.json', encoding='utf-8'))
plays = [p for p in d['plays'] if p.get('route')]
F = np.array([p['f'] for p in plays], dtype=float)
route = np.array([p['route'] for p in plays])
match = np.array([p['match'] for p in plays])
print(f'플레이 {len(plays)}판 · 경기 {len(set(match))}개 · 특징 {F.shape[1]}개\n')

matches = sorted(set(match))
held = set(matches[::7])                      # 고정 test 경기
is_test = np.array([m in held for m in match])
Xte, yte = F[is_test], route[is_test]
Xpool, ypool = F[~is_test], route[~is_test]
rng = np.random.default_rng(5)
base = Counter(ypool).most_common(1)[0]
print(f'  아무 생각 없이 가장 흔한 유형({base[0]})만 찍으면  {np.mean(yte==base[0]):.1%}\n')
print(' 참고자료판수   이웃 40개 투표로 프로의 유형 맞히기')
for n in (100, 200, 400, 800, 1200, len(Xpool)):
    if n > len(Xpool): break
    hits=[]
    for seed in range(3):
        take = rng.permutation(len(Xpool))[:n]
        X, y = Xpool[take], ypool[take]
        k = min(40, n)
        got = []
        for row in Xte:
            gap = np.linalg.norm(X - row, axis=1)
            near = y[np.argsort(gap)[:k]]
            got.append(Counter(near).most_common(1)[0][0])
        hits.append(np.mean(np.array(got) == yte))
    print(f'  {n:>8}      {np.mean(hits):6.1%}  (±{np.std(hits):.1%})')
