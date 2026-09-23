"""판을 늘리면 순위가 얼마나 나아지나 — 두 갈래를 따로 잰다.

선수 (2026-09-24): *"2176판이 사용되면 model이 믿을만해져?"*

두 갈래는 서로 다른 자원이고, 한쪽만 늘려도 다른 쪽이 막으면 안 움직인다.

1) **순위 식이 배우는 판** (alternatives.jsonl) — 400판에서 포화한다.
2) **이웃을 찾는 풀** (model.json) — 두 배마다 약 +2점.

2026-09-24 실측 (이웃 40명, 경기 단위 10겹):

    가) 학습   50판  1등 31.8%      나) 풀   616판  1등 31.1%
        학습  100판  1등 32.8%          풀  1233판  1등 33.0%
        학습  200판  1등 33.0%          풀  1849판  1등 34.0%
        학습  400판  1등 34.9%          풀  2466판  1등 34.7%
        학습  800판  1등 35.0%
        학습  전부   1등 34.7%

    ~/.venvs/carom/bin/python tools/data_curve.py
"""
import json, math, os, sys
from collections import Counter
import numpy as np
ROOT0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT0); sys.path.insert(0, os.path.join(ROOT0, 'tools'))
from learn_choices import (load, with_prior, as_arrays, fit, where_it_landed,
                           features, NAMES, NEIGHBOURS)
from route_model import player_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rounds_all = load()

model = json.load(open(os.path.join(ROOT, 'data', 'model.json'), encoding='utf-8'))
rows = model if isinstance(model, list) else model.get('plays', model.get('rows'))
rows = [r for r in rows if r.get('route') and r.get('layout_mm')]
F = np.array([player_features(r) for r in rows], dtype=float)
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
match = np.array([r['match'] for r in rows])
seat = {f"{r['match']}:{r['inning']}:{r['shot']}": i for i, r in enumerate(rows)}

def bucket(row):
    cue = np.array(row['layout_mm'][row['cue']], dtype=float)
    gaps = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue))
            for c, xy in row['layout_mm'].items() if c != row['cue']}
    first, side = row.get('first_object_ball'), row.get('struck_side')
    if not first or first not in gaps or side is None:
        return None
    return f"{row['route']}|{gaps[first] == min(gaps.values())}|{'left' if side > 0 else 'right'}"
picked = [bucket(r) for r in rows]

rng = np.random.default_rng(0)

def attach(rounds, pool):
    """이웃 풀을 pool(인덱스 집합)로 제한해 이웃 항을 붙인다."""
    keep = np.zeros(len(rows), dtype=bool); keep[list(pool)] = True
    out = []
    for one in rounds:
        i = seat.get(one['id'])
        if i is None: continue
        gap = np.linalg.norm(F - F[i], axis=1)
        gap[match == match[i]] = np.inf
        gap[~keep] = np.inf
        votes, wins = {}, {}
        for j in np.argsort(gap)[:NEIGHBOURS]:
            if not np.isfinite(gap[j]): break
            k = picked[j]
            if not k: continue
            votes[k] = votes.get(k, 0) + 1
            wins[k] = wins.get(k, 0) + (1 if rows[j].get('scored') else 0)
        at = one['layout']; cue_at = np.array(at[one['cue']], dtype=float)
        reach = {c: float(np.linalg.norm(np.array(xy, dtype=float) - cue_at))
                 for c, xy in at.items() if c != one['cue']}
        for b in one['found']:
            k = f"{b['route']}|{reach.get(b['first']) == min(reach.values())}|{b['face']}"
            b['chosen'] = votes.get(k, 0)
            b['rate'] = (wins.get(k, 0) + 1) / (b['chosen'] + 2)
        out.append(one)
    return out

def score(rounds, pool, train_n=None):
    data = as_arrays(with_prior(attach(rounds, pool)))
    matches = sorted({o.get('match') for _r, _c, o in data})
    folds = min(10, len(matches))
    where = {m: i % folds for i, m in enumerate(matches)}
    place = []
    for f in range(folds):
        tr = [d for d in data if where[d[2].get('match')] != f]
        te = [d for d in data if where[d[2].get('match')] == f]
        if not tr or not te: continue
        if train_n is not None and len(tr) > train_n:
            tr = [tr[k] for k in rng.choice(len(tr), train_n, replace=False)]
        w = fit(tr)
        for X, ch, _o in te:
            place.append(where_it_landed(X, ch, w))
    p = np.array(place, dtype=float)
    return np.mean(p == 1), np.mean(p <= 3), len(p)

full = set(range(len(rows)))
print(f"채점할 판 {len(rounds_all)}개 · 이웃 풀 {len(rows)}판\n")
print("가) 순위 식이 배우는 판을 늘리면 (이웃 풀은 전부)")
for n in [50, 100, 200, 400, 800, None]:
    a, b, m = score(rounds_all, full, n)
    print(f"  학습 {str(n or '전부'):>4}판   1등 {a:5.1%} · 3등 안 {b:5.1%}")
print("\n나) 이웃 풀을 늘리면 (순위 식은 전부로 배움)")
for share in [0.25, 0.5, 0.75, 1.0]:
    pool = set(rng.choice(len(rows), int(len(rows) * share), replace=False)) if share < 1 else full
    a, b, m = score(rounds_all, pool)
    print(f"  풀 {int(len(rows)*share):>5}판   1등 {a:5.1%} · 3등 안 {b:5.1%}")
