// 우리 대표 줄의 당점이 **프로가 실제로 준 당점**에서 얼마나 떨어졌나.
//
// 새 자다 (2026-09-25). 전에는 당점을 잴 자가 없었다 — 영상에 팁이 없었기
// 때문이다. `tools/fit_tip.js`가 되찾아 준 뒤로 잴 수 있게 됐다.
//
// 2026-09-25 실측 (프로 배치 86판, 같은 경기는 이웃에서 뺐다):
//
//   양보한 여유   당점 차이   가까워진/멀어진   표준편차
//     0.00도      1.95팁       10 / 11          0.2
//     0.50도      1.76팁       26 / 24          0.3
//     1.00도      1.76팁       28 / 28          0.0
//
// ★**선택 규칙으로는 못 줄인다.** 여유를 1도나 양보해도 1.76팁에 그친다.
// ★**격자 탓도 아니다.** 2팁 반지름에서 이웃한 시계 방향 사이가 0.52팁인데
//   차이는 1.8팁이다.
// ★★**우리 시뮬레이터는 같은 공략을 프로와 다른 당점으로 만들어 낸다.**
//   2쿠션이 155 mm 어긋나는 것의 당점 쪽 표현이고, 고칠 자리는 물리다.
//
// 물리를 고칠 때 이 숫자가 줄어드는지 본다. 지금은 **1.95팁**이다.
//
//   node tools/check_tip_gap.js [판 수]
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const { search, SIM, FINE } = require(path.join(ROOT, 'tools', 'enumerate_alternatives.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const K = 40;

const pool = data.plays.filter((p) => p.f);
const middleTip = (list) => {
  if (!list || list.length < 2) return null;
  let best = null;
  for (const one of list) {
    let sum = 0;
    for (const other of list) sum += Math.hypot(one[0] - other[0], one[1] - other[1]);
    if (!best || sum < best.sum) best = { sum, tip: one };
  }
  return best.tip;
};

// 채점 대상: 당점이 믿을 만하고 겨냥·강도가 있는 판.
const targets = data.plays.filter((p) => p.tip && p.tip_ok && p.aim !== null && p.speed && p.f);
const N = Number(process.argv[2] || 30);
const step = Math.max(1, Math.floor(targets.length / N));
const pick = [];
for (let i = 0; i < targets.length && pick.length < N; i += step) pick.push(targets[i]);

const BANDS = [0, 0.25, 0.5, 1.0];
const away = { 없이: [] }, room = {};
for (const g of BANDS) { away['양보' + g] = []; room['양보' + g] = []; }
for (const play of pick) {
  // 같은 경기를 뺀 이웃 40명.
  const near = pool.filter((q) => q.match !== play.match).map((q) => {
    let sum = 0;
    for (let i = 0; i < q.f.length; i++) { const d = q.f[i] - play.f[i]; sum += d * d; }
    return { q, gap: sum };
  }).sort((a, b) => a.gap - b.gap).slice(0, K);
  const tips = new Map();
  for (const { q } of near) {
    if (!q.tip || !q.tip_ok) continue;
    for (const at of [`${q.route}|${q.near}|${q.face}`, q.route]) {
      if (!tips.has(at)) tips.set(at, []);
      tips.get(at).push(q.tip);
    }
  }
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const hits = search(layout, 'white');
  if (!hits.length) continue;
  // 프로가 친 그 통을 찾는다.
  const isNear = (c) => Math.hypot(layout[c][0] - layout.white[0], layout[c][1] - layout.white[1])
    <= Math.hypot(layout[c === 'yellow' ? 'red' : 'yellow'][0] - layout.white[0],
                  layout[c === 'yellow' ? 'red' : 'yellow'][1] - layout.white[1]);
  const want = middleTip(tips.get(`${play.route}|${play.near}|${play.face}`))
    || middleTip(tips.get(play.route));
  const mine = hits.filter((h) => h.route === play.route
    && (isNear(h.first) === play.near) && h.face === play.face);
  if (!mine.length) continue;

  // 여유를 굵은 눈금으로 재서 동점을 만든다 (조언판 finish()와 같다).
  const slot = (h, deg) => `${h.speed}:${h.side}:${Math.round(deg / FINE) * FINE}`;
  const at = new Map();
  for (const h of hits) at.set(slot(h, h.deg), h);
  for (const h of mine) {
    let room = FINE;
    for (const dir of [FINE, -FINE]) {
      let d = h.deg + dir;
      while (at.has(slot(h, (d + 360) % 360))) { room += FINE; d += dir; }
    }
    h.rough = room;
  }
  const top = Math.max(...mine.map((h) => h.rough));
  const dist = (h) => Math.hypot(h.side - play.tip[0], (h.up || 0) - play.tip[1]);
  const tight = mine.filter((h) => h.rough === top);
  away['없이'].push(dist(tight[0]));
  // 여유를 얼마나 양보하면 프로 당점 쪽으로 갈 수 있나.
  for (const give of BANDS) {
    const near2 = mine.filter((h) => h.rough >= top - give);
    let best = near2[0];
    if (want) {
      for (const h of near2) {
        if (Math.hypot(h.side - want[0], (h.up || 0) - want[1])
            < Math.hypot(best.side - want[0], (best.up || 0) - want[1])) best = h;
      }
    }
    away['양보' + give].push(dist(best));
    room['양보' + give].push(top - best.rough);
  }
}
const mid = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
console.log(`채점한 판 ${away['없이'].length}개 (같은 경기는 이웃에서 뺐다)\n`);
console.log('여유를 얼마나 양보하면 당점이 프로에 가까워지나\n');
console.log('  양보    당점 차이(팁)   내준 여유(도)   가까워진/멀어진   표준편차');
for (const g of BANDS) {
  const k = '양보' + g;
  let up = 0, down = 0;
  for (let i = 0; i < away['없이'].length; i++) {
    if (away[k][i] < away['없이'][i]) up++; else if (away[k][i] > away['없이'][i]) down++;
  }
  const sd = (up + down) ? Math.abs(up - (up + down) / 2) / Math.sqrt((up + down) * 0.25) : 0;
  console.log(`  ${g.toFixed(2)}도    ${mid(away[k]).toFixed(2)}          `
    + `${mid(room[k]).toFixed(2)}           ${String(up).padStart(3)} / ${String(down).padStart(3)}`
    + `        ${sd.toFixed(1)}` + (g === 0 ? '   ← 지금' : ''));
}
