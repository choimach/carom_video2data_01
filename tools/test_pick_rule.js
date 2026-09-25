// 대표 줄을 무엇으로 고를 것인가 — "여유가 가장 넓은 것" 대 "들어갈 확률".
//
// 배경 (2026-09-25): 우리 대표 줄의 당점이 프로가 실제로 준 것에서 1.95팁
// 떨어져 있다. 갈라 보니 **물리 오차가 아니다**:
//
//   프로가 친 줄이 우리 탐색 안에 있나   있다 (0.79팁, 격자 간격이 0.52팁)
//   프로 줄의 여유 대 우리 줄           0.194도 대 0.318도 (프로 쪽이 진짜 좁다)
//   어느 쪽이 잘 들어가나               30.0% 대 31.7% — 같다 (0.6 표준편차)
//
// 같은 공략을 치는 두 방법이고 품질이 같다. 그래서 고르는 규칙의 문제다.
//
// 새 규칙: **들어갈 확률이 최고와 비슷한 줄들 중에서 프로 당점에 가장 가까운 것**.
// 지금 규칙(여유가 가장 넓은 것)과 견준다. 재는 것은 둘 —
//   ① 프로 당점과의 차이가 줄어드나
//   ② 들어갈 확률을 얼마나 내주나
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const { search, SIM, FINE } = require(path.join(ROOT, 'tools', 'enumerate_alternatives.js'));
const { chance } = require(path.join(ROOT, 'tools', 'make_probability.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const pool = data.plays.filter((p) => p.f);
const K = 40, WOBBLE = { aim: 0.1, speed: 0.08, tip: 0.3 };
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
const targets = data.plays.filter((p) => p.tip && p.tip_ok && p.aim !== null && p.speed && p.f);
const N = Number(process.argv[2] || 40);
const step = Math.max(1, Math.floor(targets.length / N));
const pick = [];
for (let i = 0; i < targets.length && pick.length < N; i += step) pick.push(targets[i]);
const mid = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
const gapNow = [], gapNew = [], pNow = [], pNew = [];
// 순환이 아닌 검사: 고른 줄의 쿠션 자리가 **영상**과 얼마나 맞나.
const railNow = [], railNew = [];
const cushionsOf = (h) => {
  const rad = h.deg * Math.PI / 180, v = SIM.speedFor(h.strength);
  const shot = SIM.play(layoutOf, 'white', [Math.cos(rad) * v, Math.sin(rad) * v], h.side, h.up || 0);
  return shot.events.filter((e) => e.kind === 'cushion').map((e) => e.p);
};
let layoutOf = null;
for (const play of pick) {
  const near = pool.filter((q) => q.match !== play.match).map((q) => {
    let s = 0; for (let i = 0; i < q.f.length; i++) { const d = q.f[i] - play.f[i]; s += d * d; }
    return { q, gap: s };
  }).sort((a, b) => a.gap - b.gap).slice(0, K);
  const tips = new Map();
  for (const { q } of near) {
    if (!q.tip || !q.tip_ok) continue;
    for (const at of [`${q.route}|${q.near}|${q.face}`, q.route]) {
      if (!tips.has(at)) tips.set(at, []);
      tips.get(at).push(q.tip);
    }
  }
  const want = middleTip(tips.get(`${play.route}|${play.near}|${play.face}`)) || middleTip(tips.get(play.route));
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  layoutOf = layout;
  const hits = search(layout, 'white');
  if (!hits.length) continue;
  const isNear = (c) => {
    const o = c === 'yellow' ? 'red' : 'yellow';
    return Math.hypot(layout[c][0] - layout.white[0], layout[c][1] - layout.white[1])
      <= Math.hypot(layout[o][0] - layout.white[0], layout[o][1] - layout.white[1]);
  };
  const mine = hits.filter((h) => h.route === play.route && isNear(h.first) === play.near && h.face === play.face);
  if (!mine.length || !want) continue;
  const slot = (h, d) => `${h.speed}:${h.side}:${Math.round(d / FINE) * FINE}`;
  const at = new Map();
  for (const h of hits) at.set(slot(h, h.deg), h);
  for (const h of mine) {
    let r = FINE;
    for (const dir of [FINE, -FINE]) { let d = h.deg + dir; while (at.has(slot(h, (d + 360) % 360))) { r += FINE; d += dir; } }
    h.rough = r;
  }
  // 지금 규칙
  const now = mine.reduce((x, y) => (y.rough > x.rough ? y : x), mine[0]);
  // 새 규칙: 여유 상위 여덟 줄의 확률을 재고, 최고에서 5점 안이면 후보로 본다.
  const top = [...mine].sort((a, b) => b.rough - a.rough).slice(0, 8);
  const line = (h) => ({ deg: h.deg, speed: SIM.speedFor(h.strength), side: h.side, up: h.up || 0 });
  for (const h of top) h.p = chance(layout, 'white', line(h), WOBBLE, 40, 7);
  const most = Math.max(...top.map((h) => h.p));
  const ok = top.filter((h) => h.p >= most - 0.05);
  let neu = ok[0];
  for (const h of ok) {
    if (Math.hypot(h.side - want[0], (h.up || 0) - want[1])
        < Math.hypot(neu.side - want[0], (neu.up || 0) - want[1])) neu = h;
  }
  gapNow.push(Math.hypot(now.side - play.tip[0], (now.up || 0) - play.tip[1]));
  gapNew.push(Math.hypot(neu.side - play.tip[0], (neu.up || 0) - play.tip[1]));
  pNow.push(now.p !== undefined ? now.p : chance(layout, 'white', line(now), WOBBLE, 40, 7));
  pNew.push(neu.p);
  if (play.rails && play.rails.length >= 2) {
    const d = (h) => {
      const cs = cushionsOf(h);
      if (cs.length < 2) return null;
      return (Math.hypot(cs[0][0] - play.rails[0][0], cs[0][1] - play.rails[0][1])
            + Math.hypot(cs[1][0] - play.rails[1][0], cs[1][1] - play.rails[1][1])) / 2;
    };
    const a = d(now), b = d(neu);
    if (a !== null && b !== null) { railNow.push(a); railNew.push(b); }
  }
}
console.log(`프로 배치 ${gapNow.length}개 (같은 경기는 이웃에서 뺐다)\n`);
console.log('                        프로 당점과의 차이   들어갈 확률');
console.log(`  지금 (여유가 가장 넓은 것)    ${mid(gapNow).toFixed(2)}팁          ${(mid(pNow) * 100).toFixed(1)}%`);
console.log(`  새 규칙 (확률 비슷 + 프로 당점) ${mid(gapNew).toFixed(2)}팁          ${(mid(pNew) * 100).toFixed(1)}%`);
let w = 0, l = 0;
for (let i = 0; i < gapNow.length; i++) { if (gapNew[i] < gapNow[i]) w++; else if (gapNew[i] > gapNow[i]) l++; }
const sd = (w + l) ? Math.abs(w - (w + l) / 2) / Math.sqrt((w + l) * 0.25) : 0;
console.log(`\n  당점이 가까워진 판 ${w} · 멀어진 판 ${l} · 같은 판 ${gapNow.length - w - l}  (${sd.toFixed(1)} 표준편차)`);
console.log(`\n★순환이 아닌 검사 — 고른 줄의 1·2쿠션 자리가 **영상**과 얼마나 맞나 (${railNow.length}판)`);
console.log(`  지금 (여유)     ${Math.round(mid(railNow))} mm`);
console.log(`  새 규칙 (확률)  ${Math.round(mid(railNew))} mm`);
let rw = 0, rl = 0;
for (let i = 0; i < railNow.length; i++) { if (railNew[i] < railNow[i]) rw++; else if (railNew[i] > railNow[i]) rl++; }
const rsd = (rw + rl) ? Math.abs(rw - (rw + rl) / 2) / Math.sqrt((rw + rl) * 0.25) : 0;
console.log(`  가까워진 판 ${rw} · 멀어진 판 ${rl} · 같은 판 ${railNow.length - rw - rl}  (${rsd.toFixed(1)} 표준편차)`);
