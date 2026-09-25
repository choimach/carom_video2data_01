// 프로가 준 당점을 관측 여럿으로 한꺼번에 맞춘다 (선수 제안, 2026-09-25).
//
// 영상에서 겨냥과 강도를 알므로 **미지수는 좌우·상하 당점 둘**이다. 관측은:
//
//   1쿠션 자리   좌우 당점에 강하다 (쿠션이 팁당 5~7도 틀어 준다)
//   곡선 깊이    **상하 당점에 강하다** (−3~+3팁에서 폭 236 mm)
//   분리각       상하 당점의 부호 (두꺼울 때만)
//   2쿠션 자리   ← **맞추는 데 쓰지 않는다. 검증용으로 뺀다.**
//
// 혼자서는 어느 것도 당점을 못 준다 — 분리각은 얇은 두께에서 폭이 0.03이고,
// 곡선은 U자라 크기만 준다. 함께 쓰면 갈린다.
//
// 채점: 맞춘 당점으로 2쿠션을 얼마나 잘 맞히나. 1쿠션만으로 맞춘 것(지금
// 방식)과 견준다. 나아지면 곡선과 분리각이 진짜 정보를 담은 것이다.
//
// 2026-09-25 실측 (프로 플레이 370판):
//
//                          1쿠션만   합동
//   전체            n=370   116mm    89mm   2.7 표준편차  → 합동이 낫다
//   격자 안쪽        n=250    91mm    83mm   3.0 표준편차  → 확실하다
//   격자 끝에 걸림   n=120   153mm   135mm   0.4 표준편차  → 아무것도 아니다
//
// ★**이긴 것은 전부 "격자 안쪽" 덕이다.** 그 판들은 애초에 오차도 작다
// (91 대 153 mm) — **우리 물리가 설명할 수 있는 판들**이다.
//
// ★**그러므로 "격자 안에 머물렀나"가 곧 신뢰 표시다.** 끝(±3팁)에 걸린
// 32%는 어떤 물리적 당점으로도 설명이 안 되는 판이고, 거기서 나온 당점은
// 맞추기가 모형 오차를 떠넘긴 쓰레기통이다. 68%의 당점만 쓴다.
//
//   node tools/fit_tip.js [판 수]

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const mid = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);

// 한 번 쳐서 관측 네 가지를 읽는다.
function observe(layout, cue, first, aim, side, up) {
  const shot = SIM.play(layout, cue, aim, side, up);
  const ball = shot.events.find((e) => e.kind === 'ball');
  const rails = shot.events.filter((e) => e.kind === 'cushion');
  if (!ball || !rails.length) return null;
  const one = rails.find((e) => e.at > ball.at);
  if (!one) return null;
  const two = rails.find((e) => e.at > one.at);
  const way = shot.paths[cue], obj = shot.paths[first];
  const near = (p) => {
    let k = 0, best = Infinity;
    for (let i = 0; i < way.length; i++) {
      const d = gap(way[i], p);
      if (d < best) { best = d; k = i; }
    }
    return k;
  };
  const i = near(ball.p), j = near(one.p);
  let curve = null;
  if (j - i >= 5) {
    const a = way[i], b = way[j], len = gap(a, b);
    if (len >= 200) {
      const u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
      let m = 0;
      for (let k = i; k <= j; k++) {
        m = Math.max(m, Math.abs((way[k][0] - a[0]) * u[1] - (way[k][1] - a[1]) * u[0]));
      }
      curve = m;
    }
  }
  let rise = null;
  if (obj && i + 5 < way.length && i + 4 < obj.length) {
    const unit = (p, q) => {
      const v = [q[0] - p[0], q[1] - p[1]], n = Math.hypot(v[0], v[1]);
      return n < 1e-6 ? null : [v[0] / n, v[1] / n];
    };
    const c = unit(obj[i], obj[i + 4]), o = unit(way[i + 1], way[i + 5]);
    if (c && o) rise = o[0] * c[0] + o[1] * c[1];
  }
  return { one: one.p, two: two ? two.p : null, curve, rise };
}

const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
// 검증에 2쿠션이 필요하므로 쿠션이 둘 이상인 판만 쓴다.
const plays = data.plays.filter((p) => p.aim !== null && p.speed && p.hit
  && p.rails && p.rails.length >= 2 && p.curve && p.rise !== null && p.rise !== undefined);
const N = Number(process.argv[2] || 250);
const step = Math.max(1, Math.floor(plays.length / N));
const pick = [];
for (let i = 0; i < plays.length && pick.length < N; i += step) pick.push(plays[i]);

// 잔차를 견줄 수 있게 각자의 보통 크기로 나눈다.
const SCALE = { one: 40, curve: 20, rise: 0.15 };
const TIPS = [];
for (let s = -3; s <= 3.0001; s += 0.25) {
  for (let u = -3; u <= 3.0001; u += 0.25) TIPS.push([Math.round(s * 100) / 100, Math.round(u * 100) / 100]);
}

const joint = [], alone = [];
const tips = [];
for (const play of pick) {
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const rad = play.aim * Math.PI / 180;
  const aim = [Math.cos(rad) * play.speed, Math.sin(rad) * play.speed];
  const first = gap(play.balls[0], play.hit) < gap(play.balls[1], play.hit) ? 'yellow' : 'red';
  let best = null, only = null;
  for (const [side, up] of TIPS) {
    const got = observe(layout, 'white', first, aim, side, up);
    if (!got) continue;
    const oneOff = gap(got.one, play.rails[0]);
    // 1쿠션만 보는 지금 방식
    if (!only || oneOff < only.off) only = { off: oneOff, two: got.two, side, up };
    // 셋을 함께 보는 방식
    if (got.curve === null || got.rise === null) continue;
    const cost = (oneOff / SCALE.one) ** 2
      + ((got.curve - play.curve[0]) / SCALE.curve) ** 2
      + ((got.rise - play.rise) / SCALE.rise) ** 2;
    if (!best || cost < best.cost) best = { cost, two: got.two, side, up, oneOff };
  }
  if (!best || !only || !best.two || !only.two) continue;
  if (only.off > 60) continue;                     // 1쿠션도 못 맞춘 판은 뺀다
  joint.push(gap(best.two, play.rails[1]));
  alone.push(gap(only.two, play.rails[1]));
  tips.push({ side: best.side, up: best.up, oneOff: best.oneOff,
              edge: Math.abs(best.side) >= 2.99 || Math.abs(best.up) >= 2.99 });
}

console.log(`맞출 수 있었던 판 ${joint.length} / ${pick.length}\n`);
console.log('맞추는 데 쓰지 않은 **2쿠션 자리**를 얼마나 맞히나 (중앙값)');
console.log(`  1쿠션만 보고 맞춘 당점 (지금 방식)   ${Math.round(mid(alone))} mm`);
console.log(`  1쿠션 + 곡선 + 분리각 (합동)        ${Math.round(mid(joint))} mm`);
let better = 0, worse = 0;
for (let i = 0; i < joint.length; i++) {
  if (joint[i] < alone[i]) better++; else if (joint[i] > alone[i]) worse++;
}
const away = (better + worse) ? Math.abs(better - (better + worse) / 2) / Math.sqrt((better + worse) * 0.25) : 0;
console.log(`\n  합동이 더 가까운 판 ${better} · 더 먼 판 ${worse} · 같은 판 ${joint.length - better - worse}`);
console.log(`  우연으로 보기 어려운 정도 ${away.toFixed(1)} 표준편차`
  + (away > 2 && better > worse ? '  → 합동이 낫다' : (away > 2 ? '  → 1쿠션만이 낫다' : '  → 가릴 수 없다')));

// 격자 끝에 걸린 판과 안쪽에 머문 판을 갈라 본다. 이긴 것이 안쪽 덕이면
// 그 판들의 당점은 믿을 만하고, 끝에 걸린 판의 당점은 쓰레기통이다.
console.log('\n격자 안쪽에 머문 판과 끝에 걸린 판을 갈라서');
for (const [name, want] of [['안쪽 (당점을 믿을 만함)', false], ['끝에 걸림 (못 믿음)', true]]) {
  const idx = tips.map((t, i) => (t.edge === want ? i : -1)).filter((i) => i >= 0);
  if (idx.length < 20) continue;
  const a = idx.map((i) => alone[i]), j = idx.map((i) => joint[i]);
  let up2 = 0, down2 = 0;
  for (const i of idx) { if (joint[i] < alone[i]) up2++; else if (joint[i] > alone[i]) down2++; }
  const sd = (up2 + down2) ? Math.abs(up2 - (up2 + down2) / 2) / Math.sqrt((up2 + down2) * 0.25) : 0;
  console.log(`  ${name.padEnd(24)} n=${String(idx.length).padStart(3)}`
    + `  1쿠션만 ${Math.round(mid(a))}mm → 합동 ${Math.round(mid(j))}mm`
    + `  (좋아짐 ${up2} · 나빠짐 ${down2} · ${sd.toFixed(1)} 표준편차)`);
}

const ups = tips.map((t) => t.up).sort((a, b) => a - b);
const sides = tips.map((t) => t.side).sort((a, b) => a - b);
const q = (a, f) => a[Math.min(a.length - 1, Math.floor(f * a.length))];
console.log(`\n맞춰진 당점`);
console.log(`  좌우  25% ${q(sides, 0.25).toFixed(2)} · 중앙값 ${q(sides, 0.5).toFixed(2)} · 75% ${q(sides, 0.75).toFixed(2)}팁`);
console.log(`  상하  25% ${q(ups, 0.25).toFixed(2)} · 중앙값 ${q(ups, 0.5).toFixed(2)} · 75% ${q(ups, 0.75).toFixed(2)}팁`);
// 격자 끝에 몰리면 맞추기가 모형 오차를 당점으로 떠넘기고 있다는 뜻이다.
const edge = tips.filter((t) => Math.abs(t.side) >= 2.99 || Math.abs(t.up) >= 2.99).length;
console.log(`  격자 끝(±3팁)에 걸린 판 ${edge} (${(edge / tips.length * 100).toFixed(0)}%)`
  + '  ← 많으면 맞추기가 모형 오차를 당점으로 떠넘기는 것이다');
