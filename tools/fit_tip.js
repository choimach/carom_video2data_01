// 프로가 준 당점을 관측 여럿으로 한꺼번에 맞춘다 (선수 제안, 2026-09-25).
//
// ★★2026-10-01: 아래 옛 방식은 **좌우를 못 가렸다** — 내보내는 것은 이제 v2다.
//   "1쿠션 자리는 좌우 당점에 강하다"는 틀렸다. 좌우는 쿠션에서 **나가는 방향**을 바꾸지
//   **닿는 자리**를 바꾸지 않는다: 좌우 −3→+3팁에 첫 쿠션 자리는 중앙값 21 mm만 움직였다
//   (어긋남 74 mm). 비용이 좌우에 평평하니 동점이면 격자를 먼저 훑은 −3팁이 이겼고,
//   −3에 100판 · +3에 0판이 붙어 좌우 중앙값 −1.25팁으로 치우쳤다. 이 값이 조언판의
//   "가까운 프로 n명은 ○시를 줬습니다"에 들어가 있었다 — 선수: "좌우가 바뀌는 것".
//   v2: 쿠션을 양쪽 다 처음부터 세어 **첫째·둘째 쿠션 자리** + 곡선 + 분리각을 맞추고,
//   동점이면 가운데(작은 벌점). 셋째 쿠션(안 씀)으로 채점, 600판:
//                   좌우 중앙값   ±3 끝   셋째 쿠션
//     옛 방식        −1.25팁      29%     212 mm
//     v2             0.00팁        0%      93 mm   (가까워진 판 154 · 멀어진 판 68)
//   ⚠️ app-data의 `spin`(첫 쿠션 반사 각의 틀어짐)은 좌우 부호가 아니다 — 어느 맞추기와도
//   부호 일치가 50%다. 부호는 `tools/check_side_sign.py`(english_side)로 잰다.
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
//   node tools/fit_tip.js [판 수]        # 채점만 한다
//   node tools/fit_tip.js --export       # 전부 맞춰 data/tips.json에 적는다
//
// `--export`가 적은 것을 `tools/app_data.py`가 읽어 각 플레이에 `tip`(좌우·상하
// 팁)과 `tip_ok`(믿을 만한가)로 붙인다. 따로 파일에 두는 이유: app-data를 다시
// 만들어도 살아남고, 이 비싼 계산이 파이썬 경로에 들어가지 않는다.

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
  // 쿠션 전부, 처음부터 (1적구 앞 쿠션 포함) — 영상의 rails와 같은 정의.
  return { one: one.p, two: two ? two.p : null, curve, rise, all: rails.map((e) => e.p) };
}

const EXPORT = process.argv.includes('--export');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
// 채점에는 2쿠션이 필요하지만(검증용), 내보낼 때는 1쿠션만 있어도 맞출 수 있다.
const plays = data.plays.filter((p) => p.aim !== null && p.speed && p.hit
  && p.rails && p.rails.length >= (EXPORT ? 1 : 2)
  && p.curve && p.rise !== null && p.rise !== undefined);
let pick;
if (EXPORT) {
  pick = plays;
} else {
  const N = Number(process.argv[2] || 250);
  const step = Math.max(1, Math.floor(plays.length / N));
  pick = [];
  for (let i = 0; i < plays.length && pick.length < N; i += step) pick.push(plays[i]);
}
console.log(`맞출 판 ${pick.length}개${EXPORT ? ' (전부)' : ''}`);

// 잔차를 견줄 수 있게 각자의 보통 크기로 나눈다.
const SCALE = { one: 40, two: 60, curve: 20, rise: 0.15 };
const TIPS = [];
for (let s = -3; s <= 3.0001; s += 0.25) {
  for (let u = -3; u <= 3.0001; u += 0.25) TIPS.push([Math.round(s * 100) / 100, Math.round(u * 100) / 100]);
}

const joint = [], alone = [];
// DUMP=파일: 채점할 때 판별 맞춤(합동·1쿠션만)을 적는다 — tips.json은 안 건드린다.
const dumped = {};
const tips = [];
const out = {};
let done = 0;
const began = Date.now();
for (const play of pick) {
  if (EXPORT && ++done % 100 === 0) {
    const each = (Date.now() - began) / done / 1000;
    process.stdout.write(`  ${done}/${pick.length} · 판당 ${each.toFixed(2)}초`
      + ` · 남은 시간 ${Math.round((pick.length - done) * each / 60)}분\n`);
  }
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const rad = play.aim * Math.PI / 180;
  const aim = [Math.cos(rad) * play.speed, Math.sin(rad) * play.speed];
  const first = gap(play.balls[0], play.hit) < gap(play.balls[1], play.hit) ? 'yellow' : 'red';
  let best = null, only = null, best2 = null;
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
    // v2 (2026-10): 좌우는 쿠션에 **닿는 자리**가 아니라 **나가는 방향**에서 드러난다.
    // 첫 쿠션 자리는 좌우 −3→+3팁에 중앙값 21 mm만 움직여(어긋남은 74 mm) 비용이 평평했고,
    // 동점이면 먼저 훑은 −3팁이 이겨 −3에 100판 · +3에 0판이 붙었다. 그래서 둘째 쿠션까지
    // 넣고(셋째로 검증), 쿠션은 양쪽 다 처음부터 센다(옛 방식은 영상은 1적구 앞 쿠션부터,
    // 시뮬은 1적구 뒤부터 세어 뱅크샷에서 다른 쿠션을 견줬다). 동점이면 가운데 — 작은 벌점.
    if (got.all.length >= 2 && play.rails.length >= 2) {
      const c2 = (gap(got.all[0], play.rails[0]) / SCALE.one) ** 2
        + (gap(got.all[1], play.rails[1]) / SCALE.two) ** 2
        + ((got.curve - play.curve[0]) / SCALE.curve) ** 2
        + ((got.rise - play.rise) / SCALE.rise) ** 2
        + 1e-3 * (side * side + up * up);
      if (!best2 || c2 < best2.cost) {
        best2 = { cost: c2, side, up, three: got.all[2] || null,
                  d1: gap(got.all[0], play.rails[0]), d2: gap(got.all[1], play.rails[1]) };
      }
    }
  }
  if (EXPORT && best2) {
    // v2로 내보낸다 (2026-10). 믿을 만한가 = 두 쿠션을 다 맞췄나.
    out[`${play.match}:${play.inning}:${play.shot}`] = {
      tip: [best2.side, best2.up],
      ok: best2.d1 <= 60 && best2.d2 <= 150,
      one: Math.round(best2.d1), two: Math.round(best2.d2), fit: 'v2',
    };
    continue;
  }
  if (EXPORT) {
    // 쿠션이 하나뿐인 판 — v2를 못 쓴다. 옛 방식은 좌우를 못 가리므로(위 주석) 상하만 믿고
    // 좌우는 0으로 둔다. 믿을 만한 것으로 치지 않는다.
    if (!best) continue;
    out[`${play.match}:${play.inning}:${play.shot}`] = { tip: [0, best.up], ok: false, one: Math.round(best.oneOff), fit: 'up-only' };
    continue;
  }
  if (EXPORT) {
    if (!best) continue;
    const edge = Math.abs(best.side) >= 2.99 || Math.abs(best.up) >= 2.99;
    out[`${play.match}:${play.inning}:${play.shot}`] = {
      // 좌우·상하 팁. 믿을 만한가는 **격자 끝에 걸렸는지**로 정한다 — 끝에
      // 걸린 판은 어떤 물리적 당점으로도 설명되지 않아, 맞추기가 모형 오차를
      // 당점으로 떠넘긴 것이다 (2026-09-25에 그렇게 갈렸다).
      tip: [best.side, best.up],
      ok: !edge && best.oneOff <= 60,
      one: Math.round(best.oneOff),
    };
    continue;
  }
  if (!best || !only || !best.two || !only.two) continue;
  if (only.off > 60) continue;                     // 1쿠션도 못 맞춘 판은 뺀다
  joint.push(gap(best.two, play.rails[1]));
  alone.push(gap(only.two, play.rails[1]));
  tips.push({ side: best.side, up: best.up, oneOff: best.oneOff,
              edge: Math.abs(best.side) >= 2.99 || Math.abs(best.up) >= 2.99 });
  // 셋째 쿠션(어느 맞추기에도 안 쓴다)으로 옛 방식과 v2를 견준다.
  let threeOld = null;
  if (play.rails.length >= 3) {
    const shot = SIM.play(layout, 'white', aim, best.side, best.up);
    const r = shot.events.filter((e) => e.kind === 'cushion');
    if (r.length >= 3) threeOld = Math.round(gap(r[2].p, play.rails[2]));
  }
  dumped[`${play.match}:${play.inning}:${play.shot}`] = { side: best.side, up: best.up,
    oneOff: Math.round(best.oneOff), onlySide: only.side, onlyUp: only.up,
    v2: best2 && { side: best2.side, up: best2.up,
                   three: best2.three && play.rails.length >= 3 ? Math.round(gap(best2.three, play.rails[2])) : null },
    threeOld };
}

if (EXPORT) {
  const where = path.join(ROOT, 'data', 'tips.json');
  fs.writeFileSync(where, JSON.stringify(out));
  const ok = Object.values(out).filter((r) => r.ok).length;
  console.log(`\n맞춘 판 ${Object.keys(out).length}개 · 믿을 만한 것 ${ok}개`
    + ` (${(ok / Object.keys(out).length * 100).toFixed(0)}%) -> ${where}`);
  process.exit(0);
}

if (process.env.DUMP) fs.writeFileSync(process.env.DUMP, JSON.stringify(dumped));
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
