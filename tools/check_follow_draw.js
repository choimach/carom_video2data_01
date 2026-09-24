// 영상의 밀어치기/끌어치기(spin_y)가 진짜인가 — 시뮬레이터가 맞추는 당점과 견준다.
//
// `follow_draw()`는 파이프라인이 재 놓고도 검증된 적이 없다 ("UNVALIDATED").
// 예전 독립 검사(밀어치기면 충돌 뒤 더 멀리 가야 한다)는 상관 −0.01로 실패했다 —
// 3쿠션은 충돌 뒤 반 초 안에 쿠션을 만나므로 그 자가 통하지 않았다.
//
// 지금은 다른 자가 있다. 프로의 실제 겨냥·강도로 다시 치고 **첫 쿠션 자리에
// 가장 잘 맞는 당점**을 찾으면, 그 상하 성분은 영상과 무관하게 얻은 추정이다.
// 둘이 같은 것을 가리키면 둘 다 진짜다.
//
//   node tools/check_follow_draw.js [판 수]

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

const atClock = (h, t) => [t * Math.sin(h / 12 * 2 * Math.PI), t * Math.cos(h / 12 * 2 * Math.PI)];
const TIPS = [[0, 0]];
for (const h of [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) {
  for (const t of [1, 2, 3]) TIPS.push(atClock(h, t));
}
const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const plays = data.plays.filter((p) => p.aim !== null && p.speed && p.rails
  && p.rails.length >= 1 && p.rise !== null && p.rise !== undefined);
const N = Number(process.argv[2] || 400);
const step = Math.max(1, Math.floor(plays.length / N));
const pick = [];
for (let i = 0; i < plays.length && pick.length < N; i += step) pick.push(plays[i]);

const pairs = [];
for (const play of pick) {
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const rad = play.aim * Math.PI / 180;
  const aim = [Math.cos(rad) * play.speed, Math.sin(rad) * play.speed];
  let best = null;
  for (const [side, up] of TIPS) {
    const shot = SIM.play(layout, 'white', aim, side, up);
    const first = shot.events.find((e) => e.kind === 'cushion');
    if (!first) continue;
    const e = gap(first.p, play.rails[0]);
    if (!best || e < best.e) best = { e, up };
  }
  // 첫 쿠션을 못 맞춘 판은 그 당점도 못 믿는다.
  if (best && best.e <= 60) pairs.push([play.rise, best.up]);
}

const xs = pairs.map((p) => p[0]), ys = pairs.map((p) => p[1]);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const mx = mean(xs), my = mean(ys);
let num = 0, dx = 0, dy = 0;
for (let i = 0; i < xs.length; i++) {
  num += (xs[i] - mx) * (ys[i] - my);
  dx += (xs[i] - mx) ** 2; dy += (ys[i] - my) ** 2;
}
const r = num / Math.sqrt(dx * dy);
console.log(`첫 쿠션을 60 mm 안에 맞춘 판 ${pairs.length}개\n`);
console.log(`  영상의 밀어/끌어와 맞춰진 상하 당점의 상관   ${r.toFixed(3)}`);
console.log('  (0이면 둘 중 하나 이상이 잡음이다. 양수면 같은 것을 가리킨다.)\n');

// 영상이 밀어라고 한 것과 끌어라고 한 것에서 맞춰진 당점이 실제로 갈리는가.
const band = (lo, hi) => ys.filter((_, i) => xs[i] >= lo && xs[i] < hi);
const mid = (a) => (a.length ? [...a].sort((p, q) => p - q)[a.length >> 1] : NaN);
console.log('영상이 말하는 것        판수   맞춰진 상하 당점 중앙값');
for (const [name, lo, hi] of [['끌어치기 (< −0.2)', -99, -0.2], ['스턴 (−0.2~0.2)', -0.2, 0.2],
                              ['밀어치기 (0.2~0.6)', 0.2, 0.6], ['많이 밀어 (> 0.6)', 0.6, 99]]) {
  const b = band(lo, hi);
  console.log(`  ${name.padEnd(20)} ${String(b.length).padStart(4)}          ${mid(b).toFixed(2)}팁`);
}
