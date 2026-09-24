// 밀어치기/끌어치기(spin_y)가 진짜인가 — **속도**로 따로 잰다.
//
// `spin_y`는 수구가 충돌 뒤 **어느 쪽으로** 떠나는지(각도)에서 나온 값이다.
// 충돌 뒤 **얼마나 빠른지**(속도)는 같은 영상에서 따로 읽히는 값이고, 물리가
// 옳다면 둘이 같이 움직여야 한다 — 밀어치기는 각도를 중심선 쪽으로 눕히면서
// 속도도 더 남기고, 끌어치기는 반대다. 서로 다른 관측이므로 순환이 아니다.
//
// 앞서 두 검사는 힘이 없었다: (1) 충돌 뒤 이동거리는 쿠션이 가로막고,
// (2) 첫 쿠션 자리로 상하 당점을 맞추면 거의 항상 0이 나온다 — 첫 쿠션은
// 상하 당점을 식별하지 못한다.
//
//   node tools/check_follow_speed.js

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));

// 저장된 경로는 프레임 간격으로 고르게 솎았으므로 점 사이 거리가 곧 속도다.
const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const rows = [];
for (const play of data.plays) {
  if (play.rise === null || play.rise === undefined || !play.hit || !play.path) continue;
  const way = play.path;
  if (way.length < 10) continue;
  let at = 0, best = Infinity;
  for (let i = 0; i < way.length; i++) {
    const d = gap(way[i], play.hit);
    if (d < best) { best = d; at = i; }
  }
  // 접촉점이 경로 위에 제대로 잡혀야 하고, 앞뒤로 잴 자리가 있어야 한다.
  if (best > 120 || at < 3 || at > way.length - 4) continue;
  const before = (gap(way[at - 3], way[at - 2]) + gap(way[at - 2], way[at - 1])) / 2;
  const after = (gap(way[at + 1], way[at + 2]) + gap(way[at + 2], way[at + 3])) / 2;
  if (before < 8) continue;                      // 거의 멈춘 공은 비율이 믿기 어렵다
  rows.push({ rise: play.rise, kept: after / before, thick: play.thick });
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const mid = (a) => (a.length ? [...a].sort((p, q) => p - q)[a.length >> 1] : NaN);
function corr(xs, ys) {
  const mx = mean(xs), my = mean(ys);
  let n = 0, dx = 0, dy = 0;
  for (let i = 0; i < xs.length; i++) {
    n += (xs[i] - mx) * (ys[i] - my); dx += (xs[i] - mx) ** 2; dy += (ys[i] - my) ** 2;
  }
  return n / Math.sqrt(dx * dy);
}
console.log(`접촉 앞뒤 속도를 읽을 수 있는 판 ${rows.length}개\n`);
console.log(`  밀어/끌어와 "충돌 뒤 남은 속도 비율"의 상관   ${corr(rows.map((r) => r.rise), rows.map((r) => r.kept)).toFixed(3)}`);
console.log('  (물리가 옳고 둘 다 진짜면 양수여야 한다.)\n');
console.log('영상이 말하는 것        판수   충돌 뒤 남은 속도 비율 (중앙값)');
for (const [name, lo, hi] of [['끌어치기 (< −0.2)', -99, -0.2], ['스턴 (−0.2~0.2)', -0.2, 0.2],
                              ['밀어치기 (0.2~0.6)', 0.2, 0.6], ['많이 밀어 (> 0.6)', 0.6, 99]]) {
  const b = rows.filter((r) => r.rise >= lo && r.rise < hi);
  console.log(`  ${name.padEnd(20)} ${String(b.length).padStart(4)}          ${mid(b.map((r) => r.kept)).toFixed(3)}`);
}
// 두께가 비율을 크게 좌우하므로 두께를 묶어 놓고도 본다.
const withT = rows.filter((r) => r.thick !== null && r.thick !== undefined);
console.log(`\n두께를 묶어 놓고 (두께가 있는 ${withT.length}판)`);
console.log('  두께대         판수   밀어/끌어와의 상관');
for (const [lo, hi] of [[0, 0.2], [0.2, 0.35], [0.35, 0.55], [0.55, 1.01]]) {
  const b = withT.filter((r) => r.thick >= lo && r.thick < hi);
  if (b.length < 40) continue;
  console.log(`  ${lo.toFixed(2)}~${hi.toFixed(2)}    ${String(b.length).padStart(4)}        `
    + corr(b.map((r) => r.rise), b.map((r) => r.kept)).toFixed(3));
}
