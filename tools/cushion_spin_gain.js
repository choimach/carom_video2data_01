// 쿠션에서 회전 1팁이 반사각을 몇 도 틀어 놓는가.
// 방향은 경로 점이 아니라 **쿠션 자리 자체**로 읽는다: 들어온 방향은
// 출발점→1쿠션, 나간 방향은 1쿠션→2쿠션. 성긴 경로 점에 흔들리지 않는다.
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');
// ⚠️ 적구는 쏘는 쿠션에서 **멀리** 치운다. 한때 y=60에 두었다가 수구가
// 쿠션이 아니라 공을 맞아, 회전이 1팁에서 포화한다는 엉뚱한 결론을 냈다.
const layout = { white: [1422, 900], yellow: [40, 1380], red: [2800, 1380] };
const cue = 'white';
const ang = (v) => Math.atan2(v[1], v[0]) * 180 / Math.PI;

function turn(incidence, strength, side) {
  const v = SIM.speedFor(strength);
  const deg = 90 - incidence;
  const rad = deg * Math.PI / 180;
  const shot = SIM.play(layout, cue, [Math.cos(rad) * v, Math.sin(rad) * v], side, 0);
  const cs = shot.events.filter((e) => e.kind === 'cushion');
  if (cs.length < 2) return null;
  const a = layout[cue], b = cs[0].p, c = cs[1].p;
  const inDir = [b[0] - a[0], b[1] - a[1]];
  const outDir = [c[0] - b[0], c[1] - b[1]];
  // 1쿠션이 어느 벽인지 보고 그 축으로 거울을 잡는다.
  const flipY = Math.abs(b[1]) < 60 || Math.abs(b[1] - SIM.W) < 60;
  const mirror = flipY ? [inDir[0], -inDir[1]] : [-inDir[0], inDir[1]];
  let d = ang(outDir) - ang(mirror);
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

const TIPS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3];
console.log('팁을 늘리면 반사각이 어떻게 움직이나 (거울 대비, 강도 5.5)\n');
console.log('입사각 ' + TIPS.map((t) => (t + '팁').padStart(7)).join(''));
for (const inc of [25, 40, 55]) {
  const v = TIPS.map((t) => turn(inc, 5.5, t));
  if (v.some((x) => x === null)) continue;
  console.log(`  ${String(inc).padStart(2)}도 ` + v.map((x) => ((x >= 0 ? '+' : '') + x.toFixed(1)).padStart(7)).join(''));
}
console.log('\n한 팁 늘릴 때마다 몇 도 (기울기)');
for (const inc of [25, 40, 55]) {
  const v = TIPS.map((t) => turn(inc, 5.5, t));
  if (v.some((x) => x === null)) continue;
  const g = [];
  for (let i = 1; i < TIPS.length; i++) g.push(((v[i] - v[i-1]) / (TIPS[i] - TIPS[i-1])).toFixed(1));
  console.log(`  ${String(inc).padStart(2)}도 ` + g.map((x) => x.padStart(7)).join(''));
}
