// 이웃이 말하는 샷을 이 배치에서 실제로 쳐 보고, 프로가 지나간 길과 재 본다.
//
// tools/carry_shot.py가 만든 build/_carry_jobs.json을 읽는다. 옮겨 오는 것은
// **절대 각도가 아니라 관계**다 — 가까운 공이냐 먼 공이냐, 어느 면, 몇 두께,
// 얼마나 세게. 그 두께로 그 면을 맞히는 겨냥은 **이 배치에서 다시 계산한다.**
//
// 재는 자는 이름이 아니라 궤적이다: 프로의 수구가 지나간 길과 우리가 내놓은 길이
// 몇 mm 떨어져 있는가. 선수가 정한 차례의 ①과 ②를 한꺼번에 재는 셈이다.
//
//   node tools/carry_shot.js

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

const FINE = 0.25;
const CLOCK = [12, 1.5, 3, 4.5, 6, 7.5, 9, 10.5];
const atClock = (h, t) => { const a = (h % 12) / 12 * 2 * Math.PI; return [t * Math.sin(a), t * Math.cos(a)]; };
// 당점은 app-data.json에 없어 이웃에게서 옮길 수 없다. 여기서 채운다.
const TIPS = [[0, 0], ...CLOCK.map((h) => atClock(h, 2))];

// 두 길을 같은 개수의 점으로 고쳐 재고, 점마다의 거리를 평균한다. 길이가 다르면
// 짧은 쪽 길이까지만 — 카메라가 먼저 끊긴 것을 우리 탓으로 돌리지 않는다.
function apart(a, b, steps = 24) {
  const walk = (p) => {
    const d = [0];
    for (let i = 1; i < p.length; i++) d.push(d[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
    return d;
  };
  const da = walk(a), db = walk(b);
  const reach = Math.min(da[da.length - 1], db[db.length - 1]);
  if (reach < 200) return null;
  const at = (p, d, s) => {
    let i = 1; while (i < d.length - 1 && d[i] < s) i++;
    const span = d[i] - d[i - 1] || 1, t = (s - d[i - 1]) / span;
    return [p[i - 1][0] + (p[i][0] - p[i - 1][0]) * t, p[i - 1][1] + (p[i][1] - p[i - 1][1]) * t];
  };
  let sum = 0;
  for (let k = 0; k <= steps; k++) {
    const s = reach * k / steps;
    const [ax, ay] = at(a, da, s), [bx, by] = at(b, db, s);
    sum += Math.hypot(ax - bx, ay - by);
  }
  return sum / (steps + 1);
}

// 이웃이 준 (공·면·두께·세기)로 이 배치에서 겨냥을 세우고, 그 둘레를 다듬는다.
function play(layout, carried) {
  const cue = layout.white;
  const gaps = ['yellow', 'red'].map((c) => Math.hypot(layout[c][0] - cue[0], layout[c][1] - cue[1]));
  const first = carried.near === (gaps[0] <= gaps[1]) ? 'yellow' : 'red';
  const to = layout[first];
  const reach = Math.hypot(to[0] - cue[0], to[1] - cue[1]);
  if (reach < 70) return null;
  const straight = Math.atan2(to[1] - cue[1], to[0] - cue[0]) * 180 / Math.PI;
  const thickness = carried.thickness == null ? 0.35 : carried.thickness;
  const offset = SIM.DIAMETER * (1 - thickness);
  const swing = Math.asin(Math.min(1, offset / reach)) * 180 / Math.PI;
  // 왼쪽 면을 맞히려면 수구가 적구의 왼쪽으로 지나가야 한다.
  const aimed = straight + (carried.face === 'left' ? -swing : swing);

  const speeds = carried.speed
    ? [carried.speed * 0.8, carried.speed, carried.speed * 1.25]
    : [3.5, 4.5, 5.5, 7].map((s) => SIM.speedFor(s));
  let best = null;
  for (const speed of speeds) {
    for (const [side, up] of TIPS) {
      // 두께는 이웃의 중앙값일 뿐이므로 ±3도를 훑는다. 이 배치에서 실제로
      // 들어가는 선을 찾는 것이 목적이고, 그것이 "궤적은 이 배치에서"다.
      for (let d = -3; d <= 3; d += FINE) {
        const deg = aimed + d;
        const r = deg * Math.PI / 180;
        const shot = SIM.play(layout, 'white', [Math.cos(r) * speed, Math.sin(r) * speed], side, up);
        const judged = SIM.judge(shot);
        if (!judged.scored || judged.first !== first) continue;
        const across = Math.abs(Math.cos(r) * (to[1] - cue[1]) - Math.sin(r) * (to[0] - cue[0]));
        const got = Math.max(0, Math.min(1, 1 - across / SIM.DIAMETER));
        // 이웃이 말한 두께에 가까운 것을 고른다 — 같은 값을 두 번 쓰지만,
        // 하나는 겨냥을 세우는 데, 하나는 고르는 데다.
        const cost = Math.abs(got - thickness);
        if (!best || cost < best.cost) best = { shot, cost, deg, speed, side, up, thickness: got };
      }
    }
  }
  return best;
}

function main() {
  const jobs = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', '_carry_jobs.json'), 'utf8')).jobs;
  const gaps = [], firstRight = [], faceRight = [], rightGaps = [], wrongGaps = [];
  let played = 0;
  for (const job of jobs) {
    const layout = { white: job.layout.cue, yellow: job.layout.balls[0], red: job.layout.balls[1] };
    // 이웃이 가장 무겁게 미는 하나만 쓴다 — 화면에 궤적 하나를 보여주는 것이
    // 목표이므로, 첫 답이 맞는지가 바로 그 값이다.
    const top = job.carried[0];
    if (!top) continue;
    const got = play(layout, top);
    if (!got) continue;
    played++;
    firstRight.push(top.near === job.truth.near ? 1 : 0);
    faceRight.push((top.near === job.truth.near && top.face === job.truth.face) ? 1 : 0);
    const d = apart(got.shot.paths.white, job.path);
    if (d !== null) {
      gaps.push(d);
      const right = top.near === job.truth.near && top.face === job.truth.face;
      (right ? rightGaps : wrongGaps).push(d);
    }
  }
  const mid = (a) => a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : NaN;
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  console.log(`일감 ${jobs.length}개 중 득점하는 선을 찾은 것 ${played}개\n`);
  console.log(`  1적구를 맞혔나 (가까운/먼)   ${(mean(firstRight) * 100).toFixed(0)}%`);
  console.log(`  1적구 + 면까지 맞혔나        ${(mean(faceRight) * 100).toFixed(0)}%`);
  console.log(`\n  프로의 수구가 간 길과의 거리 (${gaps.length}개)`);
  console.log(`    중앙값 ${mid(gaps).toFixed(0)} mm · 평균 ${mean(gaps).toFixed(0)} mm`);
  const sorted = [...gaps].sort((a, b) => a - b);
  for (const q of [0.25, 0.5, 0.75, 0.9]) {
    console.log(`    ${(q * 100).toFixed(0)}%  ${sorted[Math.floor(q * sorted.length)].toFixed(0)} mm`);
  }
  console.log(`\n  공과 면을 맞게 골랐을 때 (${rightGaps.length}개)  중앙값 ${mid(rightGaps).toFixed(0)} mm`);
  console.log(`  틀리게 골랐을 때      (${wrongGaps.length}개)  중앙값 ${mid(wrongGaps).toFixed(0)} mm`);
  console.log('\n  당구대 장축이 2844 mm, 공 하나가 62 mm다.');
}

main();
