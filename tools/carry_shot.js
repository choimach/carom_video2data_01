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

// 궤적을 **고리별로** 읽는다. 선수가 정한 차례 (2026-09-21): 반사각 > 첫 쿠션
// 포인트 > 그리로 가는 곡선 > 총 이동거리 > 둘째 쿠션 포인트. 앞의 고리일수록
// 무겁다 — 오차가 뒤로 갈수록 불어나기 때문이다. 전체 평균 거리 하나로 재면
// 어느 고리가 끊어졌는지 가려진다.
function links(path, firstBall) {
  // 쿠션은 "쿠션 가까이 있는 점"으로 찾지 않는다. 프로의 길은 32점으로 솎아져
  // 있어 점 간격이 200 mm인데 쿠션 띠는 49 mm라, 대부분 놓친다. 대신 **방향이
  // 뒤집히는 자리**로 찾는다 — 쿠션을 맞으면 x나 y의 부호가 바뀌고, 그건 성기게
  // 솎아도 남는다.
  function turns(p, from) {
    const out = [];
    for (let i = from + 2; i < p.length - 1; i++) {
      for (const ax of [0, 1]) {
        const before = p[i][ax] - p[i - 1][ax], after = p[i + 1][ax] - p[i][ax];
        if (Math.abs(before) < 20 || Math.abs(after) < 20) continue;
        if ((before > 0) === (after > 0)) continue;
        // 방향이 뒤집혔다. 그 축에서 쿠션 쪽에 있어야 쿠션이다.
        const wall = ax === 0 ? SIM.L : SIM.W;
        if (p[i][ax] > wall * 0.25 && p[i][ax] < wall * 0.75) continue;
        if (out.length && i - out[out.length - 1].i < 2) continue;
        out.push({ i, at: p[i] });
      }
      if (out.length >= 2) break;
    }
    return out;
  }
  // 1적구에 가장 가까이 간 자리를 충돌 지점으로 본다. 얇은 길은 어차피 스치므로
  // 이보다 나은 것을 이 32점짜리 길에서 뽑아내기 어렵다.
  let hit = 0, near = Infinity;
  for (let i = 0; i < path.length; i++) {
    const d = Math.hypot(path[i][0] - firstBall[0], path[i][1] - firstBall[1]);
    if (d < near) { near = d; hit = i; }
  }
  // 충돌 뒤 떠나는 방향 — 반사각의 방향과 크기가 여기서 나온다.
  const after = Math.min(hit + 2, path.length - 1);
  const leave = (after > hit)
    ? Math.atan2(path[after][1] - path[hit][1], path[after][0] - path[hit][0]) * 180 / Math.PI
    : null;
  // 충돌 뒤 처음으로 쿠션에 닿는 자리, 그 다음 자리.
  const found = turns(path, hit);
  const rails = found.map((f) => f.at);
  // 충돌부터 첫 쿠션까지가 얼마나 휘었나: 직선 거리 대비 실제로 간 거리.
  let curve = null;
  if (rails.length) {
    let gone = 0, upto = found[0].i;
    for (let i = hit + 1; i <= upto; i++) gone += Math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]);
    const straight = Math.hypot(rails[0][0] - path[hit][0], rails[0][1] - path[hit][1]);
    if (straight > 100) curve = gone / straight;
  }
  let travel = 0;
  for (let i = 1; i < path.length; i++) travel += Math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]);
  return { hitAt: path[hit], leave, rail1: rails[0] || null, rail2: rails[1] || null, curve, travel };
}

// 접촉 자리와 쿠션 자리를 알고 있을 때, 선수가 정한 다섯 고리를 뽑는다.
function fromEvents(path, hit, rails) {
  let travel = 0;
  for (let i = 1; i < path.length; i++) {
    travel += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  }
  const rail1 = rails[0] || null, rail2 = rails[1] || null;
  // 1. 반사각의 방향 — 1적구를 맞고 첫 쿠션으로 떠나는 방향.
  const leave = (hit && rail1)
    ? Math.atan2(rail1[1] - hit[1], rail1[0] - hit[0]) * 180 / Math.PI : null;
  // 3. 그 사이가 얼마나 휘었나 — 실제로 간 거리 ÷ 직선 거리. 1.0이면 곧은 길.
  let curve = null;
  if (hit && rail1) {
    let gone = 0, started = false;
    for (let i = 1; i < path.length; i++) {
      const d0 = Math.hypot(path[i - 1][0] - hit[0], path[i - 1][1] - hit[1]);
      const d1 = Math.hypot(path[i][0] - rail1[0], path[i][1] - rail1[1]);
      if (!started && d0 < 120) started = true;
      if (started) gone += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
      if (started && d1 < 120) break;
    }
    const straight = Math.hypot(rail1[0] - hit[0], rail1[1] - hit[1]);
    if (straight > 200 && gone > 0) curve = gone / straight;
  }
  return { leave, rail1, rail2, curve, travel };
}

const angleGap = (a, b) => {
  if (a === null || b === null) return null;
  let d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
};
const pointGap = (a, b) => (a && b) ? Math.hypot(a[0] - b[0], a[1] - b[1]) : null;

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
  const link = { leave: [], rail1: [], curve: [], travel: [], rail2: [] };
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
      // 고리별 오차는 **고르기가 맞았을 때만** 본다. 틀린 공을 친 궤적의 반사각을
      // 재는 것은 뜻이 없다.
      if (right) {
        // 우리 쪽은 시뮬레이터가 이벤트로 알려 주고, 프로 쪽은 파이프라인이
        // 실어 보낸 값을 쓴다. 양쪽 다 추측이 아니다.
        const ev = got.shot.events;
        const ball = ev.find((e) => e.kind === 'ball');
        const cush = ev.filter((e) => e.kind === 'cushion' && ball && e.at > ball.at);
        const mine = fromEvents(got.shot.paths.white, ball && ball.p, cush.map((e) => e.p));
        const theirs = fromEvents(job.path, job.hit, job.rails || []);
        const put = (k, v) => { if (v !== null && Number.isFinite(v)) link[k].push(v); };
        put('leave', angleGap(mine.leave, theirs.leave));
        put('rail1', pointGap(mine.rail1, theirs.rail1));
        put('curve', (mine.curve && theirs.curve) ? Math.abs(mine.curve - theirs.curve) : null);
        put('travel', Math.abs(mine.travel - theirs.travel));
        put('rail2', pointGap(mine.rail2, theirs.rail2));
      }
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
  console.log('\n  고리별로 (고르기가 맞은 것만) — 앞의 고리일수록 무겁다');
  const unit = { leave: '도', rail1: 'mm', curve: '배', travel: 'mm', rail2: 'mm' };
  const name = { leave: '1. 반사각 방향', rail1: '2. 첫 쿠션 자리', curve: '3. 곡선',
                 travel: '4. 총 이동거리 ⚠️', rail2: '5. 둘째 쿠션 자리' };
  for (const k of ['leave', 'rail1', 'curve', 'travel', 'rail2']) {
    const v = link[k];
    if (!v.length) { console.log(`    ${name[k]}  —`); continue; }
    console.log(`    ${name[k].padEnd(16)} 중앙값 ${mid(v).toFixed(k === 'curve' ? 2 : 0)} ${unit[k]}  (n=${v.length})`);
  }
  console.log('\n  ⚠️ 4번은 믿지 말 것. 영상의 궤적은 카메라 창이 닫히며 잘린다 —');
  console.log('     추적된 공의 대부분이 창이 닫힐 때 아직 구르고 있었다. 끝까지');
  console.log('     굴린 시뮬레이터와 견주면 우리가 더 간 것처럼 보일 뿐이다.');
  console.log('     이 함정에 한 번 걸려 감쇠곡선을 잘못 맞춘 적이 있다.');
  console.log('\n  당구대 장축이 2844 mm, 공 하나가 62 mm다.');
}

main();
