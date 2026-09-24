// 영상의 밀어/끌어(spin_y)에서 **프로가 준 상하 당점**을 되찾는다.
//
// `spin_y = dot(수구 출발방향, 중심선)` = **분리각의 코사인**이다 (스턴 0,
// 밀어치기 양수, 끌어치기 음수). 시뮬레이터에서 같은 값을 계산할 수 있으므로,
// 프로의 실제 겨냥·강도로 치면서 상하 당점만 바꿔 영상의 값과 맞추면 된다.
//
// ⚠️⚠️ **안 된다. 2026-09-25에 해 보고 접었다.** 이 파일은 왜 안 되는지를
// 남겨 두려고 둔다 — `--sensitivity`로 그 근거를 다시 볼 수 있다.
//
// 당점을 −3팁에서 +3팁까지 **끝에서 끝까지** 바꿔도 분리각의 코사인이 움직이는
// 폭은:
//
//     두께 0.84 (두껍게)   0.23~0.55   식별 가능
//     두께 0.59            0.07~0.16   약하다
//     두께 0.27 (얇게)     0.03~0.05   전혀 안 된다
//
// 게다가 두꺼운 구간에서도 **단조롭지 않다**: −3팁 0.01 · −1팁 0.08 · 0팁 0.56 ·
// +1팁 0.54 · +3팁 0.52. **0팁 위로는 평평하다** — 공이 이미 구르고 있으면
// 상단을 더 줘도 분리각이 바뀌지 않는다. 끌어치기만 갈린다.
//
// 그리고 **프로의 두께는 중앙값 0.32, 90%가 0.56 이하**다. 식별 가능한 구간에
// 드는 판이 사실상 없다.
//
// 그래도 돌려 보면 65%가 영상 값에 0.08 안으로 맞고 당점이 −2.75~+2.50팁으로
// 퍼진다. **그것은 잡음을 맞춘 것이다** — 민감도가 없는데 답이 나오면 그 답은
// 자료가 아니라 격자가 만든 것이다.
//
// 처음 생각은 이랬다: 충돌 순간의 구름 상태는 〈당점 높이〉 + 〈거리에서 저절로
// 생긴 구름〉인데, 프로 샷은 수구→1적구가 중앙값 593 mm로 저절로 구르기에는
// 너무 짧으니(강도 5.5면 3,620 mm 필요) 남는 것이 당점이다. 그 전제는 맞다.
// 틀린 것은 **분리각이 당점을 되비쳐 주지 않는다**는 쪽이었다.
//
//   node tools/recover_tip.js [판 수]
//   node tools/recover_tip.js --sensitivity

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

// 시뮬레이션에서 분리각의 코사인을 읽는다 — follow_draw()와 같은 정의로.
function cosSeparation(shot, cue, first) {
  const ball = shot.events.find((e) => e.kind === 'ball');
  if (!ball) return null;
  const cueWay = shot.paths[cue], objWay = shot.paths[first];
  if (!cueWay || !objWay) return null;
  // 접촉 자리에 가장 가까운 프레임을 찾는다.
  let at = 0, best = Infinity;
  for (let i = 0; i < cueWay.length; i++) {
    const d = Math.hypot(cueWay[i][0] - ball.p[0], cueWay[i][1] - ball.p[1]);
    if (d < best) { best = d; at = i; }
  }
  const span = 4;
  if (at + span >= cueWay.length) return null;
  const unit = (a, b) => {
    const v = [b[0] - a[0], b[1] - a[1]];
    const n = Math.hypot(v[0], v[1]);
    return n < 1e-6 ? null : [v[0] / n, v[1] / n];
  };
  // 적구가 떠나는 방향이 곧 중심선이다.
  const centres = unit(objWay[at], objWay[at + span]);
  const cueOut = unit(cueWay[at + 1], cueWay[at + 1 + span]);
  if (!centres || !cueOut) return null;
  return cueOut[0] * centres[0] + cueOut[1] * centres[1];
}

if (process.argv.includes('--sensitivity')) {
  console.log('상하 당점을 −3에서 +3까지 바꾸면 분리각의 코사인이 얼마나 움직이나\n');
  console.log('  거리   강도   두께      −3팁    −1팁     0팁    +1팁    +3팁    폭');
  for (const dist of [300, 600, 1200]) {
    for (const st of [3.5, 5.5]) {
      for (const off of [10, 25, 45]) {
        const layout = { white: [200, 711], yellow: [200 + dist, 711], red: [2700, 120] };
        const v = SIM.speedFor(st), ang = Math.asin(off / dist);
        const aim = [Math.cos(ang) * v, Math.sin(ang) * v];
        const got = [-3, -1, 0, 1, 3].map((up) =>
          cosSeparation(SIM.play(layout, 'white', aim, 0, up), 'white', 'yellow'));
        if (got.some((x) => x === null)) continue;
        console.log(`  ${String(dist).padStart(4)}mm  ${st.toFixed(1)}   ${(1 - off / 61.5).toFixed(2)}   `
          + got.map((x) => x.toFixed(2).padStart(7)).join('')
          + `   ${(Math.max(...got) - Math.min(...got)).toFixed(2)}`);
      }
    }
  }
  console.log('\n  프로의 두께는 중앙값 0.32, 90%가 0.56 이하다 — 식별 가능한 구간에');
  console.log('  드는 판이 사실상 없다.');
  process.exit(0);
}

const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const plays = data.plays.filter((p) => p.aim !== null && p.speed && p.hit
  && p.rise !== null && p.rise !== undefined);
const N = Number(process.argv[2] || 400);
const step = Math.max(1, Math.floor(plays.length / N));
const pick = [];
for (let i = 0; i < plays.length && pick.length < N; i += step) pick.push(plays[i]);

const UPS = [];
for (let u = -3; u <= 3.0001; u += 0.25) UPS.push(Math.round(u * 100) / 100);

const out = [];
for (const play of pick) {
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const rad = play.aim * Math.PI / 180;
  const aim = [Math.cos(rad) * play.speed, Math.sin(rad) * play.speed];
  // 어느 공을 먼저 맞았는지는 접촉 자리로 안다.
  const first = Math.hypot(play.balls[0][0] - play.hit[0], play.balls[0][1] - play.hit[1])
    < Math.hypot(play.balls[1][0] - play.hit[0], play.balls[1][1] - play.hit[1]) ? 'yellow' : 'red';
  let best = null;
  for (const up of UPS) {
    const shot = SIM.play(layout, 'white', aim, 0, up);
    const got = cosSeparation(shot, 'white', first);
    if (got === null) continue;
    const off = Math.abs(got - play.rise);
    if (!best || off < best.off) best = { off, up, got };
  }
  if (best) out.push({ rise: play.rise, up: best.up, off: best.off, play });
}

const mid = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
console.log(`되찾을 수 있었던 판 ${out.length} / ${pick.length}\n`);
const close = out.filter((r) => r.off < 0.08);
console.log(`  영상 값에 0.08 안으로 맞춘 판   ${close.length} (${(close.length / out.length * 100).toFixed(0)}%)`);
console.log(`  못 맞춘 판의 남은 차이 중앙값    ${mid(out.filter((r) => r.off >= 0.08).map((r) => r.off)).toFixed(2)}\n`);
console.log('되찾은 상하 당점 (0.08 안으로 맞은 판만)');
const ups = close.map((r) => r.up).sort((a, b) => a - b);
const q = (f) => ups[Math.min(ups.length - 1, Math.floor(f * ups.length))];
console.log(`  10% ${q(0.1).toFixed(2)}팁 · 25% ${q(0.25).toFixed(2)}팁 · 중앙값 ${q(0.5).toFixed(2)}팁`
  + ` · 75% ${q(0.75).toFixed(2)}팁 · 90% ${q(0.9).toFixed(2)}팁`);
const band = (lo, hi) => close.filter((r) => r.up >= lo && r.up < hi).length;
console.log(`\n  끌어치기(< −0.5) ${band(-9, -0.5)} · 중단(−0.5~0.5) ${band(-0.5, 0.5)}`
  + ` · 상단(0.5~2) ${band(0.5, 2)} · 많이 상단(≥2) ${band(2, 9)}`);
console.log(`\n  선수 본인은 "대개 상단 2팁"이라고 했다 (2026-09-22).`);
