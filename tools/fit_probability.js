// 오차의 크기를 프로의 실제 득점률에 맞추고, 그렇게 잰 확률이 쓸모 있는지 본다.
//
// 손으로 "겨냥 오차는 0.3도쯤" 하고 정하면 그건 내 짐작이지 측정이 아니다.
// 대신 **프로 2,150판이 실제로 넣은 비율(61%)에 맞도록** 오차를 키우거나
// 줄인다. 그러면 남은 질문은 하나다 — 그렇게 잰 확률이 **어느 판이 들어가고
// 어느 판이 안 들어가는지 가르는가.**
//
//   node tools/fit_probability.js [판 수]

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const { chance } = require(path.join(ROOT, 'tools', 'make_probability.js'));

const rounds = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l))
  .filter((r) => r.reached && r.scored !== null && r.scored !== undefined);

const N = Number(process.argv[2] || 400);
const step = Math.max(1, Math.floor(rounds.length / N));
const pick = [];
for (let i = 0; i < rounds.length && pick.length < N; i += step) pick.push(rounds[i]);

const lines = pick.map((one) => {
  const b = one.found.find((x) => x.key === one.chose);
  return b && b.deg !== undefined
    ? { layout: one.layout, cue: one.cue, scored: one.scored, match: one.match,
        line: { deg: b.deg, speed: 0, side: b.side, up: b.up, strength: b.strength } }
    : null;
}).filter(Boolean);

const { SIM } = require(path.join(ROOT, 'tools', 'make_probability.js'));
for (const x of lines) x.line.speed = SIM.speedFor(x.line.strength);

const truth = lines.reduce((s, x) => s + (x.scored ? 1 : 0), 0) / lines.length;
console.log(`프로가 실제로 친 그 공략 ${lines.length}판 · 실제 득점률 ${(truth * 100).toFixed(1)}%\n`);

function auc(score, label) {
  const order = score.map((s, i) => [s, label[i]]).sort((a, b) => a[0] - b[0]);
  const pos = label.filter(Boolean).length, neg = label.length - pos;
  if (!pos || !neg) return NaN;
  let seen = 0, sum = 0;
  for (const [, t] of order) { if (t) sum += seen; else seen++; }
  return sum / (pos * neg);
}

console.log('오차 크기        평균 확률   실제와 차이   가르는 정도(AUC)');
const label = lines.map((x) => (x.scored ? 1 : 0));
let best = null;
for (const aim of [0.1, 0.2, 0.3, 0.5, 0.8]) {
  const wobble = { aim, speed: 0.08, tip: 0.3 };
  const t0 = Date.now();
  const p = lines.map((x, i) => chance(x.layout, x.cue, x.line, wobble, 40, i + 1));
  const mean = p.reduce((a, b) => a + b, 0) / p.length;
  const a = auc(p, label);
  console.log(`  겨냥 ${aim.toFixed(1)}도      ${(mean * 100).toFixed(1)}%      `
    + `${((mean - truth) * 100).toFixed(1).padStart(6)}점      ${a.toFixed(3)}`
    + `   (${((Date.now() - t0) / 1000).toFixed(0)}초)`);
  if (!best || Math.abs(mean - truth) < Math.abs(best.mean - truth)) best = { aim, mean, auc: a };
}
console.log(`\n실제 득점률에 가장 가까운 것: 겨냥 오차 ${best.aim}도 · 가르는 정도 ${best.auc.toFixed(3)}`);
console.log('비교: 지금 쓰는 특징 열 개를 전부 합치면 0.650, 줄두께 하나로는 0.551');
