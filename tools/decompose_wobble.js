// 어느 오차가 공략을 무너뜨리는가 — 하나씩만 주고 본다.
//
// 2026-09-24 실측 (프로가 친 줄 200개, 실제 득점률 61%):
//
//     아무 오차 없음      100.0%
//     겨냥만 0.05도        75.1%      강도만 2%    81.4%     당점만 0.1팁  67.4%
//     겨냥만 0.1도         65.1%      강도만 5%    71.9%     당점만 0.2팁  51.7%
//     겨냥만 0.2도         56.0%      강도만 8%    64.8%     당점만 0.3팁  41.4%
//     겨냥만 0.4도         43.5%      강도만 15%   54.4%     당점만 0.5팁  31.0%
//
// ★당점이 가장 파괴적이다. 0.3팁은 1.5 mm인데 득점이 41%로 떨어진다.
// ★셋을 같이 주면 곱해진다 — 사람다운 크기(0.1도·8%·0.3팁)에서 32%가 되는데
//   프로는 실제로 61%를 넣는다. **시뮬레이터가 실제보다 두 배 어렵다.**
//
//   node tools/decompose_wobble.js
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const { chance, SIM } = require(path.join(ROOT, 'tools', 'make_probability.js'));
const rounds = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l)).filter((r) => r.reached);
const N = 200, step = Math.max(1, Math.floor(rounds.length / N));
const lines = [];
for (let i = 0; i < rounds.length && lines.length < N; i += step) {
  const one = rounds[i];
  const b = one.found.find((x) => x.key === one.chose);
  if (!b || b.deg === undefined) continue;
  lines.push({ layout: one.layout, cue: one.cue,
    line: { deg: b.deg, speed: SIM.speedFor(b.strength), side: b.side, up: b.up } });
}
const mean = (w, tries) => {
  let s = 0;
  lines.forEach((x, i) => { s += chance(x.layout, x.cue, x.line, w, tries, i + 1); });
  return s / lines.length;
};
console.log(`프로가 친 줄 ${lines.length}개 · 실제 득점률은 61%\n`);
console.log('오차를 하나씩만 주면 (나머지는 0)');
console.log(`  아무 오차 없음               ${(mean({aim:0,speed:0,tip:0}, 4) * 100).toFixed(1)}%`);
for (const a of [0.05, 0.1, 0.2, 0.4])
  console.log(`  겨냥만 ${a}도                ${(mean({aim:a,speed:0,tip:0}, 40) * 100).toFixed(1)}%`);
for (const s of [0.02, 0.05, 0.08, 0.15])
  console.log(`  강도만 ${(s*100).toFixed(0)}%                 ${(mean({aim:0,speed:s,tip:0}, 40) * 100).toFixed(1)}%`);
for (const t of [0.1, 0.2, 0.3, 0.5])
  console.log(`  당점만 ${t}팁                ${(mean({aim:0,speed:0,tip:t}, 40) * 100).toFixed(1)}%`);
