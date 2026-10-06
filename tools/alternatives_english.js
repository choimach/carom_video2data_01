// 열거한 줄마다 1적구 뒤 첫 쿠션에서의 회전(정회전 running / 역회전 reverse / 없음 none)을 다시 쳐서 적는다
// (2026-10-07). 선수가 "당점 좌우 반대"라고 한 다섯 번이 모두 역회전 줄이었다. 대표 줄 고르기로는 못 고친다 —
// 같은 길 안에서는 당점이 거의 정해져 있어, 정회전은 **다른 길(다른 후보)**에 있다. 그래서 순위의 특징으로.
//
//   node tools/alternatives_english.js        # → data/alternatives_english.json  {id: ["running", ...]}
const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

function englishAfterFirst(layout, cue, b) {
  if (Math.abs(b.side || 0) < 0.5) return 'none';
  const speed = SIM.speedFor(b.strength), rad = b.deg * Math.PI / 180;
  const shot = SIM.play(layout, cue, [Math.cos(rad) * speed, Math.sin(rad) * speed], b.side || 0, b.up || 0);
  const ball = shot.events.find((e) => e.kind === 'ball');
  const c = shot.events.find((e) => e.kind === 'cushion' && (!ball || e.at > ball.at));
  return c ? c.english : 'none';
}

const out = {};
let n = 0;
for (const line of fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const one = JSON.parse(line);
  out[one.id] = (one.found || []).map((b) => englishAfterFirst(one.layout, one.cue, b));
  if (++n % 500 === 0) console.error(`  ${n}판`);
}
fs.writeFileSync(path.join(ROOT, 'data', 'alternatives_english.json'), JSON.stringify(out));
console.log(`${n}판 -> data/alternatives_english.json`);
module.exports = { englishAfterFirst };
